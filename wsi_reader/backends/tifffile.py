import numpy as np
import re
import tifffile
import xml.etree.ElementTree as ET

from fractions import Fraction
from functools import cached_property
from os import PathLike
from typing import IO, Any

from ..base import WSIReader


class TiffReader(WSIReader):
    """Implementation of the WSIReader interface backed by tifffile."""

    def __init__(
        self, slide: str | PathLike[Any] | tifffile.FileHandle | IO[bytes], series: int = 0, **kwargs
    ) -> None:
        """Opens a slide. The object may be used as a context manager, in which case it will be closed upon exiting the context.

        Args:
            slide (str | PathLike[Any] | tifffile.FileHandle | IO[bytes]): Slide to open.
            series (int, optional): For multi-series formats, image series to open. Defaults to 0.
        """
        self.series = series
        self.slide = slide
        self._tiff = tifffile.TiffReader(slide)

    def close(self) -> None:
        self._tiff.close()

    @cached_property
    def tile_dimensions(self) -> tuple[tuple[int, int], ...]:
        tile_dimensions = []
        for level in range(self.level_count):
            tile_h, tile_w = self._tiff.series[self.series].levels[0].keyframe.chunks[:2]
            tile_dimensions.append((tile_w, tile_h))
        return tuple(tile_dimensions)

    def _read_region(self, x_y: tuple[int, int], level: int, tile_size: tuple[int, int]):
        page = self._tiff.series[self.series].levels[level].keyframe

        if not page.is_tiled:
            raise ValueError("Image must be tiled.")

        x_start, y_start = x_y
        x_end, y_end = x_start + tile_size[0], y_start + tile_size[1]

        tile_x_start, tile_y_start = (
            x_start // page.tilewidth,
            y_start // page.tilelength,
        )
        tile_x_end, tile_y_end = (x_end - 1) // page.tilewidth + 1, (
            y_end - 1
        ) // page.tilelength + 1

        tiles_per_row = page.imagewidth // page.tilewidth + 1

        out = np.zeros(
            (
                page.imagedepth,
                (tile_y_end - tile_y_start) * page.tilelength,
                (tile_x_end - tile_x_start) * page.tilewidth,
                page.samplesperpixel,
            ),
            dtype=page.dtype,
        )

        fh = page.parent.filehandle

        for tile_y in range(tile_y_start, tile_y_end):
            for tile_x in range(tile_x_start, tile_x_end):
                index = int(tile_y * tiles_per_row + tile_x)

                offset = page.dataoffsets[index]
                bytecount = page.databytecounts[index]

                if bytecount > 0:
                    fh.seek(offset)
                    data = fh.read(bytecount)

                    tile, _, _ = page.decode(
                        data,
                        index,
                        jpegtables=page.jpegtables,
                        jpegheader=page.jpegheader,
                    )
                    out_x_start = (tile_x - tile_x_start) * page.tilewidth
                    out_y_start = (tile_y - tile_y_start) * page.tilelength
                    out_x_end = out_x_start + page.tilewidth
                    out_y_end = out_y_start + page.tilelength
                    out[:, out_y_start:out_y_end, out_x_start:out_x_end, :] = tile

        out_x_start = x_start - tile_x_start * page.tilewidth
        out_y_start = y_start - tile_y_start * page.tilelength
        out_x_end = out_x_start + tile_size[0]
        out_y_end = out_y_start + tile_size[1]

        return out[0, out_y_start:out_y_end, out_x_start:out_x_end]

    @cached_property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        level_dimensions = []
        for level in self._tiff.series[self.series].levels:
            height, width = level.shape[:2]
            level_dimensions.append((width, height))
        return tuple(level_dimensions)

    @cached_property
    def level_downsamples(self) -> tuple[float, ...]:
        level_downsamples = []
        width, height = self.level_dimensions[0]
        for level in range(len(self.level_dimensions)):
            w, h = self.level_dimensions[level]
            ds = float(round(width / w))
            level_downsamples.append(ds)
        return tuple(level_downsamples)

    @property
    def level_count(self) -> int:
        return len(self._tiff.series[self.series].levels)

    @cached_property
    def mpp(self) -> tuple[float | None, float | None]:
        mpp: tuple[float | None, float | None] = (None, None)
        page = self._tiff.series[self.series].keyframe
        if page.is_svs:
            metadata = tifffile.tifffile.svs_description_metadata(
                page.description
            )
            mpp = (metadata["MPP"], metadata["MPP"])
        elif page.is_ome:
            root = ET.fromstring(page.description)
            namespace_match = re.search("^{.*}", root.tag)
            namespace = namespace_match.group() if namespace_match else ""
            pixels = list(root.findall(namespace + "Image"))[self.series].find(
                namespace + "Pixels"
            )
            mpp_x = pixels.get("PhysicalSizeX") if pixels else None
            mpp_y = pixels.get("PhysicalSizeY") if pixels else None
            mpp = (
                float(mpp_x) if mpp_x else None,
                float(mpp_y) if mpp_y else None,
            )
        elif page.is_philips:
            root = ET.fromstring(page.description)
            mpp_attribute = root.find(
                "./Attribute/[@Name='PIM_DP_SCANNED_IMAGES']/Array/DataObject/[@ObjectType='DPScannedImage']/Attribute/[@Name='PIM_DP_IMAGE_TYPE'][.='WSI']/Attribute[@Name='PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE']/Array/DataObject[@ObjectType='PixelDataRepresentation']/Attribute[@Name='DICOM_PIXEL_SPACING']"
            )
            _mpp = (
                float(mpp_attribute.text)
                if mpp_attribute and mpp_attribute.text
                else None
            )
            mpp = (_mpp, _mpp)
        elif page.is_ndpi or page.is_scn or page.is_qpi or True:
            if (
                "ResolutionUnit" in page.tags
                and page.tags["ResolutionUnit"].value == 3
                and "XResolution" in page.tags
                and "YResolution" in page.tags
            ):
                mpp = (
                    1e4 / float(Fraction(*page.tags["XResolution"].value)),
                    1e4 / float(Fraction(*page.tags["YResolution"].value)),
                )
        return mpp

    @property
    def dtype(self) -> np.dtype:
        return self._tiff.series[self.series].dtype

    @property
    def n_channels(self) -> int:
        return self._tiff.series[self.series].shape[2]

    @cached_property
    def bounds(self) -> dict:
        height, width = self._tiff.series[self.series].shape[:2]
        bounds = {
            "type": "MultiPolygon", 
            "coordinates": [
                [
                    [
                        [0, 0],
                        [width, 0],
                        [width, height],
                        [0, height],
                        [0, 0]
                    ]
                ]
            ]
        }

        return bounds
    
    def __eq__(self, other) -> bool:
        return isinstance(other, TiffReader) and self.slide == other.slide and self.series == other.series
    
    def __getstate__(self):
        return {'slide': self.slide, 'series': self.series}
    
    def __setstate__(self, state):
        self.__init__(state['slide'], series=state['series'])