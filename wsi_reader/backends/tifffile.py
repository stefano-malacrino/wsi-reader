import numpy as np
import re
import tifffile
import xml.etree.ElementTree as ET
import zarr

from fractions import Fraction
from functools import cached_property
from os import PathLike
from typing import Optional

from ..base import WSIReader, FileLike


class TiffReader(WSIReader):
    """Implementation of the WSIReader interface backed by tifffile."""

    def __init__(
        self, slide: PathLike | str | FileLike, series: int = 0, **kwargs
    ) -> None:
        """Opens a slide. The object may be used as a context manager, in which case it will be closed upon exiting the context.

        Args:
            slide (PathLike | str | FileLike): Slide to open.
            series (int, optional): For multi-series formats, image series to open. Defaults to 0.
        """
        self._series = series
        self._slide = slide
        self._store: tifffile.ZarrTiffStore = tifffile.imread(slide, aszarr=True, series=series)
        assert isinstance(self._store, tifffile.ZarrTiffStore)
        try:
            self._zarr: zarr.Group = zarr.open(self._store, mode="r")
            assert isinstance(self._zarr, zarr.Group)
        except:
            self._store.close()
            raise

    def close(self) -> None:
        self._store.close()

    @cached_property
    def tile_dimensions(self) -> tuple[tuple[int, int], ...]:
        tile_dimensions = []
        for level in range(self.level_count):
            tile_h, tile_w = self._zarr[level].chunks[:2]
            tile_dimensions.append((tile_w, tile_h))
        return tuple(tile_dimensions)

    def _read_region(
        self, x_y: tuple[int, int], level: int, tile_size: tuple[int, int]
    ) -> np.ndarray:
        x, y = x_y
        tile_w, tile_h = tile_size
        return self._zarr[level][y : y + tile_h, x : x + tile_w]

    @cached_property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        level_dimensions = []
        for level in range(len(self._zarr)):
            height, width = self._zarr[level].shape[:2]
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
        return len(self._zarr)

    @cached_property
    def mpp(self) -> tuple[Optional[float], Optional[float]]:
        mpp: tuple[Optional[float], Optional[float]] = (None, None)
        page: tifffile.TiffPage = self._store._data[0].pages[0]
        if page.is_svs:
            metadata = tifffile.tifffile.svs_description_metadata(
                page.description
            )
            mpp = (metadata["MPP"], metadata["MPP"])
        elif page.is_ome:
            root = ET.fromstring(page.description)
            namespace_match = re.search("^{.*}", root.tag)
            namespace = namespace_match.group() if namespace_match else ""
            pixels = list(root.findall(namespace + "Image"))[self._series].find(
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
        return self._zarr[0].dtype

    @property
    def n_channels(self) -> int:
        return self._zarr[0].shape[2]

    @cached_property
    def bounds(self) -> dict:
        height, width = self._zarr[0].shape[:2]
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
    
    def __getstate__(self):
        return {'slide': self._slide, 'series': self._series}
    
    def __setstate__(self, state):
        self.__init__(state['slide'], series=state['series'])