import numpy as np
import openslide

from functools import cached_property
from os import PathLike
from typing import Optional

from ..base import WSIReader


class OpenslideReader(WSIReader):
    """Implementation of the WSIReader interface backed by openslide"""

    def __init__(self, slide_path: PathLike | str, **kwargs) -> None:
        """Opens a slide. The object may be used as a context manager, in which case it will be closed upon exiting the context.

        Args:
            slide_path (PathLike | str): Path of the slide to open.
        """

        self.slide_path = slide_path
        self._slide = openslide.open_slide(str(slide_path))

    def close(self) -> None:
        self._slide.close()

    def _read_region(
        self, x_y: tuple[int, int], level: int, tile_size: tuple[int, int]
    ) -> np.ndarray:
        ds = self.level_downsamples[level]
        x, y = x_y
        x_y = (round(x * ds), round(y * ds))
        tile = np.array(self._slide.read_region(x_y, level, tile_size), dtype=np.uint8)
        tile[:, :, 3][tile[:, :, 3] < 255] = 0
        return tile

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self._slide.get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return self._slide.level_dimensions

    @property
    def level_count(self) -> int:
        return self._slide.level_count

    @cached_property
    def mpp(self) -> tuple[float | None, float | None]:
        mpp_x = self._slide.properties.get("openslide.mpp-x")
        mpp_x = mpp_x and float(mpp_x)
        mpp_y = self._slide.properties.get("openslide.mpp-y")
        mpp_y = mpp_y and float(mpp_y)
        return mpp_x, mpp_y

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8)

    @property
    def n_channels(self) -> int:
        return 4

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self._slide.level_downsamples

    @cached_property
    def tile_dimensions(self) -> tuple[tuple[int, int], ...]:
        tile_dimensions = []
        for level in range(self.level_count):
            tile_width = int(
                self._slide.properties[f"openslide.level[{level}].tile-width"]
            )
            tile_height = int(
                self._slide.properties[f"openslide.level[{level}].tile-height"]
            )
            tile_dimensions.append((tile_width, tile_height))
        return tuple(tile_dimensions)

    @cached_property
    def bounds(self) -> dict:
        x = self._slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0)
        y = self._slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0)
        width = self._slide.properties.get(
            openslide.PROPERTY_NAME_BOUNDS_WIDTH, self.level_dimensions[0][0]
        )
        height = self._slide.properties.get(
            openslide.PROPERTY_NAME_BOUNDS_HEIGHT, self.level_dimensions[0][1]
        )
        bounds = {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [x, y],
                        [x + width, y],
                        [x + width, y + height],
                        [x, y + height],
                        [x, y],
                    ]
                ]
            ],
        }

        return bounds

    def __getstate__(self):
        return {"slide_path": self.slide_path}

    def __setstate__(self, state):
        self.__init__(state["slide_path"])
