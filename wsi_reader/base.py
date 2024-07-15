import cv2
import numpy as np

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from enum import Enum, auto
from functools import cached_property
from os import SEEK_SET
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class FileLike(Protocol):
    def read(self, n: Optional[int]) -> int: ...
    def seek(self, offset: int, whence: int = SEEK_SET): ...
    def tell(self) -> int: ...


class Resolution(Enum):
    LEVEL = auto()
    DOWNSAMPLE = auto()


class Interpolation(Enum):
    NEAREST = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC


class DownsampleDimensions(Mapping[float, tuple[int, int]]):
    def __init__(
        self,
        level_downsamples: tuple[float, ...],
        level_dimensions: tuple[tuple[int, int], ...],
    ):
        self._level_downsamples = level_downsamples
        self._level_dimensions = level_dimensions
        super().__init__()

    def __getitem__(self, downsample: float) -> tuple[int, int]:
        if downsample <= 0:
            raise ValueError("Invalid downsample factor")
        try:
            level = self._level_downsamples.index(downsample)
            return self._level_dimensions[level]
        except ValueError:
            w, h = self._level_dimensions[0]
            return round(w / downsample), round(h / downsample)

    def __iter__(self):
        return iter(self._level_downsamples)

    def __len__(self):
        return len(self._level_downsamples)


class WSIReader(metaclass=ABCMeta):
    """Interface class for a WSI reader."""

    @abstractmethod
    def close(self) -> None:
        """Closes the slide.

        Returns:
            None
        """
        raise NotImplementedError

    def __enter__(self) -> "WSIReader":
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self.close()

    def read_region(
        self,
        x_y: tuple[int, int],
        resolution: int | float,
        tile_size: tuple[int, int] | int,
        normalize: bool = False,
        downsample_base_res: bool = False,
        resolution_unit: Resolution = Resolution.LEVEL,
        interpolation: Interpolation = Interpolation.BICUBIC,
    ) -> np.ndarray:
        """Reads the contents of a region in the slide from the specified resolution. Resolution can be expressed as level or downsample factor.

        Args:
            x_y (tuple[int, int]): coordinates of the top left pixel of the region in the specified resolution reference frame.
            resolution (int | float): the desired resolution.
            tile_size (tuple[int, int] | int): size of the region. Can be a tuple in the format (width, height) or a scalar to specify a square region.
            normalize (bool, optional): True to normalize the pixel values in the range [0,1]. Defaults to False.
            downsample_base_res (bool, optional): True to render the region by downsampling from the base (highest) resolution. Defaults to False.
            resolution_unit (Resolution): resolution unit. Defaults to Resolution.LEVEL.
            interpolation (Interpolation): interpolation method to use for downsampling. Defaults to Interpolation.BICUBIC.

        Returns:
            np.ndarray: pixel data of the specified region.
        """
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)

        if resolution_unit == Resolution.LEVEL:
            if not isinstance(resolution, int):
                raise ValueError("Level must be an integer")

            if downsample_base_res:
                resolution = self.level_downsamples[resolution]
                res = self._read_region_downsample(
                    x_y, resolution, tile_size, True, interpolation
                )
            else:
                res = self._read_region_level(x_y, resolution, tile_size)
        elif resolution_unit == Resolution.DOWNSAMPLE:
            res = self._read_region_downsample(
                x_y, resolution, tile_size, downsample_base_res, interpolation
            )
        else:
            raise ValueError("Invalid value for resolution_unit")

        if normalize:
            res = self._normalize(res)

        return res

    def _get_region_coords(
        self, start: int, tile_size: int, dim_size: int
    ) -> tuple[int, int, tuple[int, int]]:
        end = start + tile_size
        if start >= dim_size or end <= 0:
            raise ValueError("Region out of image bounds")
        pad_start = abs(min(start, 0))
        start += pad_start
        pad_end = max(end - dim_size, 0)
        end -= pad_end
        tile_size = end - start
        return start, tile_size, (pad_start, pad_end)

    def _read_region_level(
        self,
        x_y: tuple[int, int],
        level: int,
        tile_size: tuple[int, int],
    ) -> np.ndarray:

        if level < 0:
            raise ValueError("Level must be non negative")

        x_raw, y_raw = x_y
        tile_w_raw, tile_h_raw = tile_size
        width, height = self.level_dimensions[level]

        x, tile_w, pad_w = self._get_region_coords(x_raw, tile_w_raw, width)
        y, tile_h, pad_h = self._get_region_coords(y_raw, tile_h_raw, height)

        tile = self._read_region((x, y), level, (tile_w, tile_h))
        tile = np.pad(
            tile,
            (pad_h, pad_w, *(((0, 0),) * (tile.ndim - 2))),
            "constant",
            constant_values=0,
        )

        return tile

    def _read_region_downsample(
        self,
        x_y: tuple[int, int],
        downsample: float,
        tile_size: tuple[int, int],
        downsample_base_res: bool,
        interpolation: Interpolation,
    ) -> np.ndarray:
        if downsample <= 0:
            raise ValueError("Downsample factor must be positive")

        if downsample == 1:
            downsample_base_res = False

        if not downsample_base_res and downsample in self.level_downsamples:
            level = self.get_best_level_for_downsample(downsample)
            tile = self._read_region_level(x_y, level, tile_size)
        else:
            level = (
                0
                if downsample_base_res
                else self.get_best_level_for_downsample(downsample)
            )
            x_y_level = (
                round(x_y[0] * downsample / self.level_downsamples[level]),
                round(x_y[1] * downsample / self.level_downsamples[level]),
            )
            tile_size_level = (
                round(tile_size[0] * downsample / self.level_downsamples[level]),
                round(tile_size[1] * downsample / self.level_downsamples[level]),
            )
            tile = self._read_region_level(x_y_level, level, tile_size_level)
            tile = cv2.resize(tile, tile_size, interpolation=interpolation.value)

        return tile

    @abstractmethod
    def _read_region(
        self, x_y: tuple[int, int], level: int, tile_size: tuple[int, int]
    ) -> np.ndarray:
        raise NotImplementedError

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Returns the best resolution level for the given downsample factor.

        Args:
            downsample (float): the downsample factor.

        Returns:
            int: the resolution level.
        """
        if downsample < self.level_downsamples[0]:
            return 0

        for i in range(1, self.level_count):
            if downsample < self.level_downsamples[i]:
                return i - 1

        return self.level_count - 1

    def get_downsampled_slide(
        self,
        dims: tuple[int, int],
        normalize: bool = False,
        interpolation: Interpolation = Interpolation.BICUBIC,
    ) -> np.ndarray:
        """Returns a downsampled version of the slide with the given dimensions.

        Args:
            dims (tuple[int, int]): size of the downsampled slide asa (width, height) tuple.
            normalize (bool, optional): True to normalize the pixel values in therange [0,1]. Defaults to False.
            interpolation (Interpolation): interpolation method to use for downsampling. Defaults to Interpolation.BICUBIC.

        Returns:
            np.ndarray: pixel data of the downsampled slide.
        """
        downsample = min(a / b for a, b in zip(self.level_dimensions[0], dims))
        slide_downsampled = self.read_region(
            (0, 0),
            downsample,
            self.downsample_dimensions[downsample],
            normalize=normalize,
            resolution_unit=Resolution.DOWNSAMPLE,
            interpolation=interpolation,
        )
        return slide_downsampled

    @cached_property
    def downsample_dimensions(self) -> Mapping[float, tuple[int, int]]:
        """A mapping between downsample factors and slide dimensions."""
        return DownsampleDimensions(self.level_downsamples, self.level_dimensions)

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Number of resolution levels in the slide."""
        raise NotImplementedError

    @property
    @abstractmethod
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        """Slide dimensions for each slide level as a tuple of (width, height) tuples."""
        raise NotImplementedError

    @property
    @abstractmethod
    def tile_dimensions(self) -> tuple[tuple[int, int], ...]:
        """Tile dimensions for each slide level as a tuple of (width, height) tuples."""
        raise NotImplementedError

    @property
    @abstractmethod
    def mpp(self) -> tuple[Optional[float], Optional[float]]:
        """A tuple containing the number of microns per pixel of level 0 in the X and Y dimensions respectively, if known."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Numpy data type of the slide pixels."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of channels in the slide."""
        raise NotImplementedError

    @property
    @abstractmethod
    def level_downsamples(self) -> tuple[float, ...]:
        """A tuple of downsample factors for each resolution level of the slide."""
        raise NotImplementedError

    @property
    @abstractmethod
    def bounds(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def _normalize(pixels: np.ndarray) -> np.ndarray:
        if np.issubdtype(pixels.dtype, np.integer):
            pixels = (pixels / 255).astype(np.float32)
        return pixels

    @abstractmethod
    def __getstate__(self):
        raise NotImplementedError

    @abstractmethod
    def __setstate__(self, state):
        raise NotImplementedError
