import cv2
import numpy as np

from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Protocol, runtime_checkable
from typing_extensions import Buffer


@runtime_checkable
class FileLike(Protocol):
    def read(self, n: int | None = ..., /) -> Buffer: ...
    def seek(self, offset: int, whence: int = ..., /) -> int: ...
    def tell(self) -> int: ...


class Resolution(Enum):
    LEVEL = auto()
    DOWNSAMPLE = auto()
    MPP = auto()


class Interpolation(Enum):
    NEAREST = cv2.INTER_NEAREST
    BILINEAR = cv2.INTER_LINEAR
    BICUBIC = cv2.INTER_CUBIC


def _get_region_coords(
        start: int, tile_size: int, dim_size: int
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


def _normalize(pixels: np.ndarray) -> np.ndarray:
        if np.issubdtype(pixels.dtype, np.integer):
            pixels = (pixels / 255).astype(np.float32)
        return pixels


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
            normalize (bool, optional): True: normalize the pixel values in the range [0,1]. Defaults to False.
            downsample_base_res (bool, optional): False: render the region by downsampling from the base (highest) resolution. Defaults to False.
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
                ds = self.level_downsamples[resolution]
                res = self._read_region_downsample(
                    x_y, ds, tile_size, True, interpolation
                )
            else:
                res = self._read_region_level(x_y, resolution, tile_size)
        elif resolution_unit == Resolution.DOWNSAMPLE:
            res = self._read_region_downsample(
                x_y, resolution, tile_size, downsample_base_res, interpolation
            )
        elif resolution_unit == Resolution.MPP:
            mpp = self.mpp[0] or self.mpp[1]
            if not mpp:
                raise ValueError("No mpp information available")
            ds = resolution / mpp
            res = self._read_region_downsample(
                x_y, ds, tile_size, downsample_base_res, interpolation
            )
        else:
            raise ValueError("Invalid value for resolution_unit")

        if normalize:
            res = _normalize(res)

        return res

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

        x, tile_w, pad_w = _get_region_coords(x_raw, tile_w_raw, width)
        y, tile_h, pad_h = _get_region_coords(y_raw, tile_h_raw, height)

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

    def downsample_dimensions(self, downsample: float) -> tuple[int, int]:
        """Get slide dimensions for a downsaple factor.

        Args:
            downsample (float): the downsample factor.

        Returns:
            tuple[int, int]: slide dimensions.
        """
        try:
            level = self.level_downsamples.index(downsample)
            return self.level_dimensions[level]
        except ValueError:
            w, h = self.level_dimensions[0]
            return round(w / downsample), round(h / downsample)
        
    def mpp_dimensions(self, mpp: float) -> tuple[int, int]:
        """Slide dimensions for a mpp value.
        Args:
            downsample (float): the downsample factor.

        Returns:
            tuple[int, int]: slide dimensions.
        """
        if mpp <= 0:
            raise ValueError("Invalid mpp")
        mpp_x, mpp_y = self.mpp
        if mpp_x is None or mpp_y is None:
            raise ValueError("No mpp information available")
        ds_w = mpp / mpp_x
        ds_h = mpp / mpp_y
        try:
            if ds_w == ds_h:
                level = self.level_downsamples.index(ds_w)
                return self.level_dimensions[level]
        except ValueError:
            pass
        w, h = self.level_dimensions[0]
        return round(w / ds_w), round(h / ds_h)

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
    def mpp(self) -> tuple[float | None, float | None]:
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

    @abstractmethod
    def __getstate__(self):
        raise NotImplementedError

    @abstractmethod
    def __setstate__(self, state):
        raise NotImplementedError
