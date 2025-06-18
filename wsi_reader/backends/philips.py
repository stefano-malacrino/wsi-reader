import numpy as np

from collections.abc import Generator
from contextlib import contextmanager
from functools import cached_property
from os import PathLike
from queue import Queue

from pixelengine import PixelEngine
from softwarerendercontext import SoftwareRenderContext
from softwarerenderbackend import SoftwareRenderBackend

from ..base import WSIReader, FileLike


class _PixelEngineCache:
    def __init__(
        self, size: int, slide: str | FileLike, container_name, cache_path: str
    ):
        self._queue = Queue(size)
        self._pe_args = [slide, container_name, "r", cache_path]
        self._closed = False

        for _ in range(size):
            self._queue.put(None)

    def _put(self, pe: PixelEngine) -> None:
        self._queue.put(pe)

    def _get(self) -> PixelEngine:
        pe = self._queue.get()
        if pe is None:
            pe = PixelEngine(SoftwareRenderBackend(), SoftwareRenderContext())
            pe["in"].open(*self._pe_args)
            trunc_bits = {0: [0, 0, 0]}
            pe["in"]["WSI"].source_view.truncation(False, False, trunc_bits)
        return pe

    def close(self) -> None:
        if not self._closed:
            for _ in range(self._queue.maxsize):
                pe = self._queue.get()
                if pe is not None:
                    pe["in"].close()
            self._closed = True

    @contextmanager
    def get(self) -> Generator[PixelEngine, None, None]:
        pe = None
        try:
            pe = self._get()
            yield pe
        finally:
            self._put(pe)


class IsyntaxReader(WSIReader):
    """Implementation of the WSIReader interface for the isyntax format backed by the Philips pathology SDK."""

    def __init__(
        self,
        slide: FileLike | PathLike | str,
        cache_path: PathLike | str | None = "",
        pe_cache_size: int = 8,
        **kwargs
    ) -> None:
        """Opens a slide. The object may be used as a context manager, in which case it will be closed upon exiting the context.

        Args:
            slide (FileLike | PathLike | str): Slide to open.
            cache_path (PathLike | str | None, optional): Path to the cache for the image. If it's a path to a folder the file name is assumed to be the same as the image. If an empty string the path is set to the directory where the image is stored. If None caching is disabled. Defaults to "".
            pe_cache_size (int, optional): Size of cache of PixelEngine handles (used for thread-safety). Defaults to 8.
        """

        if isinstance(slide, PathLike) or isinstance(slide, str):
            slide = str(slide)
            container_name = "caching-ficom" if cache_path is not None else "ficom"
            cache_path = str(cache_path) if cache_path is not None else ""
        else:
            container_name = "ficom"
            cache_path = ""

        self._pe_cache = _PixelEngineCache(
            pe_cache_size,
            slide,
            container_name,
            cache_path,
        )
        with self._pe_cache.get() as pe:
            pass

        self.slide = slide
        self._cache_path = cache_path
        self._pe_cache_size = pe_cache_size

    def close(self) -> None:
        self._pe_cache.close()

    @cached_property
    def tile_dimensions(self) -> tuple[tuple[int, int], ...]:
        level_count = self.level_count
        with self._pe_cache.get() as pe:
            tile_w, tile_h = pe["in"]["WSI"].block_size()[:2]
            tile_dimensions = ((tile_w, tile_h),) * level_count
        return tile_dimensions

    def _read_region(
        self, x_y: tuple[int, int], level: int, tile_size: tuple[int, int]
    ) -> np.ndarray:
        x_start, y_start = x_y
        ds = self.level_downsamples[level]
        tile_w, tile_h = tile_size
        x_start = round(x_start * ds)
        y_start = round(y_start * ds)
        x_end = x_start + round((tile_w - 1) * ds)
        y_end = y_start + round((tile_h - 1) * ds)
        view_range = [x_start, x_end, y_start, y_end, level]
        tile = np.empty(np.prod(tile_size) * 4, dtype=np.uint8)
        with self._pe_cache.get() as pe:
            view = pe["in"]["WSI"].source_view
            regions = view.request_regions(
                [view_range],
                view.data_envelopes(level),
                True,
                [255, 255, 255, 0],
                PixelEngine.BufferType.RGBA,
            )
            (region,) = regions
            pe.wait_all(regions)
            region.get(tile)
        tile.shape = (tile_h, tile_w, 4)
        return tile

    @cached_property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        level_count = self.level_count
        level_dimensions = []
        with self._pe_cache.get() as pe:
            view = pe["in"]["WSI"].source_view
            x_end = view.dimension_ranges(0)[0][2] + 1
            y_end = view.dimension_ranges(0)[1][2] + 1
            for level in range(level_count):
                x_step = view.dimension_ranges(level)[0][1]
                y_step = view.dimension_ranges(level)[1][1]
                range_x = round(x_end / x_step)
                range_y = round(y_end / y_step)
                level_dimensions.append((range_x, range_y))
        return tuple(level_dimensions)

    @cached_property
    def level_count(self) -> int:
        with self._pe_cache.get() as pe:
            view = pe["in"]["WSI"].source_view
            level_count = view.num_derived_levels + 1
        return level_count

    @cached_property
    def mpp(self) -> tuple[float | None, float | None]:
        with self._pe_cache.get() as pe:
            view = pe["in"]["WSI"].source_view
            mpp_x, mpp_y = view.scale[:2]
        return (mpp_x, mpp_y)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8)

    @property
    def n_channels(self) -> int:
        return 4

    @cached_property
    def level_downsamples(self) -> tuple[float, ...]:
        level_count = self.level_count
        with self._pe_cache.get() as pe:
            view = pe["in"]["WSI"].source_view
            level_downsamples = [
                float(view.dimension_ranges(level)[0][1])
                for level in range(level_count)
            ]
        return tuple(level_downsamples)

    @cached_property
    def bounds(self) -> dict:
        with self._pe_cache.get() as pe:
            view = pe["in"]["WSI"].source_view
            bounds = {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max],
                            [x_min, y_min],
                        ]
                    ]
                    for (x_min, x_max, y_min, y_max) in view.data_envelopes(
                        0
                    ).as_rectangles()
                ],
            }

        return bounds

    def __getstate__(self):
        return {
            "slide": self.slide,
            "cache_path": self._cache_path,
            "pe_cache_size": self._pe_cache_size,
        }

    def __setstate__(self, state):
        self.__init__(
            state["slide"],
            state["cache_path"],
            state["pe_cache_size"],
        )
