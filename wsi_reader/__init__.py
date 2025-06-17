"""wsi-reader is a format-independent WSI reader. The API is designed to be similar to openslide.

Backends based on the following libraries are available: tifffile, openslide and the Philips pathology SDK.
The reader works with all the image formats supported by the installed backends.
"""

from os import PathLike
from pathlib import Path

from .base import WSIReader, Resolution, Interpolation


def mpp_to_magnification(mpp: float) -> float:
    """Converts mpp to magnification factor.

    Args:
        mpp (float): mpp value.

    Returns:
        int: Converted magnification factor.
    """
    return {0.2: 40, 0.3: 40, 0.4: 20, 0.5: 20, 1: 10, 2: 5, 4: 2.5, 8: 1.25}[
        round(mpp, 1)
    ]


def open_slide(slide_path: PathLike | str, **reader_kwargs) -> WSIReader:
    """Opens the slide using an implementation of WSIReader interface based on the image file extension. If no suitable implementation is found an exception is thrown.

    Args:
        slide_path (PathLike | str): Path of the image file.

    Returns:
        WSIReader: Opened slide.
    """
    slide_path = Path(slide_path)
    if slide_path.suffix == ".isyntax":
        from .backends.philips import IsyntaxReader

        return IsyntaxReader(slide_path, **reader_kwargs)

    with open(slide_path, "rb") as fh:
        header = fh.read(2)
        if header in (b"II", b"MM", b"EP"):
            from .backends.tifffile import TiffReader

            return TiffReader(slide_path, **reader_kwargs)

    if slide_path.suffix in (".mrxs", ".svslide"):
        from .backends.openslide import OpenslideReader

        return OpenslideReader(slide_path, **reader_kwargs)

    raise ValueError(f"No suitable WSIReader implementation found for {slide_path}")
