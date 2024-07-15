# wsi_reader package

wsi-reader is a format-independent WSI reader. The API is designed to be similar to openslide.

Backends based on the following libraries are available: tifffile, openslide and the Philips pathology SDK.
The reader works with all the image formats supported by the installed backends.

To install the package from the latest commit with a desired backend (e.g. tifffile) use the following command:

> pip install "wsi-reader[tifffile] @ https://github.com/stefano-malacrino/wsi-reader.git"
