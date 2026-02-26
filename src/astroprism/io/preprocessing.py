"""
preprocessing.py

Functions for preprocessing data and PSF files into a single dataset.
"""

# === Imports ======================================================================================

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# === Main =========================================================================================

def append_psf_extension(data_file: str, psf_file: str, output_file: str, overwrite: bool = False) -> fits.HDUList:

    # Load data file
    with fits.open(data_file) as data_hdul:
        hdul = fits.HDUList([hdu.copy() for hdu in data_hdul])

    # Load PSF file
    psf = np.load(psf_file)
    psf_x = psf["x_arcsec"]
    psf_y = psf["y_arcsec"]
    psf_flux = psf["psf_data"]

    # Append PSF extension to data file
    psf_hdul = fits.ImageHDU(psf_flux, name="PSF")
    hdul.append(psf_hdul)

    # Create a parent directory for the output file if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to output file
    hdul.writeto(output_file, overwrite=overwrite)

    return hdul

