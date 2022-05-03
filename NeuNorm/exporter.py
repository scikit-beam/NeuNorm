from PIL import Image
from astropy.io import fits


def make_tiff(data=[], metadata=[], file_name=''):
    '''create tiff file'''
    new_image = Image.fromarray(data)
    new_image.save(file_name, tiffinfo=metadata)


def make_fits(data=[], file_name=''):
    '''create fits file'''
    fits.writeto(file_name, data, overwrite=True)
