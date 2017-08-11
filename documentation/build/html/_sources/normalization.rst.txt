*************
Normalization
*************

Normalization using ROI (optional)
**********************************

If you want to specify a region of your sample to match with the OB

Let's use the following region 

- x0 = 10
- y0 = 10
- x1 = 50
- y1 = 50

>>> my_norm_roi = ROI(x0=10, y0=10, x1=50, y1=50)

then the normalization can be run

>>> o_norm.normalization(norm_roi=my_norm_roi)

Normalization without ROI (optional)
************************************

If you don't want any normalization ROI, simply run the normalization

>>> o_norm.normalization()

How to get the normalized data

Each of the data set in the sample and ob will then be normalized.
If a norm_roi has been provided, the sample arrays will be divided by the average of the 
region defined. Same thing for the ob. Those normalized array can be retrieved this way

>>> sample_normalized_array = o_norm.data['sample']['data']
>>> ob_normalized_array = o_gretting.data['ob']['data']

Cropping the data (optional)
****************************

You have the option to crop the data but if you do, this must be done after running the normalization. 
The algorithm only cropped the normalized sample and ob data

- the 4 corners of the region of interest (ROI)
- the top left corner coordinates, width and height of the ROI

let's use the first method and let's pretend the ROI is defined by

- x0 = 5
- y0 = 5
- x1 = 200
- y1 = 250

>>> my_crop_roi = ROI(x0=5, y0=5, x1=200, y1=250)
>>> o_norm.crop(roi=my_crop_roi)

Full Normalization
==================

The sample/OB normalized data can be recovered this way

>>> normalized_data = neunorm.data['normalized']

Retrieve data
=============

You can retrieve the data using either this way

>>> sample = o_norm.data['sample']['data']
>>> ob = o_norm.data['ob']['data']
>>> df = o_norm.data['df']['data']
>>> norm = o_norm.data['normalization']

or

>>> sample = o_norm.get_sample_data()
>>> ob = o_norm.get_ob_data()
>>> df = o_norm.get_df_data()
>>> normalized = o_norm.get_normalized_data()

Export Data
===========

It is possible to export any of the data you worked on (sample, ob, df or normalized) either
as a 'tif' or as a 'fits' file (default being 'tif')

>>> output_folder = '/users/my_output_folder'
>>> o_norm.export(folder=output_folder, data_type='normalized', file_type='tif')

or if you prefer 'fits'

>>> o_norm.export(folder=output_folder, data_type='normalized', file_type='fits')