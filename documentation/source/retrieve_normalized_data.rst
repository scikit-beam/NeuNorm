************************
Retrieve Normalized Data
************************

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