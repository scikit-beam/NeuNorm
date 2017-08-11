*************
Normalization
*************

Normalization using ROI (optional)
##################################

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
####################################

If you don't want any normalization ROI, simply run the normalization

>>> o_norm.normalization()

How to get the normalized data

Each of the data set in the sample and ob will then be normalized.
If a norm_roi has been provided, the sample arrays will be divided by the average of the 
region defined. Same thing for the ob. Those normalized array can be retrieved this way

>>> sample_normalized_array = o_norm.data['sample']['data']
>>> ob_normalized_array = o_gretting.data['ob']['data']
