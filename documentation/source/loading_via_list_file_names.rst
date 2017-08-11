***************************
Loading via list file names
***************************
  
  
  
  
  
  
Let's pretend that our images are in the folder **/Users/me/sample/** and named 

- image001.fits
- image002.fits
- image003.fits

>>> sample_folder = '/Users/me/sample/'
>>> o_norm = Normalization()
>>> o_norm.load(file='/Users/me/sample/image001.fits')
>>> o_norm.load(file='/Users/me/sample/image002.fits')
>>> o_norm.load(file='/Users/me/sample/image003.fits')

At this point all the data have been loaded in memory and can be accessed as followed

>>> image001 = o_norm.data['sample']['data'][0]
>>> image002 = o_norm.data['sample']['data'][1]

and the file names

>>> image003_file_name = o_norm.data['sample']['file_name'][2]

Let's now load the rest of our data, the OB and the DF

Our OB are in the folder **/Users/me/ob/** and named

- ob001.fits
- ob002.fits
- ob003.fits

>>> o_norm.load(file='/Users/me/ob/ob001.fits', data_type='ob')
>>> o_norm.load(file='/Users/me/ob/ob002.fits', data_type='ob')
>>> o_norm.load(file='/Users/me/ob/ob003.fits', data_type='ob')

again, all the data can be retrieved as followed

>>> ob1 = o_norm.data['ob']['data'][0]
>>> ob2_file_name = o_norm.data['ob']['file_name'][1]

For this library, DF are optional but for the sake of this exercise, let's load them 

- df001.fits
- df002.fits

>>> o_norm.load(file='/Users/me/df/df001.fits', data_type='df')
>>> o_norm.load(file='/Users/me/df/df002.fits', data_type='df')

By default, a gamma filtering will take place when you load your data. You can manually turn off
this filtering by adding the following False flag

>>> o_norm.load(file='/Users/me/df/df002.fits', data_type='df', gamma_filter=False)

The gamma filtering is an algorithm that replaces all the very bright pixel counts with the average value
of the 8 neighbors. What do we mean by very bright? The pixel counts that have 10% of their value above the average
counts of the entire image. The threshold value can be change by doing

>>> o_norm.gamma_filter_threshold = 0.2

**WARNING:**
#1 From this point, any operation on your data will overwrite the inital data loaded. Those
data can be retrieved at any point by doing
#2 The program won't let you run the same algorithm twice (normalization, df_correction, 
oscillation, rebin). But it's possible to overwrite this option by making a flag **force**
equal to True. Use this feature at your own risk!

>>> data = o_norm.data['sample']['data']
>>> ob = o_norm.data['ob']['data']
