from pathlib import Path
import numpy as np
import os
import copy
import time
from scipy.ndimage import convolve

from NeuNorm.loader import load_hdf, load_tiff, load_fits
from NeuNorm.exporter import make_fits, make_tif
from NeuNorm.roi import ROI
from NeuNorm._utilities import get_sorted_list_images, average_df


class Normalization(object):

    working_data_type = np.float32

    def __init__(self):
        self.shape = {'width': np.NaN,
                      'height': np.NaN}
        self.dict_image = { 'data': [],
                            'oscilation': [],
                            'file_name': [],
                            'metadata': [],
                            'shape': self.shape.copy()}
        self.dict_ob = {'data': [],
                        'oscilation': [],
                        'metadata': [],
                        'file_name': [],
                        'data_mean': [],
                        'shape': self.shape.copy()}
        self.dict_df = {'data': [],
                        'metadata': [],
                        'data_average': [],
                        'file_name': [],
                        'shape': self.shape.copy()}

        __roi_dict = {'x0': np.NaN,
                      'x1': np.NaN,
                      'y0': np.NaN,
                      'y1': np.NaN}
        self.roi = {'normalization': __roi_dict.copy(),
                    'crop': __roi_dict.copy()}

        self.__exec_process_status = {'df_correction': False,
                                      'normalization': False,
                                      'crop': False,
                                      'oscillation': False,
                                      'bin': False}

        self.data = {}
        self.data['sample'] = self.dict_image
        self.data['ob'] = self.dict_ob
        self.data['df'] = self.dict_df
        self.data['normalized'] = []
        self.export_file_name = []
    
    def load(self, file='', folder='', data=[], data_type='sample', auto_gamma_filter=True,
            manual_gamma_filter=False, notebook=False, manual_gamma_threshold=0.1):
        '''
        Function to read individual files, entire files from folder, list of files or event data arrays.
        Data are also gamma filtered if requested.
        
        Parameters:
           file: list -  full path to a single file, or list of files
           folder: string - full path to folder containing files to load
           data: numpy array - 2D array of data to load
           data_type: string - 'sample', 'ob' or 'df (default 'sample')
           auto_gamma_filter: boolean - will correct the gamma filter automatically (highest count possible
                for the data type will be replaced by the average of the 9 neighboring pixels) (default True)
           manual_gamma_filter: boolean - apply or not gamma filtering to the data loaded (default False)
           notebooks: boolean - turn on this option if you run the library from a
             notebook to have a progress bar displayed showing you the progress of the loading (default False)
            manual_gamma_threshold: float between 0 and 1 - manual gamma coefficient to use (default 0.1)

        Warning:
            Algorithm won't be allowed to run if any of the main algorithm have been run already, such as
                oscillation, crop, binning, df_correction.

        '''

        list_exec_flag = [_flag for _flag in self.__exec_process_status.values()]
        box1 = None
        if True in list_exec_flag:
            raise IOError("Operation not allowed as you already worked on this data set!")
       
        if notebook:
            from ipywidgets import widgets
            from IPython.core.display import display
        
        if not file == '':
            if isinstance(file, str):
                self.load_file(file=file,
                               data_type=data_type,
                               auto_gamma_filter=auto_gamma_filter,
                               manual_gamma_filter=manual_gamma_filter,
                               manual_gamma_threshold=manual_gamma_threshold)
            elif isinstance(file, list):
                if notebook:
                    # turn on progress bar
                    _message = "Loading {}".format(data_type)
                    box1 = widgets.HBox([widgets.Label(_message,
                                                       layout=widgets.Layout(width='20%')),
                                         widgets.IntProgress(max=len(file)),
                                         widgets.Label("Time remaining:",
                                                       layout=widgets.Layout(width='10%')),
                                         widgets.Label(" >> calculating << ")])
                    display(box1)
                    w1 = box1.children[1]                    
                    time_remaining_ui = box1.children[-1]

                start_time = time.time()
                for _index, _file in enumerate(file):
                    self.load_file(file=_file,
                                   data_type=data_type,
                                   auto_gamma_filter=auto_gamma_filter,
                                   manual_gamma_filter=manual_gamma_filter,
                                   manual_gamma_threshold=manual_gamma_threshold)
                    if notebook:
                        w1.value = _index+1
                        end_time = time.time()
                        takes_its_going_to_take = self.calculate_how_long_its_going_to_take(index_we_are=_index + 1,
                                                                                            time_it_took_so_far=end_time - start_time,
                                                                                            total_number_of_loop=len(file))
                        time_remaining_ui.value = "{}".format(takes_its_going_to_take)

                if notebook:
                    box1.close()

        elif not folder == '':
            # load all files from folder
            list_images = get_sorted_list_images(folder=folder)
            if notebook:
                # turn on progress bar
                _message = "Loading {}".format(data_type)
                box1 = widgets.HBox([widgets.Label(_message,
                                                   layout=widgets.Layout(width='20%')),
                                     widgets.IntProgress(max=len(list_images)),
                                     widgets.Label("Time remaining:",
                                                   layout=widgets.Layout(width='10%')),
                                     widgets.Label(" >> calculating << ")])
                display(box1)
                w1 = box1.children[1]
                time_remaining_ui = box1.children[-1]

            start_time = time.time()
            for _index, _image in enumerate(list_images):
                full_path_image = os.path.join(folder, _image)
                self.load_file(file=full_path_image,
                               data_type=data_type,
                               auto_gamma_filter=auto_gamma_filter,
                               manual_gamma_filter=manual_gamma_filter,
                               manual_gamma_threshold=manual_gamma_threshold)
                if notebook:
                    # update progress bar
                    w1.value = _index+1
                    end_time = time.time()
                    takes_its_going_to_take = self.calculate_how_long_its_going_to_take(index_we_are=_index+1,
                                                                                        time_it_took_so_far=end_time-start_time,
                                                                                        total_number_of_loop=len(list_images))
                    time_remaining_ui.value = "{}".format(takes_its_going_to_take)

            if notebook:
                box1.close()
        
        elif not data == []:
            self.load_data(data=data, data_type=data_type)

    def calculate_how_long_its_going_to_take(self, index_we_are=-1, time_it_took_so_far=0, total_number_of_loop=1):
        """Estimate how long the loading is going to take according to the time it already took to load the
        first images.

        Parameters:
            index_we_are: int - index where we are in the list of files to load (default -1)
            time_it_took_so_far: float - time it took so far to load the data (default 0)
            total_number_of_loop: int - total number of files to load (default 1)

        Returns:
            string
        """
        time_per_loop = time_it_took_so_far / index_we_are
        total_time_it_will_take = time_per_loop * total_number_of_loop
        time_left = total_time_it_will_take - time_per_loop * index_we_are

        # convert to nice format h mn and seconds
        m, s = divmod(time_left, 60)
        h, m = divmod(m, 60)

        if h == 0:
            if m == 0:
                return "%02ds" %(s)
            else:
                return "%02dmn %02ds" %(m, s)
        else:
            return "%dh %02dmn %02ds" % (h, m, s)

    def load_data(self, data=[], data_type='sample', notebook=False):
        '''Function to save the data already loaded as arrays

        Paramters:
            data: np array 2D or 3D
            data_type: string  - 'sample', 'ob' or 'df' (default 'sample')
            notebook: boolean - turn on this option if you run the library from a
                 notebook to have a progress bar displayed showing you the progress of the loading (default False)
        '''
        if notebook:
            from ipywidgets import widgets
            from IPython.core.display import display

        if len(np.shape(data)) > 2:
            if notebook:
                _message = "Loading {}".format(data_type)
                box1 = widgets.HBox([widgets.Label(_message,
                                                               layout=widgets.Layout(width='20%')),
                                                 widgets.IntProgress(max=len(list_images))])
                display(box1)
                w1 = box1.children[1]   

            for _index, _data in enumerate(data):
                _data = _data.astype(self.working_data_type)
                self.__load_individual_data(data=_data, data_type=data_type)
                if notebook:
                    # update progress bar
                    w1.value = _index+1

            if notebook:
                box1.close()
                    
        else:
            data = data.astype(self.working_data_type)
            self.__load_individual_data(data=data, data_type=data_type)
            
    def __load_individual_data(self, data=[], data_type='sample'):
        """method that loads the data one at a time

        Parameters:
            data: np array
            data_type: string - 'data', 'ob' or 'df' (default 'sample')
        """
        self.data[data_type]['data'].append(data)
        index = len(self.data[data_type]['data'])
        self.data[data_type]['file_name'].append("image_{:04}".format(index))
        self.data[data_type]['metadata'].append('')
        self.save_or_check_shape(data=data, data_type=data_type)        
        
    def load_file(self, file='', data_type='sample',
                  auto_gamma_filter=True,
                  manual_gamma_filter=False,
                  manual_gamma_threshold=0.1):
        """
        Function to read data from the specified path, it can read FITS, TIFF and HDF.
    
        Parameters
            file : string - full path of the input file with his extension.
            data_type: string - 'sample', 'df' or 'ob' (default 'sample')
            manual_gamma_filter: boolean  - apply or not gamma filtering (default False)
            manual_gamma_threshold: float (between 0 and 1) - manual gamma threshold
            auto_gamma_filter: boolean - flag to turn on or off the auto gamma fitering (default True)

        Raises:
            OSError: if file does not exist
            NotImplementedError: if file is HDF5
            OSError: if any other any file format requested

        """
        my_file = Path(file)
        if my_file.is_file():
            metadata = {}
            if file.lower().endswith('.fits'):
                data = np.array(load_fits(my_file))
            elif file.lower().endswith(('.tiff','.tif')) :
                [data, metadata] = load_tiff(my_file)
                data = np.array(data)
            elif file.lower().endswith(('.hdf','.h4','.hdf4','.he2','h5','.hdf5','.he5')):
                raise NotImplementedError
            #     data = np.array(load_hdf(my_file))
            else:
                raise OSError('file extension not yet implemented....Do it your own way!')     

            if auto_gamma_filter:
                data = self._auto_gamma_filtering(data=data)
            elif manual_gamma_filter:
                data = self._manual_gamma_filtering(data=data, manual_gamma_threshold=manual_gamma_threshold)

            data = np.squeeze(data)

            self.data[data_type]['data'].append(data)
            self.data[data_type]['metadata'].append(metadata)
            self.data[data_type]['file_name'].append(file)
            self.save_or_check_shape(data=data, data_type=data_type)

        else:
            raise OSError("The file name does not exist")

    def _auto_gamma_filtering(self, data=[]):
        '''perform the automatic gamma filtering

        This algorithm check the data format of the input data file (ex: int16, int32...)
        and will determine the maxixum value for this data type. Any pixel that have a value
        above the max value - 5 (just to give it a little bit of range) will be considered as
        being gamma pixels. Those pixels will be replaced by the average value of the 8 pixels
        surrounding this pixel

        Parameters:
            data: np array

        Returns:
            np array of the data cleaned

        Raises:
            ValueError if array is empty
        '''
        if data == []:
            raise ValueError("Data array is empty!")

        # we may be dealing with a float time, that means it does not need any gamma filtering

        try:
            max = np.iinfo(data.dtype).max
        except:
            return data

        manual_gamma_threshold = max - 5
        new_data = np.array(data, self.working_data_type)

        data_gamma_filtered = np.copy(new_data)
        gamma_indexes = np.where(new_data > manual_gamma_threshold)

        mean_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8.0
        convolved_data = convolve(data_gamma_filtered, mean_kernel, mode='constant')

        data_gamma_filtered[gamma_indexes] = convolved_data[gamma_indexes]

        return data_gamma_filtered

    def _manual_gamma_filtering(self, data=[], manual_gamma_threshold=0.1):
        '''perform manual gamma filtering on the data

        This algoritm uses the manual_gamma_threshold value to estimate if a pixel is a gamma or not.
        1. mean value of data array is calculated
        2. pixel is considered gamma if its value times the manual gamma threshold is bigger than the mean value
        3. if pixel is gamma, its value is replaced by the mean value of the 8 pixels surrounding it.

        Parameters:
            data: numpy 2D array
            manual_gamma_threshold: float - coefficient between 0 and 1 used to estimate the threshold of the
            gamma pixels (default 0.1)
        
        Returns:
            numpy 2D array

        Raises:
             ValueError if data is empty
        '''
        if data == []:
            raise ValueError("Data array is empty!")

        data_gamma_filtered = np.copy(data)
        mean_counts = np.mean(data_gamma_filtered)
        gamma_indexes = np.where(manual_gamma_threshold * data_gamma_filtered > mean_counts)

        mean_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8.0
        convolved_data = convolve(data_gamma_filtered, mean_kernel, mode='constant')

        data_gamma_filtered[gamma_indexes] = convolved_data[gamma_indexes]

        return data_gamma_filtered

    def save_or_check_shape(self, data=[], data_type='sample'):
        '''save the shape for the first data loaded (of each type) otherwise
        check if the size match

        Parameters:
            data: np array of the data to check or save shape (default [])
            data_type: string - 'ob', 'df' or 'sample' (default 'sample')

        Raises:
            IOError if size do not match
        '''
        [height, width] = np.shape(data)
        if np.isnan(self.data[data_type]['shape']['height']):
            _shape = self.shape.copy()
            _shape['height'] = height
            _shape['width'] = width
            self.data[data_type]['shape'] = _shape
        else:
            _prev_width = self.data[data_type]['shape']['width']
            _prev_height = self.data[data_type]['shape']['height']
            
            if (not (_prev_width == width)) or (not (_prev_height == height)):
                raise IOError("Shape of {} do not match previous loaded data set!".format(data_type))

    def normalization(self, roi=None, force=False, force_mean_ob=False, notebook=False):
        '''normalization of the data 
                
        Parameters:
            roi: ROI object or list of ROI objects - object defines the region of the sample and OB that have to match
        in intensity
            force: boolean - True will force the normalization to occur, even if it had been
                run before with the same data set (default False)
        notebook: boolean - turn on this option if you run the library from a
             notebook to have a progress bar displayed showing you the progress of the loading (default False)

        Return:
            True - status of the normalization (True if every went ok, this is mostly used for the unit test)

        Raises:
            IOError: if no sample loaded
            IOError: if no OB loaded
            IOError: if size of sample and OB do not match
        
        '''
        if not force:
            # does nothing if normalization has already been run
            if self.__exec_process_status['normalization']:
                return
        self.__exec_process_status['normalization'] = True
        
        # make sure we loaded some sample data
        if self.data['sample']['data'] == []:
            raise IOError("No normalization available as no data have been loaded")

        # make sure we loaded some ob data
        if self.data['ob']['data'] == []:
            raise IOError("No normalization available as no OB have been loaded")
              
        # make sure the data loaded have the same size
        if not self.data_loaded_have_matching_shape():
            raise ValueError("Data loaded do not have the same shape!")
              
        # make sure, if provided, roi has the right type and fits into the images
        b_list_roi = False
        if roi:
            if type(roi) is list:
                for _roi in roi:
                    if not type(_roi) == ROI:
                        raise ValueError("roi must be a ROI object!")
                    if not self.__roi_fit_into_sample(roi=_roi):
                        raise ValueError("roi does not fit into sample image!")
                b_list_roi = True

            elif not type(roi) == ROI:
                raise ValueError("roi must be a ROI object!")
            else:
                if not self.__roi_fit_into_sample(roi=roi):
                    raise ValueError("roi does not fit into sample image!")
        
        if notebook:
            from ipywidgets import widgets
            from IPython.core.display import display

        if roi:

            if b_list_roi:

                _sample_corrected_normalized = []
                for _sample in self.data['sample']['data']:
                    sample_mean = []
                    for _roi in roi:
                        _x0 = _roi.x0
                        _y0 = _roi.y0
                        _x1 = _roi.x1
                        _y1 = _roi.y1
                        sample_mean.append(np.mean(_sample[_y0:_y1 + 1, _x0:_x1 + 1]))

                    full_sample_mean = np.mean(sample_mean)
                    _sample_corrected_normalized.append(_sample/full_sample_mean)

                _ob_corrected_normalized = []
                for _ob in self.data['ob']['data']:
                    ob_mean = []
                    for _roi in roi:
                        _x0 = _roi.x0
                        _y0 = _roi.y0
                        _x1 = _roi.x1
                        _y1 = _roi.y1
                        ob_mean.append(np.mean(_ob[_y0:_y1 + 1, _x0:_x1 + 1]))

                    # full_ob_mean = np.mean(ob_mean)
                    _ob_corrected_normalized.append(_ob / full_sample_mean)

            else:
                _x0 = roi.x0
                _y0 = roi.y0
                _x1 = roi.x1
                _y1 = roi.y1

                _sample_corrected_normalized = [_sample / np.mean(_sample[_y0:_y1+1, _x0:_x1+1])
                                                for _sample in self.data['sample']['data']]
                _ob_corrected_normalized = [_ob / np.mean(_ob[_y0:_y1+1, _x0:_x1+1])
                                            for _ob in self.data['ob']['data']]

        else:
            _sample_corrected_normalized = copy.copy(self.data['sample']['data'])
            _ob_corrected_normalized = copy.copy(self.data['ob']['data'])
            
        self.data['sample']['data'] = _sample_corrected_normalized
        self.data['ob']['data'] = _ob_corrected_normalized
            
        # if the number of sample and ob do not match, use mean of obs
        nbr_sample = len(self.data['sample']['file_name'])
        nbr_ob = len(self.data['ob']['file_name'])
        if (nbr_sample != nbr_ob) or force_mean_ob: # work with mean ob
            _ob_corrected_normalized = np.mean(_ob_corrected_normalized, axis=0)
            self.data['ob']['data_mean'] = _ob_corrected_normalized
            _working_ob = _ob_corrected_normalized.copy()
            _working_ob[_working_ob == 0] = np.NaN

            if notebook:
                # turn on progress bar
                _message = "Normalization"
                box1 = widgets.HBox([widgets.Label(_message,
                                                   layout=widgets.Layout(width='20%')),
                                     widgets.IntProgress(max=len(self.data['sample']['data']))])
                display(box1)
                w1 = box1.children[1]    

            normalized_data = []
            for _index, _sample in enumerate(self.data['sample']['data']):
                _norm = np.divide(_sample, _working_ob)
                _norm[np.isnan(_norm)] = 0
                _norm[np.isinf(_norm)] = 0
                normalized_data.append(_norm)

                if notebook:
                    w1.value = _index+1                
            
        else: # 1 ob for each sample
            # produce normalized data
            sample_ob = zip(self.data['sample']['data'], self.data['ob']['data'])

            if notebook:
                # turn on progress bar
                _message = "Normalization"
                box1 = widgets.HBox([widgets.Label(_message,
                                                   layout=widgets.Layout(width='20%')),
                                     widgets.IntProgress(max=len(self.data['sample']['data']))])
                display(box1)
                w1 = box1.children[1]    

            normalized_data = []
            for _index, [_sample, _ob] in enumerate(sample_ob):
                _working_ob = _ob.copy()
                _working_ob[_working_ob == 0] = np.NaN
                _norm = np.divide(_sample, _working_ob)
                _norm[np.isnan(_norm)] = 0
                _norm[np.isinf(_norm)] = 0
                normalized_data.append(_norm)
                
                if notebook:
                    w1.value = _index+1                                

        self.data['normalized'] = normalized_data

        return True
    
    def data_loaded_have_matching_shape(self):
        '''check that data loaded have the same shape
        
        Returns:
            bool: result of the check
        '''
        _shape_sample = self.data['sample']['shape']
        _shape_ob = self.data['ob']['shape']
        
        if not (_shape_sample == _shape_ob):
            return False
        
        _shape_df = self.data['df']['shape']
        if not np.isnan(_shape_df['height']):
            if not (_shape_sample == _shape_df):
                return False
            
        return True
    
    def __roi_fit_into_sample(self, roi=[]):
        '''check if roi is within the dimension of the image
        
        Returns:
            bool: True if roi is within the image dimension
        
        '''
        [sample_height, sample_width] = np.shape(self.data['sample']['data'][0])
        
        [_x0, _y0, _x1, _y1] = [roi.x0, roi.y0, roi.x1, roi.y1]
        if (_x0 < 0) or (_x1 >= sample_width):
            return False
        
        if (_y0 < 0) or (_y1 >= sample_height):
            return False

        return True
    
    def df_correction(self, force=False):
        '''dark field correction of sample and ob
        
        Parameters
            force: boolean - that if True will force the df correction to occur, even if it had been
                run before with the same data set (default False)

        sample_df_corrected = sample - DF
        ob_df_corrected = OB - DF

        '''
        if not force:
            if self.__exec_process_status['df_correction']:
                return
        self.__exec_process_status['df_correction'] = True
        
        if not self.data['sample']['data'] == []:
            self.__df_correction(data_type='sample')
            
        if not self.data['ob']['data'] == []:
            self.__df_correction(data_type='ob')
    
    def __df_correction(self, data_type='sample'):
        '''dark field correction
        
        Parameters:
           data_type: string ['sample','ob]

        Raises:
            KeyError: if data type is not 'sample' or 'ob'
            IOError: if sample and df or ob and df do not have the same shape
        '''
        if not data_type in ['sample', 'ob']:
            raise KeyError("Wrong data type passed. Must be either 'sample' or 'ob'!")

        if self.data['df']['data'] == []:
            return
        
        if self.data['df']['data_average'] == []:
            _df = self.data['df']['data']
            if len(_df) > 1:
                _df = average_df(df=_df)
            self.data['df']['data_average'] = np.squeeze(_df)

        else:
            _df = np.squeeze(self.data['df']['data_average'])

        if np.shape(self.data[data_type]['data'][0]) != np.shape(self.data['df']['data'][0]):
            raise IOError("{} and df data must have the same shape!".format(data_type))
    
        _data_df_corrected = [_data - _df for _data in self.data[data_type]['data']]
        _data_df_corrected = [np.squeeze(_data) for _data in _data_df_corrected]
        self.data[data_type]['data'] = _data_df_corrected
    
    def crop(self, roi=None, force=False):
        ''' Cropping the sample and ob normalized data
        
        Parameters:
            roi: ROI object that defines the region to crop
            force: Boolean  - that force or not the algorithm to be run more than once
                with the same data set (default False)

        Returns:
            True (for unit test purpose)

        Raises:
            ValueError if sample and ob data have not been normalized yet
        '''
        if (self.data['sample']['data'] == []) or \
           (self.data['ob']['data'] == []):
            raise IOError("We need sample and ob Data !")

        if not type(roi) == ROI:
            raise ValueError("roi must be of type ROI")

        if not force:
            if self.__exec_process_status['crop']:
                return
        self.__exec_process_status['crop'] = True
        
        _x0 = roi.x0
        _y0 = roi.y0
        _x1 = roi.x1
        _y1 = roi.y1
        
        new_sample = [_data[_y0:_y1+1, _x0:_x1+1] for 
                      _data in self.data['sample']['data']]
        self.data['sample']['data'] = new_sample        
       
        new_ob = [_data[_y0:_y1+1, _x0:_x1+1] for 
                  _data in self.data['ob']['data']]
        self.data['ob']['data'] = new_ob        
        
        if not (self.data['df']['data'] == []):
            new_df = [_data[_y0:_y1+1, _x0:_x1+1] for 
                      _data in self.data['df']['data']]
            self.data['df']['data'] = new_df
            
        if not (self.data['normalized'] == []):
            new_normalized = [_data[_y0:_y1+1, _x0:_x1+1] for 
                              _data in self.data['normalized']]
            self.data['normalized'] = new_normalized        
        
        return True
    
    def export(self, folder='./', data_type='normalized', file_type='tif'):
        '''export all the data from the type specified into a folder
        
        Parameters:
            folder: String - where to create all the images. Folder must exist otherwise an error is
                raised (default is './')
            data_type: String - Must be one of the following 'sample','ob','df','normalized' (default is 'normalized').
            file_type: String - format in which to export the data. Must be either 'tif' or 'fits' (default is 'tif')

        Raises:
            IOError if the folder does not exist
            KeyError if data_type can not be found in the list ['normalized','sample','ob','df']

        '''
        if not os.path.exists(folder):
            raise IOError("Folder '{}' does not exist!".format(folder))

        if not data_type in ['normalized','sample','ob','df']:
            raise KeyError("data_type '{}' is wrong".format(data_type))

        prefix = ''
        if data_type == 'normalized':
            data = self.get_normalized_data()
            prefix = 'normalized'
            data_type = 'sample'
        else:
            data = self.data[data_type]['data']

        if data ==[]:
            return False

        metadata = self.data[data_type]['metadata']

        list_file_name_raw = self.data[data_type]['file_name']
        self.__create_list_file_names(initial_list=list_file_name_raw,
                                      output_folder = folder,
                                      prefix=prefix,
                                      suffix=file_type)

        self.__export_data(data=data,
                           metadata=metadata,
                           output_file_names = self._export_file_name,
                           suffix=file_type)
        
    
    def __export_data(self, data=[], metadata=[], output_file_names=[], suffix='tif'):
        '''save the list of files with the data specified
        
        Parameters:
            data: numpy array that contains the array of data to save (default [])
            output_file_names: numpy array of string of full file names (default [])
            suffix: String - format in which the file will be created (default 'tif')
        '''
        name_data_metadata_array = zip(output_file_names, data, metadata)
        for _file_name, _data, _metadata in name_data_metadata_array:
            if suffix == 'tif':
                make_tif(data=_data, metadata=_metadata, file_name=_file_name)
            elif suffix == 'fits':
                make_fits(data=_data, file_name=_file_name)
    
    def __create_list_file_names(self, initial_list=[], output_folder='', prefix='', suffix=''):
        '''create a list of the new file name used to export the images
        
        Parameters:
            initial_list: array of full file name
               ex: ['/users/me/image001.tif',/users/me/image002.tif',/users/me/image003.tif']
            output_folder: String (default is ./ as specified by calling function) where we want to create the data
            prefix: String. what to add to the output file name in front of base name
                ex: 'normalized' will produce 'normalized_image001.tif'
            suffix: String. extension to file. 'tif' for TIFF and 'fits' for FITS
        '''
        _base_name = [os.path.basename(_file) for _file in initial_list]
        _raw_name = [os.path.splitext(_file)[0] for _file in _base_name]
        _prefix = ''
        if prefix:
            _prefix = prefix + '_'
        full_file_names = [os.path.join(output_folder, _prefix + _file + '.' + suffix) for _file in _raw_name]
        self._export_file_name = full_file_names
    
    def get_normalized_data(self):
        '''return the normalized data'''
        return self.data['normalized']

    def get_sample_data(self):
        '''return the sample data'''
        return self.data['sample']['data']
   
    def get_ob_data(self):
        '''return the ob data'''
        return self.data['ob']['data']
    
    def get_df_data(self):
        '''return the df data'''
        return self.data['df']['data']    