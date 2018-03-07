from pathlib import Path
import numpy as np
import os
import copy
from scipy.ndimage import convolve

from NeuNorm.loader import load_hdf, load_tiff, load_fits
from NeuNorm.exporter import make_fits, make_tif
from NeuNorm.roi import ROI
from NeuNorm._utilities import get_sorted_list_images, average_df


class Normalization(object):

    gamma_filter_threshold = 0.1
    
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
    
    def load(self, file='', folder='', data=[], data_type='sample', 
             gamma_filter=True, notebook=False):
        '''
        Function to read individual files or entire files from folder specify for the given
        data type
        
        Parameters:
           file: full path to file
           folder: full path to folder containing files to load
           data: numpy array
           data_type: ['sample', 'ob', 'df]
           gamma_filter: boolean (default True) apply or not gamma filtering to the data loaded
           notebooks: boolean (default False) turn on this option if you run the library from a
             notebook to have a progress bar displayed showing you the progress of the loading
           
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
                self.load_file(file=file, data_type=data_type)
            elif isinstance(file, list):
                if notebook:
                    # turn on progress bar
                    _message = "Loading {}".format(data_type)
                    box1 = widgets.HBox([widgets.Label(_message,
                                                       layout=widgets.Layout(width='20%')),
                                         widgets.IntProgress(max=len(file))])
                    display(box1)
                    w1 = box1.children[1]                    
                
                for _index, _file in enumerate(file):
                    self.load_file(file=_file, data_type=data_type)
                    if notebook:
                        w1.value = _index+1

                if notebook:
                    box1.close()

        if not folder == '':
            # load all files from folder
            list_images = get_sorted_list_images(folder=folder)
            if notebook:
                # turn on progress bar
                _message = "Loading {}".format(data_type)
                box1 = widgets.HBox([widgets.Label(_message,
                                                   layout=widgets.Layout(width='20%')),
                                     widgets.IntProgress(max=len(list_images))])
                display(box1)
                w1 = box1.children[1]   
            
            for _index, _image in enumerate(list_images):
                full_path_image = os.path.join(folder, _image)
                self.load_file(file=full_path_image, data_type=data_type, gamma_filter=gamma_filter)
                if notebook:
                    # update progress bar
                    w1.value = _index+1

            if notebook:
                box1.close()
        
        if not data == []:
            self.load_data(data=data, data_type=data_type)
            
    def load_data(self, data=[], data_type='sample', notebook=False):
        '''Function to save the data already loaded as arrays

        Paramters:
        ==========
        data: np array 2D or 3D 
        data_type: string ('sample')
        notebook: boolean (default False) turn on this option if you run the library from a
             notebook to have a progress bar displayed showing you the progress of the loading
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
                _data = _data.astype(float)
                self.__load_individual_data(data=_data, data_type=data_type)
                if notebook:
                    # update progress bar
                    w1.value = _index+1

            if notebook:
                box1.close()
                    
        else:
            data = data.astype(float)
            self.__load_individual_data(data=data, data_type=data_type)
            
    def __load_individual_data(self, data=[], data_type='sample'):
        self.data[data_type]['data'].append(data)
        index = len(self.data[data_type]['data'])
        self.data[data_type]['file_name'].append("image_{:04}".format(index))
        self.save_or_check_shape(data=data, data_type=data_type)        
        
    def load_file(self, file='', data_type='sample', gamma_filter=True):
        """
        Function to read data from the specified path, it can read FITS, TIFF and HDF.
    
        Parameters
        ----------
        file : string_like
            Path of the input file with his extention.
        data_type: ['sample', 'df']
        gamma_filter: Boolean (default True) apply or not gamma filtering
    
        Notes
        -----
        In case of corrupted header it skips the header and reads the raw data.
        For the HDF format you need to specify the hierarchy.
        """
    
        my_file = Path(file)
        if my_file.is_file():
            metadata = {}
            if file.lower().endswith('.fits'):
                data = np.array(load_fits(my_file), dtype=np.float)
            elif file.lower().endswith(('.tiff','.tif')) :
                [data, metadata] = load_tiff(my_file)
                data = np.array(data, dtype=np.float)
            elif file.lower().endswith(('.hdf','.h4','.hdf4','.he2','h5','.hdf5','.he5')):
                data = np.array(load_hdf(my_file), dtype=np.float)
            else:
                raise OSError('file extension not yet implemented....Do it your own way!')     

            if gamma_filter:
                data = self._gamma_filtering(data=data)

            data = np.squeeze(data)

            self.data[data_type]['data'].append(data)
            self.data[data_type]['metadata'].append(metadata)
            self.data[data_type]['file_name'].append(file)
            self.save_or_check_shape(data=data, data_type=data_type)

        else:
            raise OSError("The file name does not exist")

    def _gamma_filtering(self, data=[]):
        '''perform gamma filtering on the data
        
        Algorithm looks for all the very hight counts
        
        if self.gamma_filter_threshold * pixels[x,y] > mean_counts(data) then this pixel counts
        is replaced by the average value of the 8 pixels surrounding him
        
        Parameters:
        ===========
        data: numpy 2D array
        
        Returns:
        =======
        numpy 2D array 
        '''
        if data == []:
            raise ValueError("Data array is empty!")

        data_gamma_filtered = np.copy(data)
            
        # find mean counts
        mean_counts = np.mean(data_gamma_filtered)
        
        thresolded_data_gamma_filtered = data_gamma_filtered * self.gamma_filter_threshold
        
        # get pixels where value is above threshold
        position = []
        [height, width] = np.shape(data_gamma_filtered)
        for _x in np.arange(width):
            for _y in np.arange(height):
                if thresolded_data_gamma_filtered[_y, _x] > mean_counts:
                    position.append([_y, _x])
                    
        # convolve entire image using 3x3 kerne
        mean_kernel = np.array([[1,1,1], [1,0,1], [1,1,1]]) / 8.0
        convolved_data = convolve(data_gamma_filtered, mean_kernel, mode='constant')
        
        # replace only pixel above threshold by convolved data
        for _coordinates in position:
            [_y, _x] = _coordinates
            data_gamma_filtered[_y, _x] = convolved_data[_y, _x]
            
        return data_gamma_filtered        

    def save_or_check_shape(self, data=[], data_type='sample'):
        '''save the shape for the first data loaded (of each type) otherwise
        check the size match
    
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
        ===========
        roi: ROI object or list of ROI objects that defines the region of the sample and OB that have to match
        in intensity
        force: boolean (default False) that if True will force the normalization to occur, even if it had been
        run before with the same data set
        notebook: boolean (default False) turn on this option if you run the library from a
             notebook to have a progress bar displayed showing you the progress of the loading

        Raises:
        =======
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
            from ipywidgets.widgets import interact
            from ipywidgets import widgets
            from IPython.core.display import display, HTML                

        # heat normalization algorithm
        _sample_corrected_normalized = []
        _ob_corrected_normalized = []

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

                    full_ob_mean = np.mean(ob_mean)
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
        =======
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
        ========
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
        ==========
        force: boolean (default False) that if True will force the df correction to occur, even if it had been
        run before with the same data set

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
        ===========
        roi: ROI object that defines the region to crop
        force: Boolean (default False) that force or not the algorithm to be run more than once
        with the same data set

        Raises:
        =======
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
        ===========
        folder: String (default is './') where to create all the images. Folder must exist otherwise an error is raised
        data_type: String (default is 'normalized'). Must be one of the following 'sample','ob','df','normalized'
        file_type: String (default is 'tif') format in which to export the data. Must be either 'tif' or 'fits'

        Raises:
        =======
        IOError if the folder does not exist
        KeyError if data_type can not be found in the list ['normalized','sample','ob','df']

        '''
        if not os.path.exists(folder):
            raise IOError("Folder '{}' does not exist!".format(folder))

        if not data_type in ['normalized','sample','ob','df']:
            raise KeyError("data_type '{}' is wrong".format(data_type))

        data = []
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
        ===========
        data: numpy array that contains the array of data to save
        output_file_names: numpy array of string of full file names        
        suffix: String (default is 'tif') format in which the file will be created
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
        ==========
        initial_list: array of full file name 
           ex: ['/users/me/image001.tif',/users/me/image002.tif',/users/me/image003.tif']
        output_folder: String (default is ./ as specified by calling function) where we want to create the data
        prefix: String. what to add to the output file name in front of base name
           ex: 'normalized' will produce 'normalized_image001.tif'
        suffix: String. extenstion to file. 'tif' for TIFF and 'fits' for FITS
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