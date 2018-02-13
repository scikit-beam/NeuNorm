import unittest
import numpy as np
import os
from PIL import Image

from NeuNorm.normalization import Normalization
from NeuNorm.roi import ROI


class TestNormalization(unittest.TestCase):
    
    def setUp(self):    
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))

    def test_loading_list_of_files(self):
        '''assert initialization using list of files'''
        list_files = [self.data_path + '/tif/sample/image001.tif',
                      self.data_path + '/tif/sample/image002.tif',
                      self.data_path + '/tif/sample/image003.tif']
        o_norm = Normalization()
        o_norm.load(file=list_files)
        data_returned = o_norm.data['sample']['data']
        assert (3,5,5) == np.shape(data_returned)
        
    def test_initialization_using_array_with_data(self):
        '''assert initialization using arrays with data'''
        sample_01 = self.data_path + '/tif/sample/image001.tif'
        sample_02 = self.data_path + '/tif/sample/image002.tif'
        data = []
        data.append(np.asarray(Image.open(sample_01)))
        data.append(np.asarray(Image.open(sample_02)))
        o_norm = Normalization()
        o_norm.load(data=data)

        data_returned = o_norm.data['sample']['data']
        assert (2,5,5) == np.shape(data_returned)
        
    def test_initialization_using_array_with_data_one_by_one(self):
        '''assert initialization using arrays with data one by one'''
        o_norm = Normalization()

        sample_01 = self.data_path + '/tif/sample/image001.tif'
        _data = np.asarray(Image.open(sample_01))
        o_norm.load(data=_data)
        
        sample_02 = self.data_path + '/tif/sample/image002.tif'
        _data = np.asarray(Image.open(sample_01))
        o_norm.load(data=_data)

        data_returned = o_norm.data['sample']['data']
        assert (2,5,5) == np.shape(data_returned)

    def test_initialization_using_array_with_ob(self):
        '''assert initialization using arrays with ob'''
        ob_01 = self.data_path + '/tif/ob/ob001.tif'
        ob_02 = self.data_path + '/tif/ob/ob002.tif'
        data = []
        data.append(np.asarray(Image.open(ob_01)))
        data.append(np.asarray(Image.open(ob_02)))
        o_norm = Normalization()
        o_norm.load(data=data, data_type='ob')

        data_returned = o_norm.data['ob']['data']
        assert (2,5,5) == np.shape(data_returned)
        
    def test_normalization_raises_error_if_no_ob_or_sample(self):
        '''assert error raises when no ob or sample provided'''
        path = self.data_path + '/tif/sample'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='sample')
        self.assertRaises(IOError, o_norm.normalization)
        
        path = self.data_path + '/tif/ob'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='ob')
        self.assertRaises(IOError, o_norm.normalization)
        
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        o_norm = Normalization()
        o_norm.load(folder=sample_path, data_type='sample')
        o_norm.load(folder=ob_path, data_type='ob')
        assert o_norm.normalization()
 
    def test_normalization_ran_only_once(self):
        '''assert normalization is only once if force switch not turn on'''
        sample_tif_folder = self.data_path + '/tif/sample'
        ob_tif_folder = self.data_path + '/tif/ob'
    
        # testing sample with norm_roi
        o_norm = Normalization()
        o_norm.load(folder=sample_tif_folder)
        o_norm.load(folder=ob_tif_folder, data_type='ob')
        roi = ROI(x0=0, y0=0, x1=3, y1=2)
        o_norm.normalization(roi=roi)
        _returned_first_time = o_norm.data['sample']['data'][0]
        o_norm.normalization(roi=roi)
        _returned_second_time = o_norm.data['sample']['data'][0]
        self.assertTrue((_returned_first_time == _returned_second_time).all())        

    def test_normalization_ran_twice_with_force_flag(self):
        '''assert normalization can be ran twice using force flag'''
        sample_tif_folder = self.data_path + '/tif/sample'
        ob_tif_folder = self.data_path + '/tif/ob'
    
        # testing sample with norm_roi
        o_norm = Normalization()
        o_norm.load(folder=sample_tif_folder)
        o_norm.load(folder=ob_tif_folder, data_type='ob')
        roi = ROI(x0=0, y0=0, x1=3, y1=2)
        o_norm.normalization(roi=roi)
        _returned_first_time = o_norm.data['sample']['data'][0]
        roi = ROI(x0=0, y0=0, x1=2, y1=3)
        o_norm.normalization(roi=roi, force=True)
        _returned_second_time = o_norm.data['sample']['data'][0]
        self.assertFalse((_returned_first_time == _returned_second_time).all())
  
    def test_normalization_works_if_input_arrays_are_type_int(self):
        '''assert normalization works when input arrays are type int'''
        o_norm = Normalization()
        
        sample_01 = self.data_path + '/tif/sample/image001.tif'
        _data = np.asarray(Image.open(sample_01), dtype=int)
        o_norm.load(data=_data)
        
        sample_02 = self.data_path + '/tif/sample/image002.tif'
        _data = np.asarray(Image.open(sample_01), dtype=int)
        o_norm.load(data=_data)

        ob_01 = self.data_path + '/tif/ob/ob001.tif'
        _data = np.asarray(Image.open(ob_01), dtype=int)
        _data[0,0] = 0
        o_norm.load(data=_data, data_type='ob')
    
        ob_02 = self.data_path + '/tif/ob/ob002.tif'
        _data = np.asarray(Image.open(ob_01), dtype=int)
        o_norm.load(data=_data, data_type='ob')

        o_norm.normalization()

    def test_normalization_works(self):
        '''assert sample and ob normalization works with and without roi'''
        sample_tif_folder = self.data_path + '/tif/sample'
        ob_tif_folder = self.data_path + '/tif/ob'

        # testing sample with norm_roi
        o_norm = Normalization()
        o_norm.load(folder=sample_tif_folder)
        o_norm.load(folder=ob_tif_folder, data_type='ob')
        roi = ROI(x0=0, y0=0, x1=3, y1=2)
        _sample = o_norm.data['sample']['data'][0]
        _expected = _sample / np.mean(_sample[0:3, 0:4])
        o_norm.normalization(roi=roi)
        _returned = o_norm.data['sample']['data'][0]
        self.assertTrue((_expected == _returned).all())

        # testing sample without norm_roi
        o_norm1 = Normalization()
        o_norm1.load(folder=sample_tif_folder)
        o_norm1.load(folder=ob_tif_folder, data_type='ob')
        _expected = o_norm1.data['sample']['data'][0]
        o_norm1.normalization()
        _returned = o_norm1.data['sample']['data'][0]
        self.assertTrue((_expected == _returned).all())
        
        # testing ob with norm_roi
        o_norm = Normalization()
        o_norm.load(folder=sample_tif_folder)
        o_norm.load(folder=ob_tif_folder, data_type='ob')
        norm_roi = ROI(x0=0, y0=0, x1=3, y1=2)
        o_norm.normalization(roi=norm_roi)
        _ob = o_norm.data['ob']['data'][0]
        _expected = _ob / np.mean(_ob[0:3, 0:4])
        _returned = o_norm.data['ob']['data'][0]
        self.assertTrue((_expected == _returned).all())
        
        # testing ob without norm_roi
        o_norm = Normalization()
        o_norm.load(folder=sample_tif_folder)
        o_norm.load(folder=ob_tif_folder, data_type='ob')
        _expected = o_norm.data['ob']['data'][0]
        o_norm.normalization()
        _returned = o_norm.data['ob']['data'][0]
        self.assertTrue((_expected == _returned).all())  
  
    def test_normalization_with_same_ob_and_sample_but_forced_mean_ob(self):
        '''assert normalization with same ob and sample number of files force to use mean ob when flag used'''
        samples_path =  self.data_path + '/tif/sample/' # 3 files
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        ob2 = self.data_path + '/tif/ob/ob002.tif'
        ob3 = self.data_path + '/tif/ob/ob003.tif'
        o_norm = Normalization()
        o_norm.load(folder=samples_path)
        o_norm.load(file=[ob1, ob2, ob3], data_type='ob')
        o_norm.normalization(force_mean_ob=True)
        expected_normalized_array = np.ones((5,5))
        expected_normalized_array[:,2] = 2
        expected_normalized_array[:,3] = 3
        expected_normalized_array[:,4] = 4
        self.assertTrue((o_norm.data['normalized'][0] == expected_normalized_array).all())

    def test_normalization_with_fewer_ob_than_sample_works(self):
        '''assert normalization works when number of ob and sample is different'''
        samples_path =  self.data_path + '/tif/sample/' # 3 files
        ob1 = self.data_path + '/tif/ob/ob001.tif' 
        ob2 = self.data_path + '/tif/ob/ob002.tif' 
        df1 = self.data_path + '/tif/df/df001.tif'
        o_norm = Normalization()
        o_norm.load(folder=samples_path)
        o_norm.load(file=[ob1, ob2], data_type='ob')
        o_norm.load(file=df1, data_type='df')
        o_norm.df_correction()
        o_norm.normalization()
        expected_normalized_array = np.zeros((5,5))
        expected_normalized_array[0,0] = 1
        self.assertTrue((o_norm.data['normalized']== expected_normalized_array).all())

    def test_nbr_data_files_same_after_normalization_by_list_roi(self):
        '''assert the number of data files is the same after normalization by a list of ROI'''
        samples_path =  self.data_path + '/tif/sample/' # 3 files
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        ob2 = self.data_path + '/tif/ob/ob002.tif'
        df1 = self.data_path + '/tif/df/df001.tif'
        o_norm = Normalization()
        o_norm.load(folder=samples_path)
        o_norm.load(file=[ob1, ob2], data_type='ob')
        o_norm.load(file=df1, data_type='df')
        _roi1 = ROI(x0=0,y0=0,x1=2,y1=2)
        _roi2 = ROI(x0=1,y0=1,x1=3,y1=3)
        _list_roi = [_roi1, _roi2]
        nbr_data_before = len(o_norm.data['sample']['data'])
        o_norm.normalization(roi=_list_roi)
        nbr_data_after = len(o_norm.data['sample']['data'])
        self.assertEqual(nbr_data_after, nbr_data_before)




class TestDFCorrection(unittest.TestCase):
    
    def setUp(self):    
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))  
        
    def test_df_correction_when_no_df(self):
        '''assert sample and ob are inchanged if df is empty'''

        # sample
        path = self.data_path + '/tif/sample'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='sample')
        data_before = o_norm.data['sample']['data'][0]
        o_norm.df_correction()
        data_after = o_norm.data['sample']['data'][0]
        self.assertTrue((data_before == data_after).all())
        
        #ob
        path = self.data_path + '/tif/ob'
        o_norm = Normalization()
        o_norm.load(folder=path, data_type='ob')
        data_before = o_norm.data['ob']['data'][0]
        o_norm.df_correction()
        data_after = o_norm.data['ob']['data'][0]
        self.assertTrue((data_before == data_after).all())
        
    def test_df_fails_when_not_identical_data_shape(self):
        o_norm = Normalization()
        sample_1 = np.ones([5,5])
        df_1 = np.ones([6,6])
        o_norm.data['sample']['data'] = sample_1
        o_norm.data['df']['data'] = df_1
        self.assertRaises(IOError, o_norm.df_correction)
        
        o_norm = Normalization()
        ob_1 = np.ones([6,6])
        o_norm.data['ob']['data'] = sample_1
        o_norm.data['df']['data'] = ob_1
        self.assertRaises(IOError, o_norm.df_correction)

    def test_df_averaging_only_run_the_first_time(self):
        '''assert the average_df is only run the first time the df_correction is run'''
        sample_path = self.data_path + '/tif/sample/'
        ob_path = self.data_path + '/tif/ob/'
        o_norm = Normalization()
        o_norm.load(folder=sample_path)
        o_norm.load(folder=ob_path, data_type='ob')
        df_file_1 = self.data_path + '/tif/df/df002.tif'
        df_file_2 = self.data_path + '/tif/df/df003.tif'
        o_norm.load(file=df_file_1, data_type='df')
        o_norm.load(file=df_file_2, data_type='df')
    
        df_average_data = o_norm.data['df']['data_average']
        self.assertTrue(df_average_data == [])
    
        #sample
        o_norm.df_correction()
        df_average_data = o_norm.data['df']['data_average']
        self.assertTrue(df_average_data != [])
    
        #ob
        o_norm.df_correction()
        expected_df_average = df_average_data
        df_average = o_norm.data['df']['data_average']
        self.assertTrue((expected_df_average == df_average).all())

    def test_df_correction(self):
        '''assert df corrction works'''
        sample_path = self.data_path + '/tif/sample/'
        ob_path = self.data_path + '/tif/ob/'
        o_norm = Normalization()
        o_norm.load(folder=sample_path)
        o_norm.load(folder=ob_path, data_type='ob')
        df_file_1 = self.data_path + '/tif/df/df002.tif'
        df_file_2 = self.data_path + '/tif/df/df003.tif'
        o_norm.load(file=df_file_1, data_type='df')
        o_norm.load(file=df_file_2, data_type='df')
        
        #sample
        o_norm.df_correction()
        _expected_data = np.zeros([5,5])
        _expected_data[:,2] = 1
        _expected_data[:,3] = 2
        _expected_data[:,4] = 3       
        _sample_data = o_norm.data['sample']['data'][0]
        self.assertTrue((_expected_data == o_norm.data['sample']['data'][0]).all())
        
        #ob
        _expected_data = np.zeros([5,5])
        _ob_data = o_norm.data['ob']['data'][0]
        self.assertTrue((_expected_data == _ob_data).all())

    def test_df_correction_locked_when_run_twice_without_force_flag(self):
        '''assert df corrction run only one time if force flag is False'''
        sample_path = self.data_path + '/tif/sample/'
        ob_path = self.data_path + '/tif/ob/'
        o_norm = Normalization()
        o_norm.load(folder=sample_path)
        o_norm.load(folder=ob_path, data_type='ob')
        df_file_1 = self.data_path + '/tif/df/df002.tif'
        df_file_2 = self.data_path + '/tif/df/df003.tif'
        o_norm.load(file=df_file_1, data_type='df')
        o_norm.load(file=df_file_2, data_type='df')
        
        # first iteration
        o_norm.df_correction()
        _sample_first_run = o_norm.data['sample']['data'][0]
        _ob_first_run = o_norm.data['ob']['data'][0]
        
        # second iteration
        o_norm.df_correction()
        _sample_second_run = o_norm.data['sample']['data'][0]
        _ob_second_run = o_norm.data['ob']['data'][0]
         
        self.assertTrue((_sample_first_run == _sample_second_run).all())
        self.assertTrue((_ob_first_run == _ob_second_run).all())
         
    def test_df_correction_run_twice_with_force_flag(self):
        '''assert df corrction run more than once with force flag'''
        sample_path = self.data_path + '/tif/sample/'
        ob_path = self.data_path + '/tif/ob/'
        o_norm = Normalization()
        o_norm.load(folder=sample_path)
        o_norm.load(folder=ob_path, data_type='ob')
        df_file_1 = self.data_path + '/tif/df/df002.tif'
        df_file_2 = self.data_path + '/tif/df/df003.tif'
        o_norm.load(file=df_file_1, data_type='df')
        o_norm.load(file=df_file_2, data_type='df')
        
        # first iteration
        o_norm.df_correction()
        _sample_first_run = o_norm.data['sample']['data'][0]
        _ob_first_run = o_norm.data['ob']['data'][0]
        _average_df = o_norm.data['df']['data_average']
        
        # second iteration
        o_norm.df_correction(force=True)
        _sample_second_run = o_norm.data['sample']['data'][0]
        _ob_second_run = o_norm.data['ob']['data'][0]

        # expected
        _expected_sample_after_second_run = _sample_first_run - _average_df
        _expected_ob_after_second_run = _ob_first_run - _average_df
         
        self.assertTrue((_sample_second_run == _expected_sample_after_second_run).all())
        self.assertTrue((_ob_second_run == _expected_ob_after_second_run).all())    
         
        
class TestApplyingROI(unittest.TestCase):
    
    def setUp(self):    
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, '../data/'))       
        
    def test_roi_type_in_normalization(self):
        '''assert error is raised when type of norm roi are not ROI in normalization'''
        sample_tif_file = self.data_path + '/tif/sample/image001.tif'
        ob_tif_file = self.data_path + '/tif/ob/ob001.tif'
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample')
        o_norm.load(file=ob_tif_file, data_type='ob')
        roi = {'x0':0, 'y0':0, 'x1':2, 'y1':2}
        self.assertRaises(ValueError, o_norm.normalization, roi)
        
    def test_roi_fit_images(self):
        '''assert norm roi do fit the images'''
        sample_tif_file = self.data_path + '/tif/sample/image001.tif'
        ob_tif_file = self.data_path + '/tif/ob/ob001.tif'
        
        # x0 < 0 or x1 > image_width
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample')
        o_norm.load(file=ob_tif_file, data_type='ob')
        roi = ROI(x0=0, y0=0, x1=20, y1=4)
        self.assertRaises(ValueError, o_norm.normalization, roi)
       
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample')
        o_norm.load(file=ob_tif_file, data_type='ob')
        roi = ROI(x0=-1, y0=0, x1=4, y1=4)
        self.assertRaises(ValueError, o_norm.normalization, roi)        
        
        # y0 < 0 or y1 > image_height
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample')
        o_norm.load(file=ob_tif_file, data_type='ob')
        roi = ROI(x0=0, y0=-1, x1=4, y1=4)
        self.assertRaises(ValueError, o_norm.normalization, roi)

        # y1>image_height
        o_norm = Normalization()
        o_norm.load(file=sample_tif_file, data_type='sample')
        o_norm.load(file=ob_tif_file, data_type='ob')
        roi = ROI(x0=0, y0=0, x1=4, y1=20)
        self.assertRaises(ValueError, o_norm.normalization, roi)        

    def test_error_raised_when_data_shape_of_different_type_do_not_match(self):
        '''assert shape of data must match to allow normalization'''
        
        # sample and ob
        image1 = self.data_path + '/tif/sample/image001.tif'
        ob1 = self.data_path + '/different_format/ob001_4_by_4.tif'
        o_norm = Normalization()
        o_norm.load(file=image1)
        o_norm.load(file=ob1, data_type='ob')
        self.assertRaises(ValueError, o_norm.normalization)
        
        # sample, ob and df
        image1 = self.data_path + '/tif/sample/image001.tif'
        ob1 = self.data_path + '/tif/ob/ob001.tif'
        df1 = self.data_path + '/different_format/df001_4_by_4.tif'
        o_norm = Normalization()
        o_norm.load(file=image1)
        o_norm.load(file=ob1, data_type='ob')
        o_norm.load(file=df1, data_type='df')
        self.assertRaises(ValueError, o_norm.normalization)

    def test_full_normalization_sample_with_several_roi(self):
        '''assert the full normalization works with several roi selected'''
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        df_path = self.data_path + '/tif/df'
        o_norm = Normalization()
        o_norm.load(folder=sample_path)
        o_norm.load(folder=ob_path, data_type='ob')
        _roi_1 = ROI(x0=0, y0=0, x1=2, y1=2)
        _roi_2 = ROI(x0=2, y0=2, x1=4, y1=4)
        o_norm.normalization(roi=[_roi_1, _roi_2])
        _norm_returned = o_norm.data['normalized'][0]
        _norm_expected = np.ones((5,5))
        _norm_expected[:,2] = 1.02325581
        _norm_expected[:,3] = 1.53488372
        _norm_expected[:,4] = 2.04651163

        self.assertAlmostEquals(_norm_expected[0,0], _norm_returned[0,0], delta=1e-8)

    def test_full_normalization_sample_divide_by_ob_works(self):
        '''assert the full normalization works (when sample is divided by ob)'''

        # without normalization roi
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        df_path = self.data_path + '/tif/df'
        o_norm = Normalization()
        o_norm.load(folder=sample_path)
        o_norm.load(folder=ob_path, data_type='ob')
        o_norm.load(folder=df_path, data_type='df')
        o_norm.normalization()
        _norm_expected = np.ones((5,5))
        _norm_expected[:,2] = 2
        _norm_expected[:,3] = 3
        _norm_expected[:,4] = 4
        _norm_returned = o_norm.data['normalized']
        self.assertTrue((_norm_expected == _norm_returned).all())
        
        # with normalization roi
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        df_path = self.data_path + '/tif/df'
        o_norm = Normalization()
        o_norm.load(folder=sample_path)
        o_norm.load(folder=ob_path, data_type='ob')
        o_norm.load(folder=df_path, data_type='df')
        _roi = ROI(x0=0, y0=0, x1=2, y1=2)
        o_norm.normalization(roi=_roi)
        _norm_expected = np.ones((5,5))
        _norm_expected.fill(0.8125)
        _norm_expected[:,2] = 1.625
        _norm_expected[:,3] = 2.4375
        _norm_expected[:,4] = 3.25
        _norm_returned = o_norm.data['normalized']
        self.assertTrue((_norm_expected == _norm_returned).all())
        
    def test_various_data_type_correctly_returned(self):
        '''assert normalized, sample, ob and df data are correctly returned'''
        sample_path = self.data_path + '/tif/sample'
        ob_path = self.data_path + '/tif/ob'
        df_path = self.data_path + '/tif/df'
        o_norm = Normalization()
        o_norm.load(folder=sample_path)
        o_norm.load(folder=ob_path, data_type='ob')
        o_norm.load(folder=df_path, data_type='df')
        
        # sample
        _data_expected = o_norm.data['sample']['data'][0]
        _data_returned = o_norm.get_sample_data()[0]
        self.assertTrue((_data_expected == _data_returned).all())
        
        # ob
        _ob_expected = o_norm.data['ob']['data']
        _ob_returned = o_norm.get_ob_data()[0]
        self.assertTrue((_ob_expected == _ob_returned).all())
        
        # df
        _df_expected = o_norm.data['df']['data']
        _df_returned = o_norm.get_df_data()
        self.assertTrue(_df_expected == _df_returned)
        
        # normalized is empty before normalization
        self.assertTrue(o_norm.get_normalized_data() == [])
        
        # run normalization
        o_norm.normalization()
        
        _norm_expected = o_norm.data['normalized'][0]
        _norm_returned = o_norm.get_normalized_data()[0]
        self.assertTrue((_norm_expected == _norm_returned).all())
        