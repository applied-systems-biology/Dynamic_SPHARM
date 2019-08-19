import unittest

import os
import pandas as pd
from ddt import ddt
import shutil
import site

import SPHARM.lib.vrml_parse as vr


@ddt
class TestVRML(unittest.TestCase):

    def test_extract_node_names(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        vr.extract_node_names(path + 'data/test_vrml.vrml', 'data/test_data/test_vrml.txt')
        f = open('data/test_data/test_vrml.txt')
        st = f.readlines()
        f.close()
        self.assertEqual(len(st), 1362)

    def test_extract_node_names_batch(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        vr.extract_node_names_batch(path + 'data/vrml/', 'data/test_data/vrml/node_names/')
        files = os.listdir('data/test_data/vrml/node_names/')
        self.assertEqual(len(files), 2)
        shutil.rmtree('data/test_data/')

    def test_extract_coordinates(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        vr.extract_coordinates(path + 'data/test_vrml.vrml', 'data/test_data/test_vrml_coord.csv')
        stat = pd.read_csv('data/test_data/test_vrml_coord.csv', sep='\t', index_col=0)
        self.assertEqual(len(stat), 858)
        shutil.rmtree('data/test_data/')

    def test_extract_coordinates_batch(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        vr.extract_coordinates_batch(path + 'data/vrml/', 'data/test_data/vrml/coordinates/')
        files = os.listdir('data/test_data/vrml/coordinates/')
        self.assertEqual(len(files), 2)
        shutil.rmtree('data/test_data/')

    def test_combine_with_tracks(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        vr.combine_with_track_data(inputfile=path + 'data/LN_deconv_set4_small.csv',
                                   trackfile=path + 'data/LN_deconv_set4_Detailed.xlsx',
                                   outputfile='data/test_data/LN_deconv_set4_tracked.csv')
        stat = pd.read_csv('data/test_data/LN_deconv_set4_tracked.csv', sep='\t', index_col=0)
        self.assertEqual('TrackID' in stat.columns, True)
        shutil.rmtree('data/test_data/')

    def test_combine_with_tracks_batch(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        vr.combine_with_track_data_batch(inputfolder=path + 'data/wrl/',
                                         trackfolder=path + 'data/track_files/',
                                         outputfolder='data/test_data/vrml/tracked/')
        files = os.listdir('data/test_data/vrml/tracked/')
        self.assertEqual(len(files), 1)
        shutil.rmtree('data/test_data/')

if __name__ == '__main__':
    unittest.main()

