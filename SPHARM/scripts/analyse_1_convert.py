from __future__ import division

import mkl
mkl.set_num_threads(1)

import sys
import numpy as np

from SPHARM.lib import spharm
import SPHARM.lib.parallel as prl
from SPHARM.lib import plotting as plt
import SPHARM.lib.segmentation as sgm
import SPHARM.lib.vrml_parse as vr


#################################
args = sys.argv[1:]
if len(args) > 0:
    path = args[0]
    if path != 'test':
        if not path.endswith('/'):
            path += '/'
        if len(path.split('T_cells')) > 1:
            metadata_file = path + 'voxel_size&frame_rate.csv'
            vr.extract_coordinates_batch(path + 'wrl/', path + 'output/coordinates/')
            vr.combine_with_track_data_batch(inputfolder=path + 'output/coordinates/',
                                             trackfolder=path + 'wrl/',
                                             outputfolder=path + 'output/coordinates_tracked/')
            path = path + 'output/'
            sgm.split_to_surfaces_batch(inputfolder=path + 'coordinates_tracked/',
                                        outputfolder=path + 'surfaces/', combine_tracks=True, adjust_frame_rate=False,
                                        metadata_file=metadata_file)
        elif len(path.split('Synthetic')) > 1:
            kwargs = {'max_threads': 5, 'voxel_size': 1}

            prl.run_parallel(process=spharm.convert_to_tiff, inputfolder=path + 'coordinates/',
                             outputfolder=path + 'output/stacks/', extensions=['*'], debug=False, combine=False,
                             exclude=['config.xml', 'log.csv', 'runanalysis.csv',
                                      'runstatistics.csv', 'shapeanalysis.csv', 'tissue.csv'],
                             **kwargs)

            path = path + 'output/'

            # prl.run_parallel(process=plt.plot_maxprojections, inputfolder=path + 'stacks/',
            #                  outputfolder=path + 'stacks_maxproj_xy/', axis=0, combine=False, **kwargs)
            #
            # prl.run_parallel(process=plt.plot_maxprojections, inputfolder=path + 'stacks/',
            #                  outputfolder=path + 'stacks_maxproj_xz/', axis=1, combine=False, **kwargs)
            #
            # prl.run_parallel(process=plt.plot_maxprojections, inputfolder=path + 'stacks/',
            #                  outputfolder=path + 'stacks_maxproj_yz/', axis=2, combine=False, **kwargs)

            if len(args) > 1:

                start_iter = int(float(args[1]))
                exclude = ['t666.csv.tif']
                for t in np.arange(0, start_iter, 500):
                    exclude.append('t' + str(t) + '.csv.tif')

                prl.run_parallel(process=sgm.extract_surfaces,
                                 inputfolder=path + 'stacks/',
                                 outputfolder=path + 'surfaces_separate/',
                                 combine=False, exclude=exclude)

                sgm.combine_surfaces(inputfolder=path + 'surfaces_separate/', outputfolder=path + 'surfaces/')













