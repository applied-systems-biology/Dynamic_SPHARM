from __future__ import division

import mkl
mkl.set_num_threads(1)
from SPHARM.lib import plotting as plt
import os


path = '../../../Data/Synthetic_cells/output/'
if os.path.exists(path):
    kwargs = {'max_threads': 100, 'voxel_size': 1}

    for gridsize in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
        plt.plot_3D_surfaces(inputfolder=path + 'surfaces/',
                             outputfolder=path + 'surfaces_3D_views/gridsize=' + str(gridsize) + '/',
                             points=True, gridsize=gridsize)




