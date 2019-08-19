from __future__ import division

import os
import sys
import pandas as pd


#################################
args = sys.argv[1:]
if len(args) > 0:
    path = args[0]
    if path != 'test':
        if not path.endswith('/'):
            path += '/'
        path += 'output/'

        surface_stat = pd.DataFrame()
        groups = os.listdir(path + 'surfaces/')
        for gr in groups:
            print(gr)
            samples = os.listdir(path + 'surfaces/' + gr + '/')
            for sample in samples:
                print(sample)
                files = os.listdir(path + 'surfaces/' + gr + '/' + sample + '/')
                for fn in files:
                    print(fn)
                    stat = pd.read_csv(path + 'surfaces/' + gr + '/' + sample + '/' + fn, sep='\t', index_col=0)
                    times = stat['Time'].unique()

                    for t in times:
                        curstat = stat[stat['Time'] == t]
                        cur_surface_stat = pd.DataFrame({'Time': [t],
                                                         'Number of surface points': [len(curstat)],
                                                         'Group': gr,
                                                         'Sample': sample,
                                                         'File': fn,
                                                         'Number of unique IDs': len(curstat['ID'].unique())})
                        surface_stat = pd.concat([surface_stat, cur_surface_stat], ignore_index=True)
        surface_stat.to_csv(path + 'surface_time_summary.csv', sep='\t')























