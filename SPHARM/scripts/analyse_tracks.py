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

        track_stat = pd.DataFrame()
        summary_stat = pd.DataFrame()
        groups = os.listdir(path + 'surfaces/')
        for gr in groups:
            print(gr)
            samples = os.listdir(path + 'surfaces/' + gr + '/')
            for sample in samples:
                print(sample)
                files = os.listdir(path + 'surfaces/' + gr + '/' + sample + '/')
                cur_summary_stat = pd.DataFrame({'Number of tracks': [len(files)],
                                                 'Sample': sample,
                                                 'Group': gr})
                cur_track_stat = pd.DataFrame()
                for fn in files:
                    print(fn)
                    stat = pd.read_csv(path + 'surfaces/' + gr + '/' + sample + '/' + fn, sep='\t', index_col=0)
                    cur_track_stat = pd.concat([cur_track_stat, pd.DataFrame({'TrackID': [stat.iloc[0]['TrackID']],
                                                                              'Track length': len(stat['Time'].unique()),
                                                                              'Start time': stat['Time'].min(),
                                                                              'End time': stat['Time'].max(),
                                                                              'File': fn,
                                                                              'Sample': sample,
                                                                              'Group': gr})],
                                               ignore_index=True)

                track_stat = pd.concat([track_stat, cur_track_stat], ignore_index=True)
                cur_summary_stat['Track length min'] = cur_track_stat['Track length'].min()
                cur_summary_stat['Track length max'] = cur_track_stat['Track length'].max()
                cur_summary_stat['Tracks >= 10 points'] = len(cur_track_stat[cur_track_stat['Track length'] >= 10])
                cur_summary_stat['Tracks >= 20 points'] = len(cur_track_stat[cur_track_stat['Track length'] >= 20])
                cur_summary_stat['Tracks >= 30 points'] = len(cur_track_stat[cur_track_stat['Track length'] >= 30])
                cur_summary_stat['Tracks >= 50 points'] = len(cur_track_stat[cur_track_stat['Track length'] >= 50])
                summary_stat = pd.concat([summary_stat, cur_summary_stat], ignore_index=True)
        summary_stat.to_csv(path + 'track_summary.csv', sep='\t')
        track_stat.to_csv(path + 'track_details.csv', sep='\t')





















