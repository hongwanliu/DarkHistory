""" Module for configuring DarkHistory.

"""
import os

data_path = '/Users/hongwan/Dropbox (MIT)/Photon Deposition/DarkHistory_data'

if data_path == '' or not os.path.isdir(data_path):
    print('NOTE: enter data directory in config.py to avoid this step.')
    data_path = input('Enter the data directory, e.g. /Users/foo/bar: ')
