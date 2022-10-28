from fileinput import filename
import os
import pandas as pd


def parseDataFile():
    # fetch the file directory
    this_dir = os.getcwd()
    print(this_dir)
    # fetch the filename
    filepath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    fname = os.path.realpath('{0}/data/200_ds.txt'.format(filepath))
    print(fname)