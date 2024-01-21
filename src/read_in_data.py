'''
    Final Project
    Joseph Nelson Farrell & Michael Missone
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    This file contains a function that will read in the data from the .arff file that stored in the data folder.
'''
from scipy.io import arff
import pandas as pd
import os
import sys
from pathlib import Path

def read_data() -> pd.DataFrame:
    '''
        Function: read_in_data
        Parameters: None
        Returns: 1 pd.DataFrame
            df: beans data

        This funtion will read in the bean data from the .arff file in the data folder and return a pd.DataFrame.

        Make sure the data directory exists before executing this function!
    '''
    # set path to arff file in data folder
    file_relative_path = '/data/raw/Dry_Bean_Dataset.arff'
    
    path = Path(os.getcwd())
    print(path)
    path = str(path.parent)
    data_path = path + file_relative_path
    print(data_path)

    # load file
    data, meta = arff.loadarff(data_path)

    # convert to df
    df = pd.DataFrame(data)

    return df

df = read_data()
df.head()