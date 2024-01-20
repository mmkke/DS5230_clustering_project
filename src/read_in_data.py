'''
    Final Project
    Joseph Nelson Farrell & Michael Missone
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    This file contains a function that will read in the data from the arff file that stored in the data folder.
'''
from scipy.io import arff
import pandas as pd

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
    path_to_data = '/Users/nelsonfarrell/Documents/Northeastern/5230/final_project/DS5230-final/data/DryBeanDataset/Dry_Bean_Dataset.arff'

    # load file
    data, meta = arff.loadarff(path_to_data)

    # convert to df
    df = pd.DataFrame(data)

    return df
