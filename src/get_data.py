'''
    Final Project
    Joseph Nelson Farrell & Michael Missone
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    This file contains a function that will access the data (as an .arff file) from the OpenML website and create a cvs.
'''
from scipy.io import arff
from urllib.request import urlopen
import pandas as pd
import io

def get_date_from_source() -> pd.DataFrame:
    '''
        Function: get_date_from_source
        Parameters: None
        Returns: 1 pd.DataFrame
            df: wine data

        This function will access the wine data set BNG(wine) (ID:1185) from the OpenML url. This file is stored as .arff file.
        This function will import the .arff file create and return a pd.DataFrame.

        Make sure the data directory exists before executing this function!
    '''
    # url to data
    url = "https://www.openml.org/data/download/150675/BNG_wine.arff"

    # open the open
    data_stream = urlopen(url)

    # extract the data
    data, meta = arff.loadarff(io.StringIO(data_stream.read().decode('utf-8')))

    # save as pd.Dataframe
    df = pd.DataFrame(data)

    return df