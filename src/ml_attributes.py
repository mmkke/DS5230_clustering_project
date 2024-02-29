'''
    Final Project
    Joseph Nelson Farrell & Michael Missone
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    The file contains utility functions that will helpful in identifying non-machine learning attributes
'''
import pandas as pd
import numpy as np

def column_dtypes(df: pd.DataFrame) -> list:
    '''
        Function: column_dtypes
        Parameters: 1 pd.Dataframe
            df: the design matrix
        Returns: 1 list
            
    '''
    nominal_columns = df.select_dtypes(include=object).columns.tolist()
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

    print("********************************************************\n")

    for col in df:
        print(f'The column: {col} is data type {df[col].dtype}')

    print("********************************************************\n")

    return nominal_columns, numerical_columns