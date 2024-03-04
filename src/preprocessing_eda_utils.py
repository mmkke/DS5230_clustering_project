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
import itertools
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

####################################################################################################################
####################################################################################################################
def missingness_cols(df: pd.DataFrame, threshdold: float) -> list:
    '''
        Function: missingness_cols
        Parameters: 1 pd.Dataframe, 1 float
            df: the dataframe whose columns will checked for missingess
            threshold: the threshold proportion to determine if column should be dropped
        Returns: 1 list
            cols_to_drop: the columns whose missingness exceeds threshold. 

        The function will find the proportion of missingness for each column in a dataframe and return a list of 
        columns whose missingness proportion exceeds the threshold
    '''
    # instantiate drop list
    cols_to_drop = []

    # iterate over columns
    for col in df.columns:

        # find missingness proportion
        missing_proportion = df[col].isna().sum() / df.shape[0]

        # check against threshold; update list
        if missing_proportion >= threshdold:
            cols_to_drop.append(col)

    return cols_to_drop

####################################################################################################################
####################################################################################################################

def column_dtypes(df: pd.DataFrame) -> list:
    '''
        Function: column_dtypes
        Parameters: 1 pd.Dataframe
            df: the design matrix
        Returns: 3 lists: nominal_columns, numerical_columns, unique_value_columns

        This function will find the nominal and numerical columns and return a list containing each set.
        It also return a list of the number of unique values in each column.  
    '''
    nominal_columns = df.select_dtypes(include=object).columns.tolist()
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    unique_value_columns = []

    for col in df:
        print(f'Column: {col}')
        print(f'Data Type: {df[col].dtype}')
        print(f'Unique value count: {df[col].nunique()}, DF length: {len(df)}, Ratio: {round(df[col].nunique()/len(df), 2)}')
        if df[col].nunique()/len(df) == 1:
            print(f'***\nFLAG column {col} for review \n***')
            unique_value_columns.append(col)
        print("__________________________________________________________\n")

    return nominal_columns, numerical_columns, unique_value_columns

####################################################################################################################
####################################################################################################################

def compute_vif(design_df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    '''
        Function: compute_vif
        Parameters: 1 pd.DataFrame, 1 list
            design_df: a pandas dataframe design matrix
            numerical_cols: a list of numerical columns with design_df to use when computing the 
                VIF score.
        Returns: 1 pd.DataFrame
            vif: a pandas dataframe containing the vif scores. 

        This function will create a pd dataframe VIF factors for each of the numeric columns
    '''
    # filter the design matrix for numerical columns
    X = design_df.loc[: , numerical_cols]

    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!= 'intercept']

    # display results
    print('Variance Inflation Factors Above Threshold(5):\n')
    print(vif[vif['VIF'] > 5])

    print('\n\nVariance Inflation Factors Below Threshold(5):\n')
    print(vif[vif['VIF'] <= 5])

    return vif

####################################################################################################################
####################################################################################################################
def generate_column_hist(df: pd.DataFrame, columns: list) -> None:
    '''
        Function: generate_col_hist
        Parameters: 1 pd.DataFrame, 1 list
            design_df: a pandas dataframe design matrix
            numerical_cols: a list of numerical columns with design_df to use when computing the 
                VIF score.
        Returns: None

        This functions will create a 4x4 plt subplot of histgrams.
    '''
    # plot params
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'lightblue'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['lines.color'] = 'white'

    # generate fig
    fig, axes = plt.subplots(2,2, figsize = (12, 12))
    fig.suptitle(f'Numerical Attribute Histograms', weight = 'bold', fontsize = 20, y = .93)

    # flatten axes
    axes_flat = axes.ravel()

    # generate subplots
    for i, col in enumerate(columns):
        ax = axes_flat[i]
        ax.hist(df[col], bins = 'auto', color = 'firebrick', edgecolor = 'white', linewidth = .3)
        ax.tick_params(axis= 'both', labelsize = 7)
        ax.grid(color = 'white')
        ax.set_xlabel(f'{col}', weight = 'bold', style = 'italic')
        ax.set_ylabel('Count', weight = 'bold', style = 'italic')
    
    return None

####################################################################################################################
####################################################################################################################
def sub_divide_pairplot(trans_df: pd.DataFrame) -> None:
    '''
        Function: sub_divide_pairplot
        Parameters: 1 pd.DataFrame
            trans_df: a pandas dataframe design matrix
        Returns: None

        This functions will create 4x4 plt subplots of all pairplots from a attribute matrix with 16 attributes. 
    '''

    # plot params
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'lightblue'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['lines.color'] = 'white'

    # Get all combinations of length 2 (pairs)
    pairs = list(itertools.product(trans_df.columns, repeat=2))

    # Iterate through the list, selecting 8 pairs at a time
    for i in range(0, len(pairs), 16): 
        selected_pairs = pairs[i:i+16]
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))

        for idx, pair in enumerate(selected_pairs):
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]
            if pair[0] == pair[1]:
                ax.hist(trans_df[pair[0]], color = 'firebrick',edgecolor = 'white', bins='auto')
                ax.set_xlabel(pair[0].split("_")[2])
                ax.set_ylabel(pair[1].split("_")[2])
            else:
                ax.scatter(trans_df[pair[0]], trans_df[pair[1]], alpha=0.8, color = 'firebrick', edgecolor = 'white')
                ax.set_xlabel(pair[0].split("_")[2])
                ax.set_ylabel(pair[1].split("_")[2])

        fig.suptitle(f'Pairplots: {pair[0].split("_")[2]}', weight = 'bold', fontsize = 20, y=1.01)
        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.savefig(f'../figs/pairplots_{pair[0].split("_")[2]}.png', bbox_inches = 'tight')
        plt.show();

    return None