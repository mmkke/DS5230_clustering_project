'''
    Final Project
    Joseph Nelson Farrell & Michael Missone
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    This file contains a function that will if a pd.Dataframe as an ID attribute.
'''

def check_for_id_matrix(df):
    '''
    Checks dataframe for potential ID column. Returns list of 
    columns that have count of uniquie values equal to number of 
    rows in dataframe.
    Input: dataframe
    Output: concern_list 
    '''
    
    # Identify number of rows in df
    print(f'Number of Rows: {df.shape[0]}')

    # Init vars
    df_rows = df.shape[0]
    concern_list = []

    # Iterate columns of df, add cols with unique value count = row count to concern list
    for col in df.columns:
        if df[col].nunique == df_rows:
            print(f'{col} has {df[col].nunique} unique rows of datatype: {df[col].dtype}')    
            concern_list.append[col]

    # If concern list not emmpty dispaly list contents
    if concern_list != []:
        print(f'Potential ID columns:')
        for num, col in enumerate(concern_list):
            print(f'{num}: {col}')
        
        # return concern_list
        return concern_list
    
    else:
        print('Original dataframe does not contain ID column.')
        return
    
