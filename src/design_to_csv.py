def design_to_csv(design_df, file_path):
    '''Saves design matrix dataframe as csv in current directory.'''


    design_df.to_csv(file_path.split('.')[0] + '_design.csv', index = False)

    print(f"Design dataframe saved at: {file_path.split('.')[0]}_design.csv")

    return

