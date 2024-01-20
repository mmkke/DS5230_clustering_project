def target_to_csv(target_df, file_path):
    '''Saves target dataframe as csv in current directory.'''


    target_df.to_csv(file_path.split('.')[0] + '_target.csv', index=False)

    print(f"Target dataframe saved at: {file_path.split('.')[0]}_target.csv")

    return