## Libraries

import time
import numpy as np
import pandas as pd
import itertools
from scipy.stats import mode
from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix

##############################################################################################################
def drop_minus_one_labels(cluster_labels, true_labels):
    '''
    Description: To be applied if clutering produced by dbscan. This function removes the noice (-1) labels 
    and the corresponding labels from the target vector.

    Parameters:
                cluster_labels (np.ndarray): labels detrmined by clustering algo=dbscan 
                true_lables (np.ndarray): labels from target vector 
    Returns:
                cluster_labels (np.ndarray): labels determined by clustering algo=dbscan wihtout noise points
                true_lables (np.ndarray): labels from target vector with labels corresponding to noice points removed.
    '''

    # concact true and cluster labels
    labels = np.concatenate((cluster_labels.reshape(-1, 1), true_labels.reshape(-1, 1)), axis=1)

    # drop noise values at each every index with -1
    labels = labels[~np.any(labels == -1, axis=1), :]
    cluster_labels = labels[:, 0]
    true_labels = labels[:, 1]

    return cluster_labels, true_labels

##############################################################################################################
##############################################################################################################

def get_modes(n_components, labels_df):
    '''
    Description: The functiont akes in the value of n_components being investigated and the labels_df and 
    return the mode and purity of the cluster_labels for the embedding with n_components. 

    Parameters:
                n_components (int): Value of n_components
                labels_df (pd.DataFrame): Combined target and cluster label dataframe
    Returns:
                Results_df (pd.nDataFrame): {
                                            'n_components': n_components,
                                            'cluster_label': i,
                                            'cluster_mode': mode_,
                                            'cluster_mode_count': value_counts.loc[mode_],
                                            'cluster_purity': value_counts.loc[mode_]/sub_array.shape[0]
                                            }
    '''

    array = labels_df[['Target', str(n_components)]].values
    df_row_dict_list = []
    for i in labels_df[str(n_components)].unique():

        sub_array = array[array[:, 1] == i]
        mode_ = mode(sub_array[:, 0]).mode

        value_counts = labels_df.loc[labels_df[str(n_components)] == i, 'Target'].value_counts()

        df_row_dict_list.append(
            {
                'n_components': n_components,
                'cluster_label': i,
                'cluster_mode': mode_,
                'cluster_mode_count': value_counts.loc[mode_],
                'cluster_purity': value_counts.loc[mode_]/sub_array.shape[0]
            }
        )
        
    results_df = pd.DataFrame(df_row_dict_list)
    return results_df.sort_values(['cluster_mode', 'cluster_mode_count'], ascending=False)

##############################################################################################################
##############################################################################################################

def get_mapping(n_components, labels_df, modes_df):
    '''
    Description: The functiont akes in the value of n_components being investigated and the labels_df and 
    return the mode and purity of the cluster_labels for the embedding with n_components. 

    Parameters:
                n_components (int): Value of n_components
                labels_df (pd.DataFrame): Combined target and cluster label dataframe
                modes_df (pd.DataFrame): Results of get_mode().
    Returns:
                cluster_mapping (dict): Dictionary containing mapping of cluster_lables onto true_labels.
    '''

    cluster_label_list = list(labels_df[str(n_components)].unique())
    true_label_list = list(labels_df.Target.unique())
    cluster_mapping = {}
    len_class_label_list = len(true_label_list)
    i = 0
    for idx, row in modes_df.sort_values(['cluster_mode', 'cluster_mode_count'], ascending=False).iterrows():
        if row['cluster_label'] in cluster_label_list and row['cluster_mode'] in true_label_list:
            cluster_mapping[row['cluster_label']] = row['cluster_mode']
            cluster_label_list.remove(row['cluster_label'])
            true_label_list.remove(row['cluster_mode'])
        else:
            i += 1
            cluster_mapping[row['cluster_label']] = len_class_label_list + i
            cluster_label_list.remove(row['cluster_label'])

    return cluster_mapping


##############################################################################################################
##############################################################################################################
def get_cont_matrix(true_labels, cluster_labels):
    '''
    Description: Gets the contingency matrix using sklearn object. 
    
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html

    Parameters:
                cluster_labels (np.ndarray): labels detrmined by clustering algo=dbscan 
                true_lables (np.ndarray): labels from target vector 
    Returns:
                cont_matrix (np.ndarray): contingency matrix where rows=true_labels and columns=cluster_labels
    '''
    cont_matrix = contingency_matrix(true_labels, cluster_labels)

    return cont_matrix

##############################################################################################################
##############################################################################################################
def get_cont_matrices(true_labels, cluster_labels, print_results=False, time_crit=False, duration=300):
    '''
    Description: This function determines the possible adjusted rand score and contingency matrices based on c;luster_label permutations. 

    Parameters:
                cluster_labels (np.ndarray): labels detrmined by clustering algo=dbscan 
                true_lables (np.ndarray): labels from target vector
    Returns:
                external_indicies_df (pd.DataFrame):
                                                    {'permutation': permutation,
                                                    'mapping': mapping,
                                                    'adjusted_rand_score': adj_rand_score,
                                                    'contingency_matrix': contingency_matrix,
                                                    'matrix_trace': matrix_trace}
    '''


    ## permute labels 
    df_row_dict_list = []
    i = 0
    duration = duration  # in seconds
    end_time = time.time() + duration
    for labels in itertools.permutations(set(cluster_labels)):

        # creating mapping between cluster_labels and permuted_cluster labls
        mapping = dict(zip(set(cluster_labels), labels))
        permuted_cluster_labels = [mapping[label] for label in cluster_labels]

        # generate contingency matrix for permuted clusters
        cont_matrix = get_cont_matrix(true_labels, permuted_cluster_labels)

        # get trace
        matrix_trace = np.trace(cont_matrix)
        
        # iterate counter
        if np.array_equal(cluster_labels, permuted_cluster_labels):            
            permutation = 'original'
        else:
            i += 1
            permutation = str(i)

        # add following results to df_row_dict_list
        df_row_dict_list.append({
                                    'permutation': permutation,
                                    'mapping': mapping,
                                    'contingency_matrix': cont_matrix,
                                    'matrix_trace': matrix_trace
                                    })
        
        # print results for each iteration 
        if print_results:

            np.set_printoptions(linewidth=200)
            print('*'*50)
            print('Permutation: ', i)
            print('Mapping:')
            print(mapping)
            print('True Labels:')
            print(set(true_labels))
            print('Premuted Cluster Labels:')
            print(set(permuted_cluster_labels))
            print('Contingency Matrix:')
            print(cont_matrix)
            print('*'*50, '\n')
        
        if time_crit:
            if time.time() > end_time:
                print("Time limit exceeded. Stopping permut.")
                break
    
    external_indicies_df = pd.DataFrame(df_row_dict_list)

    return external_indicies_df

##############################################################################################################
##############################################################################################################
def find_best_cont_matrix(external_indicies_df):
    '''
    Description: This function finds the best contingency matrix based ont the 
    permutation with the max contingency matrix trace. 

    Parameters:
                external_indicies_df (pd.DataFrame): 
                                                    {'permutation': permutation,
                                                    'mapping': mapping,
                                                    'adjusted_rand_score': adj_rand_score,
                                                    'contingency_matrix': contingency_matrix,
                                                    'matrix_trace': matrix_trace}
    Returns:
            best_indicies (pd.Series):
    '''
    max_trace_idx = external_indicies_df.loc[: , 'matrix_trace'].idxmax()

    best_indicies = external_indicies_df.iloc[max_trace_idx, :]

    return best_indicies

if __name__ == '__main__':
    pass