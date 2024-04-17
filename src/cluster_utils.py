# src/cluster_utils.py
'''
    Final Project
    Joseph Nelson Farrell & Michael Missone
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    This file contains a libray of functions used to implement the Kmeans and DBSCAN clustering algorithms.
    

    Functions: (in order)
        KMEANS
            1. kmeans_indicies
            2. clustering_kmeans
            3. find_elbow
            4. get_NN
            5. get_hopkins
        DBSCAN
            6. find_eps
            7. factor_eps
            8. cluster_dbscan

'''

## Libraries
import numpy as np
import pandas as pd

# KMEANS
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from kneed import KneeLocator
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

# DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import DBSCAN
import hdbscan.validity as dbcv_hdbscan

## KMEANS
#########################################################################################################
# 1 #######################################################################################################

def kmeans_indices(cap_x, labels):
    '''
    Description: 
            Returns a dict with the internal indices for the kmeans clustering.
    Input:
            cap_x: embedding (ndarray)
            labels: kmeans labels (ndarray)
    Returns:
            indicies_dict = {
                            'davies_bouldin_score': davies_bouldin_score,
                            'calinski_harabasz_score': calinski_harabasz_score,
                            'silhouette_score': silhouette_score
                            }
    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

    '''

    indices_dict = {}

    # davies_bouldin_score
    db_score = davies_bouldin_score(cap_x, labels)
    indices_dict["davies_bouldin_score"] = db_score

    # calinski_harabasz_score
    cap_h = calinski_harabasz_score(cap_x, labels)
    indices_dict["calinski_harabasz_score"] = cap_h

    # silhouette_score
    sil_score = silhouette_score(cap_x, labels)
    indices_dict["silhouette_score"] = sil_score

    return indices_dict
# 2 #######################################################################################################

def clustering_kmeans(cap_x, n_clusters, df_row_dict_list):
    '''
    Description: Performs k-means clustering.

    Input:
            cap_x: embedding (ndarray)
            n_cclusters: value for n_clusters (int)
            df_row_dict_list = list for dicts of kmeans results    
    Returns:
            df_row_dict_list.append({
                                    'n_clusters': n_clusters,
                                    'inertia': inertia,
                                    'calinski_harabasz_score': indices_dict['calinski_harabasz_score'],
                                    'davies_bouldin_score': indices_dict['davies_bouldin_score'],
                                    'silhouette_score': indices_dict['silhouette_score']
                                }
    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 
    '''
    # define kmeans object and set params
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init='auto',
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=42,
        copy_x=True,
        algorithm='lloyd'
        )

    # fit k means
    kmeans.fit_predict(cap_x)


    # get lables and inertia
    labels = kmeans.labels_
    inertia = kmeans.inertia_
 
    # internal indices
    indices_dict = kmeans_indices(cap_x, labels)

    # add values to dict list
    df_row_dict_list.append(
        {
            'n_clusters': n_clusters,
            'inertia': inertia,
            'calinski_harabasz_score': indices_dict['calinski_harabasz_score'],
            'davies_bouldin_score': indices_dict['davies_bouldin_score'],
            'silhouette_score': indices_dict['silhouette_score'],
            'cluster_labels': labels
        }
    )
    return df_row_dict_list
# 3 #######################################################################################################

def find_elbow(results_df, sensitivity=1.0):
    '''
    Description: Finds the best value of n_clusters using knee plot of inerita vs n_cluster

    Input:
            results_df
            sensitivity: default=1.0
    Returns:
            n_clusters

    Documentation:
            https://kneed.readthedocs.io/en/stable/api.html#kneelocator

    '''
        
    # set kneed locator params
    curve = 'convex'
    direction = 'decreasing'
    x = results_df.n_clusters
    y = results_df.inertia

    # fine 'kneedle'
    kneedle = KneeLocator(x=x, y=y, S=sensitivity, curve=curve, direction=direction)

    try:
        #print(f'\nElbow in the inertia curve is located at {kneedle.elbow}\n')
        n_clusters = kneedle.elbow
    except Exception as e:
        print(e)
        print(f'\nCould not find elbow in the inertia curve.\n')
        n_clusters = None

    return n_clusters

# 4 #######################################################################################################

def get_NN(cap_x):
    '''
    Description: Returns nearest neighbors distances list.

    Parameters:
            cap_x (np.ndarray): design matrix
    Returns:
            nn_dist_list (list): list of nearest neighbors
            
    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html

    '''
        
    # build the kdtree
    kdt = KDTree(cap_x)

    nn_dist_list = []
    for i in range(cap_x.shape[0]):
        dist, indices = kdt.query(cap_x[i, :].reshape(1, -1), 2)
        nn_dist_list.append(dist[0, -1])

    return nn_dist_list
# 5 #######################################################################################################

def get_hopkins(cap_x):
    '''
    Description: Calculates hopkin's statistic for embedding.

        cap_h = sum(cap_x_nn_dist_list) / (sum(randomly_nn_dist_list) + sum(cap_x_nn_dist_list))

    Parameters:
            cap_x (np.ndarray): design matrix
    Returns:
            cap_h (float): hopkin's statistic value
    '''

    # seed random
    np.random.seed(18)

    # get uniformly randomley distributed data
    data_max = cap_x.max(axis=0)
    data_min = cap_x.min(axis=0)
    random_dist_data = np.random.uniform(low=data_min, high=data_max, size=cap_x.shape)

    ## null hypothesis: get nearest neighbor distance (random data)

    # get list of nearest neighbors for random data
    randomly_nn_dist_list = get_NN(random_dist_data)

    # get nearest neighbor distance from embedding
    cap_x_nn_dist_list = get_NN(cap_x)

    # calculate hopkins
    cap_h = sum(cap_x_nn_dist_list) / (sum(randomly_nn_dist_list) + sum(cap_x_nn_dist_list))

    return cap_h

## DBSCAN
#########################################################################################################
# 6 #######################################################################################################

def find_eps(cap_x, eps_knee_detection_sensitivity=3.0, metric='l2'):
        ''' 
        Description: Finds the eps and min_sample value using KneeLocator


        Parameters:
                cap_x (np.ndarray): design matrix
                eps_knee_detection_sensitivity=3.0 (default)
                metric='l2'(default)
        Returns:
                eps, min_samples

        Documentation:
            https://kneed.readthedocs.io/en/stable/api.html#kneelocator

        '''

        # build the kdtree - KDTree for fast generalized N-point problems
        kdt = KDTree(cap_x, metric=metric)

        # try a few values for k, algo was designed using 4
        k_list = [3, 4, 5]

        # init dicts
        df_row_dict_list = []
        closest_kth_dict = {}

        for k in k_list:

            # get the distance to the kth closest data object for all data objects
            closest_kth = []

            for i in range(cap_x.shape[0]):
                dist, _ = kdt.query(cap_x[i, :].reshape(1, -1), k+1)  # add 1 - the subject point with distance = 0 included
                closest_kth.append(dist[0, -1])

            # sort the distances and store in dictionary
            closest_kth_dict[k] = sorted(closest_kth)

            # set the kneed parameters to find elbow
            curve = 'convex'
            direction = 'increasing'
            drop_first_percent = 0.10
            drop_first = int(drop_first_percent * cap_x.shape[0])
            y = closest_kth_dict[k][drop_first:]  # drop first 100 to drop first concave knee

            # find the elbow and get eps
            s = eps_knee_detection_sensitivity  # found using tool by setting find_right_s above to True
            kneedle = KneeLocator(list(range(len(y))), y, S=s, curve=curve, direction=direction)
            idx = kneedle.elbow
            eps = y[idx]
            df_row_dict_list.append(
                {
                    'index': drop_first + idx,
                    'k': k,
                    'eps': eps
                }
            )

        eps_k_df = pd.DataFrame(df_row_dict_list).sort_values('eps')
        #print('\n', eps_k_df, sep='')

        #eps = eps_k_df.eps.values.max()
        eps = eps_k_df.loc[:, 'eps'].max()

        min_samples = eps_k_df.loc[eps_k_df.eps == eps, 'k'].values[0]

        return eps, min_samples

# 7 #######################################################################################################

def factor_eps(eps, eps_scan_range):
    '''
    Description: Scans around eps to find a selection of potential eps values.
    Parameters:
            eps (float): eps value
            eps_scan_range (list): [starting, stop, step]
    Returns:
            f_eps_list
    '''

    f_eps_list = []
    for factor in np.arange(eps_scan_range[0], eps_scan_range[1], eps_scan_range[2]):
          f_eps_list.append(factor*eps)
    return f_eps_list
# 8 #######################################################################################################

def cluster_dbscan(cap_x, f_eps_list, min_samples):
    '''
    Description: Performs DBSCAN clustering. Choose best value of eps from eps list based on validity index score.
    Input:
            cap_x (np.ndarray): design matrix
            f_eps_list (list): eps values
            min_samples (float): min samples value
    Returns:
            results_df = pd.DataFrame({
                                'k_dist_eps': f_eps,
                                'min_samples': min_samples,
                                'n_clusters': n_clusters,
                                'validity_index': validity_index,
                                'fitted_dbscan': dbscan
                                })
    Documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    '''

    # init dict list
    df_row_dict_list = []

    metric_list = ['euclidean', 'chebyshev', 'cosine']


    # iterate over eps values
    
    for f_eps in f_eps_list:
        for metric in metric_list:
            dist_matrix = pairwise_distances(cap_x, metric=metric)

            dbscan = DBSCAN(
                eps=f_eps,
                min_samples=min_samples,
                metric='precomputed',
                metric_params=None,
                algorithm='auto',
                leaf_size=30,
                p=None,
                n_jobs=None
            )

            # fit and get clusters
            dbscan.fit(dist_matrix)
            clusters = np.unique(dbscan.labels_)
            n_clusters = clusters[clusters != -1].shape[0]

            # get labels
            labels = dbscan.labels_

            # ensure type of embedding values for validity index, must be float64 for some reason
            dist_matrix = dist_matrix.astype(np.float64)

            # get validity index score
            try:
                validity_index = dbcv_hdbscan.validity_index(X=dist_matrix, d=cap_x.shape[1], labels=dbscan.labels_, metric='precomputed')
            except ValueError as e:
                print(e)
                validity_index = np.nan

            # add results to dict list
            df_row_dict_list.append({
                'k_dist_eps': f_eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'validity_index': validity_index,
                'fitted_dbscan': dbscan,
                'dbscan_metric': metric,
                'cluster_labels': labels
            })

    results_df = pd.DataFrame(df_row_dict_list)
    return results_df
#########################################################################################################
#########################################################################################################
if __name__ == "__main__":
    pass