'''
    Final Project
    Joseph Nelson Farrell & Michael Missone
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    This file contains a libray of functions used to perform dimensionality reduction (UMAP) and
        clustering (Kmeans & DBscan)

    Functions: (in order)
        1. umap_dim_red
        2. cluster_kmeans
        3. dbscan
'''

# general
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from kneed import KneeLocator
from sklearn.neighbors import KDTree

from sklearn.metrics.pairwise import pairwise_distances


import hdbscan.validity as dbcv_hdbscan

# umap
import umap
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist, squareform

# clustering
from sklearn.utils import shuffle
from cluster_utils import *

def umap_dim_red(cap_x):
    ''' 
    Description:
    Params:
    Returns:
    '''
    
    # create umap object
    reducer = umap.UMAP()

    # fit and embed
    reducer.fit(cap_x)
    embedding = reducer.transform(cap_x)

    # verify results
    assert(np.all(embedding == reducer.embedding_))

    # get params
    params = reducer.get_params()

    
    # trustworthiness
    cap_x_dist = squareform(pdist(cap_x))
    cap_x_dist_embed = squareform(pdist(embedding))
    trust = trustworthiness(X=cap_x_dist, 
                             X_embedded=cap_x_dist_embed, 
                             n_neighbors=params['n_neighbors'],
                             metric=params['metric'])

    results_dict = {
    'embedding' : embedding,
    'n_neighbors' : params['n_neighbors'],
    'min_dist' : params['min_dist'],
    'metric' : params['metric'],
    'n_components': params['n_components'],
    'trustworthiness' : trust
    }

    return results_dict

def cluster_kmeans(cap_x, n_clusters, df_row_dict_list):
    '''
    Description: Performs k-means clustering.

        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

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
    '''
    # define kmeans object and set params
    kmeans = KMeans()

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

def kmeans(results_dict):
    ''' 
    Description:
    Params:
    Returns:
    '''
    print('*'*100)
    print('*'*100)
    print('Hyperparameters:')
    print('n_neighbors: ', results_dict['n_neighbors'])
    print('min_dist: ', results_dict['min_dist'])
    print('metric: ', results_dict['metric'])
    print('n_components: ', results_dict['n_components'])


    # set umap embedding as cap_x
    cap_x = results_dict['embedding']

    # choose a range of n_clusters to try for kmeans
    n_clusters_list = np.arange(2, 16, 1)

    # init dict
    df_row_dict_list = []

    # iterate over values of n in n_clusters_list
    for n_clusters in n_clusters_list:

        # run clusters for values of n
        # results for each iteration collected in df_row_dict_list 
        cluster_kmeans(cap_x, n_clusters, df_row_dict_list)
    
    # convert results dicts to dataframe
    results_df = pd.DataFrame(df_row_dict_list)

    # determine elbow location
    n_clusters_found = find_elbow(results_df, sensitivity=1.0)

    ## hopkins statistic
    cap_h = get_hopkins(cap_x)
    print(f"Hopkin's Statistic = {cap_h}")
    
    ## testing KMEANS using internal indicies
    n_clusters_db_score_is_min = results_df.loc[results_df['davies_bouldin_score'].idxmin(), 'n_clusters']
    n_clusters_ch_score_is_max = results_df.loc[results_df['calinski_harabasz_score'].idxmax(), 'n_clusters']
    n_clusters_silhouette_score_is_max = results_df.loc[results_df['silhouette_score'].idxmax(), 'n_clusters']
    sil_score = results_df.loc[results_df['silhouette_score'].idxmax(), 'silhouette_score']
    cluster_labels = results_df.loc[results_df['n_clusters'] == n_clusters_found, 'cluster_labels']

    # will return valid results in df_row_dict
    df_row_dict = {
        'algo': 'k_means',
        'n_clusters_found' : n_clusters_found,
        'n_clusters_db_score_is_min' : n_clusters_db_score_is_min,
        'n_clusters_ch_score_is_max' : n_clusters_ch_score_is_max,
        'n_clusters_silhouette_score_is_max' : n_clusters_silhouette_score_is_max,
        'silhouette_score' : sil_score,
        'hopkins_statistic' : cap_h,
        'umap_n_neighbors' : results_dict['n_neighbors'],
        'umap_min_dist' : results_dict['min_dist'],
        'umap_metric' : results_dict['metric'],
        'umap_n_components' : results_dict['n_components'],
        'trustworthiness' : results_dict['trustworthiness'],
        'eps' : np.nan,
        'dbscan_min_samples' : np.nan,
        'validity_index' : np.nan,
        'cluster_labels': cluster_labels,
        'embedding' : cap_x
        }
    
    # test1
    if n_clusters_found == n_clusters_db_score_is_min == n_clusters_ch_score_is_max == n_clusters_silhouette_score_is_max:
        print("Test1 Pass: Kmeans successfully clustered.")
        print('Number of Clusters: ', n_clusters_found)
        return df_row_dict, True
    # test2
    if  n_clusters_db_score_is_min == n_clusters_ch_score_is_max == n_clusters_silhouette_score_is_max:
        print("Test2 Pass: Kmeans successfully clustered.")
        print('Number of Clusters: ', n_clusters_found)
        return df_row_dict, True
    else:
        print("Fail: Kmeans did not successfully cluster.")
        return df_row_dict, False

def dbscan(results_dict):
    ''' 
    Description:
    Params:
    Returns:
    '''
    
    # set umap embedding as cap_x
    cap_x = results_dict['embedding']

    # get eps and min_samples from knee locator
    eps, min_samples = find_eps(cap_x)

    # iterate over a range near eps to find best eps value, determined by valididty score
    eps_scan_range = [0.8, 1.8, 0.1]
    f_eps_list = factor_eps(eps, eps_scan_range)
    
    # iterate dbscan over the eps values in f_eps_list
    results_df = cluster_dbscan(cap_x, f_eps_list, min_samples)
    
    # get values where validy score is greatest
    validity_index = results_df.loc[results_df['validity_index'].idxmax(), 'validity_index']
    eps = results_df.loc[results_df['validity_index'].idxmax(), 'k_dist_eps']
    min_samples = results_df.loc[results_df['validity_index'].idxmax(), 'min_samples']
    n_clusters_found = results_df.loc[results_df['validity_index'].idxmax(), 'n_clusters']
    cluster_label = results_df.loc[results_df['validity_index'].idxmax(), 'cluster_labels']

    print('DBSCAN')
    print('Number of Clusters: ', n_clusters_found)
    print('Validity Index: ', validity_index)

    # return results in df_row_dict
    df_row_dict = {
            'algo': 'dbscan',
            'n_clusters_found' : n_clusters_found,
            'n_clusters_db_score_is_min' : np.nan,
            'n_clusters_ch_score_is_max' : np.nan,
            'n_clusters_silhouette_score_is_max' : np.nan,
            'silhouette_score' : np.nan,
            'hopkins_statistic' : results_dict['hopkins_statistic'],
            'umap_n_neighbors' : results_dict['umap_n_neighbors'],
            'umap_min_dist' : results_dict['umap_min_dist'],
            'umap_metric' : results_dict['umap_metric'],
            'umap_n_components' : results_dict['umap_n_components'],
            'trustworthiness' : results_dict['trustworthiness'],
            'eps' : eps,
            'dbscan_min_samples' : min_samples,
            'validity_index' : validity_index,
            'cluster_labels': [cluster_label]
            }


    return df_row_dict