'''
    Final Project
    Joseph Nelson Farrell & Michael Missone
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    This file contains a libray of functions used to avoid false discoveries when clustering.

    Functions: (in order)

    The majority of this code is provided by: 
        Professor Steven Morin PhD.
'''

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import mode
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix

def get_randomly_distributed_data(cap_x, seed = 42, plots = False):

    data_max = cap_x.max(axis=0)
    data_min = cap_x.min(axis=0)

    np.random.seed(seed)

    randomly_distributed_data = np.random.uniform(low=data_min[0], high=data_max[0], size=(cap_x.shape[0], 1))
    for i in range(cap_x.shape[1] - 1):
        rand_i_dim = np.random.uniform(low=data_min[0], high=data_max[0], size=(cap_x.shape[0], 1))
        randomly_distributed_data = np.concatenate((randomly_distributed_data, rand_i_dim), axis=1)

    if plots and randomly_distributed_data.shape[1] == 2:
        print('\nnoise data with same number of observations')
        sns.scatterplot(x=randomly_distributed_data[:, 0], y=randomly_distributed_data[:, 1])
        plt.grid()
        plt.show()
    else:
        if plots:
            print(f'\nrandomly_distributed_data.shape[1] = {randomly_distributed_data.shape[1]} > 2 - '
                  f'no plot generated\n')

    return randomly_distributed_data

def cluster_kmeans(cap_x):
    '''
    Description: Performs k-means clustering.

        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Input:
            cap_x: embedding (ndarray)   
    Returns:
            sil_score: (float) - the silhouette score of the cluster output.
    '''
    # define kmeans object and set params
    kmeans = KMeans()

    # fit k means
    kmeans.fit_predict(cap_x)

    # get lables and inertia
    labels = kmeans.labels_

    sil_score = silhouette_score(cap_x, labels)

    return sil_score