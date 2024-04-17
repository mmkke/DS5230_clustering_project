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
from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix

def get_internal_indices(cap_x, labels_pred, type_of_clustering=None, metric='euclidean'):

    unsupervised_internal_indices_list = get_unsupervised_internal_indices_list(type_of_clustering)

    # get the metrics
    results_dict = {}
    for internal_indices in unsupervised_internal_indices_list:

        metric_name_string = get_metric_name_string_from_instantiated_object(internal_indices)
        if metric_name_string == 'silhouette_score':
            if metric == 'manhattan':
                metric = 'cityblock'
            metric_value = internal_indices(cap_x, labels=labels_pred, metric=metric)
        else:
            metric_value = internal_indices(cap_x, labels=labels_pred)

        # round the results
        try:
            results_dict[metric_name_string] = round(metric_value, 4)
        except TypeError:
            results_dict[metric_name_string] = metric_value.round(4)

    return results_dict

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

def plot_the_data_sets(cap_x, y=None, data_set_name=None):

    if data_set_name is None:
        data_set_name = ''

    if y is not None:
        # plot the data with truth labels
        ax = sns.scatterplot(x=cap_x[:, 0], y=cap_x[:, 1], hue=y)
        ax.set_aspect('equal', adjustable='box')
        plt.title(f'data_set_name: {data_set_name} - with truth labels')
        plt.grid()
        plt.show()

    # plot the data without the truth labels
    ax = sns.scatterplot(x=cap_x[:, 0], y=cap_x[:, 1])
    ax.set_aspect('equal', adjustable='box')
    plt.title(f'data_set_name: {data_set_name}\nwithout truth labels')
    plt.grid()
    plt.show()


def k_means_helper(cap_x, n_clusters_list):

    return_dict = kmu.perform_k_means_clustering(cap_x, n_clusters_list)

    n_clusters = return_dict['n_clusters']
    fitted_k_means = return_dict['fitted_k_means_dict'][n_clusters]

    k_means_internal_indices_dict = \
        clu.get_internal_indices(cap_x, fitted_k_means.labels_, type_of_clustering='prototype_based',
                                 metric='euclidean')

    silhouette_score = k_means_internal_indices_dict['silhouette_score']

    print(silhouette_score)

    return {
        'silhouette_score': silhouette_score,
        'n_clusters': n_clusters,
        'fitted_k_means': fitted_k_means
    }

def avd_demo_k_means(data_set_dict, num_random_data_sets=10):

    df_row_dict_list = []
    for data_set_name, data_set in data_set_dict.items():

        print('\n', '*' * 80, '\n', '*' * 80, '\n', '*' * 80, sep='')
        print(data_set_name)

        # load the data
        cap_x = data_set['cap_x']

        # cluster the actual data with k-means
        n_clusters_list = list(range(2, 15, 1))
        df_row_dict = avd_demo_k_means_helper(cap_x, data_set_name, 'actual', n_clusters_list)
        n_clusters = df_row_dict['n_clusters']
        df_row_dict_list.append(df_row_dict)

        # create null distribution
        for i in range(0, num_random_data_sets):

            # create equivalent random data
            randomly_distributed_data = clu.get_randomly_distributed_data(cap_x, seed=i)

            # cluster the random data with k-means using n_clusters from clustering real data - this makes silhouette
            # scores comparable - cluster under identical conditions
            df_row_dict = avd_demo_k_means_helper(randomly_distributed_data, data_set_name, 'random', [n_clusters])
            df_row_dict_list.append(df_row_dict)

    results_df = pd.DataFrame(df_row_dict_list)

    # plot_the_results(results_df, data_set_dict)

    return {
        'results_df': results_df
    }

def plot_the_results(results_df, data_set_dicts):

    min_silhouette_score = results_df.silhouette_score.min() - 0.1
    # if min_silhouette_score > 0:
    #     min_silhouette_score = 0
    # else:
    #     min_silhouette_score = -1
    # max_silhouette_score = results_df.silhouette_score.max()

    for data_set_name in results_df.data_set_name.unique():

        print('\n', '*' * 80, '\n', '*' * 80, '\n', '*' * 80, sep='')
        print(data_set_name)

        # plot the data set that was clustered
        cap_x = data_set_dicts[data_set_name]['cap_x']
        plot_the_data_sets(cap_x, y=None, data_set_name=None)

        # plot the results for significance testing
        temp_results_df = results_df.loc[results_df.data_set_name == data_set_name, :]
        sns.histplot(data=temp_results_df, x='silhouette_score', hue='data_set_type')
        plt.xlim([min_silhouette_score, 1])
        plt.grid()
        plt.show()