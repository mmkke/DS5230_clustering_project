import numpy as np
from cluster_utils import *



def clustering(results_dict):
    print('*'*100)
    print('*'*100)
    print('Hyperparameters:')
    print('n_neighbors: ', results_dict['n_neighbors'])
    print('min_dist: ', results_dict['min_dist'])
    print('metric: ', results_dict['metric'])
    print('n_components: ', results_dict['n_components'])


    # set umap embedding as cap_x
    cap_x = results_dict['embedding']

    ## KMEANS
    #######################################################################################################################################

    # choose a range of n_clusters to try for kmeans
    n_clusters_list = np.arange(2, 16, 1)

    # init dict
    df_row_dict_list = []

    # iterate over values of n in n_clusters_list
    for n_clusters in n_clusters_list:

        # run clusters for values of n
        # results for each iteration collected in df_row_dict_list 
        clustering_kmeans(cap_x, n_clusters, df_row_dict_list)
    
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
        'cluster_labels': cluster_labels
        }
    
    # test1
    if n_clusters_found == n_clusters_db_score_is_min == n_clusters_ch_score_is_max == n_clusters_silhouette_score_is_max:
        print("Test1 Pass: Kmeans successfully clustered.")
        print('Number of Clusters: ', n_clusters_found)
        return df_row_dict
    # test2
    if  n_clusters_db_score_is_min == n_clusters_ch_score_is_max == n_clusters_silhouette_score_is_max:
        print("Test2 Pass: Kmeans successfully clustered.")
        print('Number of Clusters: ', n_clusters_found)
        return df_row_dict
    print("Fail: Kmeans did not successfully cluster.")

    ## dbscan
    ################################################################################################################################

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
            'hopkins_statistic' : cap_h,
            'umap_n_neighbors' : results_dict['n_neighbors'],
            'umap_min_dist' : results_dict['min_dist'],
            'umap_metric' : results_dict['metric'],
            'umap_n_components' : results_dict['n_components'],
            'trustworthiness' : results_dict['trustworthiness'],
            'eps' : eps,
            'dbscan_min_samples' : min_samples,
            'validity_index' : validity_index,
            'cluster_labels': cluster_label
            }


    return df_row_dict

if __name__ == "__main__":
    pass