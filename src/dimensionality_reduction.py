# src/dimensionality_reduction.py
'''
    Final Project
    Joseph Nelson Farrell & Michael Missone
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    This file contains a UMAP reduction function. 

    Functions: (in order)
        1. umap_dim_red

'''

## Libraries
import numpy as np
import umap
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist, squareform

################################################################################################################
# 1 #############################################################################################################

def umap_dim_red(cap_x, n_neighbors, min_dist, metric, n_components):
    '''
    Description: Performs UMAP dimensionality reduction on cap_X design matrix.

    Parameters:
                cap_x (np.ndarray): Design matrix with no ID column.

                n_neighbors (int): This parameter controls how UMAP balances local versus 
                global structure in the data. It does this by constraining the size of the 
                local neighborhood UMAP will look at when attempting to learn the manifold 
                structure of the data. 

                min_dist (float): The min_dist parameter controls how tightly UMAP is allowed 
                to pack points together. It, quite literally, provides the minimum distance 
                apart that points are allowed to be in the low dimensional representation.

                metric (str): The metric parameter controls how distance is computed in the
                ambient space of the input data. 

                n_components (int): The n_components parameter option that allows the user 
                to determine the dimensionality of the reduced dimension space we will be 
                embedding the data into.

                https://umap-learn.readthedocs.io/en/latest/parameters.html

    Returns:
                    results_dict (dict )= {
                                    'embedding' : embedding,
                                    'n_neighbors' : n_neighbors,
                                    'min_dist' : min_dist,
                                    'metric' : metric,
                                    'n_components': n_components,
                                    'trustworthiness' : trust
                                    }
    '''
    # create umap object
    reducer = umap.UMAP(
    n_neighbors = n_neighbors, 
    n_components = n_components, 
    metric = metric, 
    min_dist = min_dist, 
    spread = 1.0, 
    random_state = 42
    )

    # fit and embed
    reducer.fit(cap_x)
    embedding = reducer.transform(cap_x)

    # verify results
    assert(np.all(embedding == reducer.embedding_))
    
    # trustworthiness
    cap_x_dist = squareform(pdist(cap_x))
    cap_x_dist_embed = squareform(pdist(embedding))
    trust = trustworthiness(X=cap_x_dist, 
                             X_embedded=cap_x_dist_embed, 
                             n_neighbors=n_neighbors,
                             metric=metric)

    results_dict = {
    'embedding' : embedding,
    'n_neighbors' : n_neighbors,
    'min_dist' : min_dist,
    'metric' : metric,
    'n_components': n_components,
    'trustworthiness' : trust
    }

    return results_dict

if __name__ == "__main__":
    pass