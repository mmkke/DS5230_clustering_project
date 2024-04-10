import numpy as np
import umap
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist, squareform

def umap_dim_red(cap_x, n_neighbors, min_dist, metric, n_components):
    
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