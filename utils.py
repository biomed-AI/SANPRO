import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr
from anndata import AnnData
import scanpy as sc

def get_R(data1, data2, dim=1, func=pearsonr):
    r1, p1 = [], []
    # print(data1.shape, data2.shape)
    for g in range(data1.shape[dim]):
        if dim == 1:
            # print(np.sum(data1[:, g]), np.sum(data2[:, g]))
            r, pv = func(data1[:, g], data2[:, g])
        elif dim == 0:
            # print(np.sum(data1[g, :]), np.sum(data2[g, :]))
            r, pv = func(data1[g, :], data2[g, :])
        # print(r)
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)

    return r1, p1

def find_resolution(features, n_clusters, random=666, var=0):
    """A function to find the louvain resolution tjat corresponds to a prespecified number of clusters, if it exists.
    Arguments:
    ------------------------------------------------------------------
    - adata_: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to low dimension features.
    - n_clusters: `int`, Number of clusters.
    - random: `int`, The random seed.
    Returns:
    ------------------------------------------------------------------
    - resolution: `float`, The resolution that gives n_clusters after running louvain's clustering algorithm.
    """

    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]

    adata_ = AnnData(features)
    sc.pp.neighbors(adata_, n_neighbors=15, use_rep="X")

    while obtained_clusters != n_clusters and iteration < 100:
        current_res = sum(resolutions) / 2
        adata = sc.tl.louvain(adata_, resolution=current_res, copy=True)
        labels = adata.obs['louvain']
        obtained_clusters = len(set(labels))

        if n_clusters - obtained_clusters > var:
            resolutions[0] = current_res
        elif obtained_clusters - n_clusters > var:
            resolutions[1] = current_res

        iteration += 1

    if iteration == 100:
        print('!!! Hard !!!')

    return current_res


def find_res_label(features, n_clusters, random=666, var=0):
    """A function to find the louvain resolution tjat corresponds to a prespecified number of clusters, if it exists.
    Arguments:
    ------------------------------------------------------------------
    - adata_: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to low dimension features.
    - n_clusters: `int`, Number of clusters.
    - random: `int`, The random seed.
    Returns:
    ------------------------------------------------------------------
    - resolution: `float`, The resolution that gives n_clusters after running louvain's clustering algorithm.
    """

    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]

    adata_ = AnnData(features)
    sc.pp.neighbors(adata_, n_neighbors=15, use_rep="X")

    while abs(obtained_clusters - n_clusters) > var and iteration < 1000:
        current_res = sum(resolutions) / 2
        # print(current_res)
        adata = sc.tl.louvain(adata_, resolution=current_res, copy=True)
        labels = adata.obs['louvain']
        obtained_clusters = len(set(labels))

        if  n_clusters - obtained_clusters > var:
            resolutions[0] = current_res
        elif obtained_clusters - n_clusters > var:
            resolutions[1] = current_res
        else:
            return adata.obs['louvain']

        iteration += 1

        if iteration == 1000:
            print("Hard!!!!")
            return adata.obs['louvain']
