#!/usr/bin/env python3

import argparse
from tqdm import tqdm
import os
import sys
sys.path.append('/home/chenjn/biock')

import torch
import h5py
from torch import Tensor
import numpy as np
import torch.nn as nn
import episcanpy as esp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Any, Dict, Iterable, List, Literal, Optional, Union
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, vstack
from biock import random_string
from biock.genomics.single_cell import tfidf_transform
import utils
from biock import HUMAN_CHROMS_NO_Y_MT
import logging
logger = logging.getLogger(__name__)

LIBRARY_SIZE_KEY = "__library_size__"
RAWCOUNT_KEY = "__raw_count__"
ATAC_LIB_SIZE = 1000
RNA_LIB_SIZE = 2000
LIB_SCALE = 1000

def get_adata_stats(adata: AnnData) -> Dict[str, Any]:
    stats = {
        "shape": adata.shape,
        "X.data(min/mean/max)": (adata.X.data.min(), adata.X.data.mean(), adata.X.data.max()),
        "density": round(np.sum(adata.X.data > 0) / adata.shape[0] / adata.shape[1], 4)
    }
    return stats


def load_adata(
        h5ad: Union[str, List[str]], 
        log1p: bool, binarize: bool, tfidf: bool, 
        min_cells: int=None, max_cells: int=None, 
        min_genes: int=None, max_genes: int=None, 
        # clip_high: float=0
    ) -> AnnData:
    """
    clip_high: remove outliers with extremely high values
    keep_counts: keep raw values
    """
    assert not (log1p and binarize), "log1p and binarize should not be used simutanously"
    # assert clip_high < 0.5, "clip ratio should be below 0.5 (50%)"

    adata = sc.read_h5ad(h5ad)

    if type(adata.X) is not csr_matrix:
        adata.X = csr_matrix(adata.X)
    raw_shape = adata.shape

    ## filtering
    if min_cells is not None:
        if min_cells < 1:
            min_cells = int(round(min_cells * adata.shape[0]))
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if max_cells is not None:
        if max_cells < 1:
            max_cells = int(round(max_cells * adata.shape[0]))
        sc.pp.filter_genes(adata, max_cells=max_cells)
    logger.info("  filtering gene: {}->{}".format(raw_shape, adata.shape))

    if min_genes is not None:
        if min_genes < 1:
            min_genes = int(round(min_genes * adata.shape[1]))
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        if max_genes < 1:
            max_genes = int(round(max_genes * adata.shape[1]))
        sc.pp.filter_cells(adata, max_genes=max_genes)
    logger.info("filtering cell: {}->{}".format(raw_shape, adata.shape))
    logger.info("stats after filtering: {}".format(get_adata_stats(adata)))

    if log1p:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        logger.info("total normalized and log-transformed, stats: {}".format(get_adata_stats(adata)))
    elif binarize:
        adata.X.data = (adata.X.data > 0).astype(np.float32)
        logger.info("binarization transformation, stats: {}".format(get_adata_stats(adata)))

    if tfidf:
        if adata.X.data.max() > 1:
            logger.warning("X in adata has not been binarized!")
        adata.X = tfidf_transform(adata.X, norm=None)
        logger.info("- using 'None' norm in TfidfTransformer")
    logger.info("finished")

    # half float max
    hf_max = np.finfo(np.float16).max
    if adata.X.max() > hf_max:
        logger.warning("values in X exceeding {} were set to {}".format(hf_max, hf_max))
        adata.X.data = np.minimum(hf_max, adata.X.data)

    if "CellType" in adata.obs.columns and "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"]=adata.obs["CellType"]

    return adata


class PairedModalDataset(Dataset):
    def __init__(self,
                 atac: AnnData,
                 mrna: AnnData,
                #  emb: any,
                 select_peak: Literal["var", "flanking"],
                 genome, 
                 seq_len=1344,
                 n_top_genes=None,
                 n_top_peaks=None,
                 flanking: int=100000,
                 aug_func: str=None,
                 aug_num: int=None,
                 **kwargs):
        
        super(PairedModalDataset, self).__init__()
        # logger.info("- {} info:{}".format(__class__.__name__,))
        self.aug_func = aug_func
        self.aug_num = aug_num

        self.seq_len = seq_len
        self.genome = h5py.File(genome, 'r')

        if select_peak == "var":
            assert n_top_peaks is not None, "n_top_peaks is required to select peak by `var`"

        # if "chrom" not in mrna.var.columns:
        #     assert gene_info is not None, "gene_info is required to obtain tss information"
        #     logger.info("  add gene info ...")
        #     mrna = utils.add_gene_info(mrna, gene_info=gene_info)
        if "chr" not in atac.var.columns:
            logger.info("  add chrom info ...")
            atac.var["chr"] = [c.split(':')[0] for c in atac.var.index]
            atac.var["start"] = [c.split(':')[1].split('-')[0] for c in atac.var.index]
            atac.var["end"] = [c.split(':')[1].split('-')[1] for c in atac.var.index]
        
        if not np.array_equal(atac.obs.index, mrna.obs.index):
            common_index = sorted(list(
                set(atac.obs.index).intersection(set(mrna.obs.index))
            ))
            atac = atac[common_index, :]
            mrna = mrna[common_index, :]
            logger.warning("Conflicting index has been fixed")
        
        # select features: mRNA: high-variable; ATAC: neighboring 100kbp
        raw_mrna_shape, raw_atac_shape = mrna.shape, atac.shape
        print(mrna.shape, atac.shape)
        sc.pp.highly_variable_genes(mrna, subset=True, n_top_genes=n_top_genes)
        logger.info("mRNA: {} -> {}".format(raw_mrna_shape, mrna.shape))

        if select_peak == "flanking":
            peaks_kept = utils.select_neighbor_peaks(mrna.var["tss"], atac.var.index, flanking=flanking)
            atac = atac[:, peaks_kept]
        else:
            logger.warning("- experimtal feature to select peaks by var")
            atac = esp.pp.select_var_feature(atac, nb_features=n_top_peaks, show=False, copy=True)
        logger.info("ATAC: {} -> {}".format(raw_atac_shape, atac.shape))

        self.n_cells = atac.shape[0]
        self.n_genes = mrna.shape[1]
        self.n_peaks = atac.shape[1]
        self.batche_ids = None

        self.atac_obs, self.atac_var = atac.obs.copy(), atac.var.copy()
        del atac.obs, atac.var
        self.atac_X = atac.X.copy()
        del atac.X, atac

        self.mrna_obs, self.mrna_var = mrna.obs.copy(), mrna.var.copy()
        del mrna.obs, mrna.var
        self.mrna_X = mrna.X.copy()
        del mrna.X, mrna

        # self.emb = emb
        # del emb

    def __len__(self):
        return self.n_cells

    def __getitem__(self, index):
        a_x = self.atac_X[index].toarray().flatten()
        m_x = self.mrna_X[index].toarray().flatten()
        # emb_x = self.emb[index].flatten()

        # chrom, start, end = self.atac_var["chr"][index], self.atac_var["start"][index], self.atac_var["end"][index]
        # mid = (int(start) + int(end)) // 2
        # left, right = mid - self.seq_len // 2, mid + self.seq_len // 2
        # left_pad, right_pad = 0, 0
        # if left < 0:
        #     left_pad = -left_pad
        #     left = 0
        # if right > self.genome[chrom].shape[0]:
        #     right_pad = right - self.genome[chrom].shape[0]
        #     right = self.genome[chrom].shape[0]
        # seq = self.genome[chrom][left:right]
        # if len(seq) < self.seq_len:
        #     seq = np.concatenate((
        #         np.full(left_pad, -1, dtype=seq.dtype),
        #         seq,
        #         np.full(right_pad, -1, dtype=seq.dtype),
        #     ))
        # # if strand == '-':
        # #     seq = get_reverse_strand(seq, integer=True)

        # return emb_x, a_x, m_x
        return a_x, m_x

class PairedModalDataset2s(Dataset):
    def __init__(self,
                 atac: AnnData,
                 mrna: AnnData,
                 emb: any,
                 select_peak: Literal["var", "flanking"],
                 genome, 
                 seq_len=1344,
                 n_top_genes=None,
                 n_top_peaks=None,
                 flanking: int=100000,
                 aug_func: str=None,
                 aug_num: int=None,
                 **kwargs):
        
        super(PairedModalDataset, self).__init__()
        # logger.info("- {} info:{}".format(__class__.__name__,))
        self.aug_func = aug_func
        self.aug_num = aug_num

        self.seq_len = seq_len
        self.genome = h5py.File(genome, 'r')

        if select_peak == "var":
            assert n_top_peaks is not None, "n_top_peaks is required to select peak by `var`"

        # if "chrom" not in mrna.var.columns:
        #     assert gene_info is not None, "gene_info is required to obtain tss information"
        #     logger.info("  add gene info ...")
        #     mrna = utils.add_gene_info(mrna, gene_info=gene_info)
        if "chr" not in atac.var.columns:
            logger.info("  add chrom info ...")
            atac.var["chr"] = [c.split(':')[0] for c in atac.var.index]
            atac.var["start"] = [c.split(':')[1].split('-')[0] for c in atac.var.index]
            atac.var["end"] = [c.split(':')[1].split('-')[1] for c in atac.var.index]
        
        if not np.array_equal(atac.obs.index, mrna.obs.index):
            common_index = sorted(list(
                set(atac.obs.index).intersection(set(mrna.obs.index))
            ))
            atac = atac[common_index, :]
            mrna = mrna[common_index, :]
            logger.warning("Conflicting index has been fixed")
        
        # select features: mRNA: high-variable; ATAC: neighboring 100kbp
        raw_mrna_shape, raw_atac_shape = mrna.shape, atac.shape
        print(mrna.shape, atac.shape)
        sc.pp.highly_variable_genes(mrna, subset=True, n_top_genes=n_top_genes)
        logger.info("mRNA: {} -> {}".format(raw_mrna_shape, mrna.shape))

        if select_peak == "flanking":
            peaks_kept = utils.select_neighbor_peaks(mrna.var["tss"], atac.var.index, flanking=flanking)
            atac = atac[:, peaks_kept]
        else:
            logger.warning("- experimtal feature to select peaks by var")
            atac = esp.pp.select_var_feature(atac, nb_features=n_top_peaks, show=False, copy=True)
        logger.info("ATAC: {} -> {}".format(raw_atac_shape, atac.shape))

        self.n_cells = atac.shape[0]
        self.n_genes = mrna.shape[1]
        self.n_peaks = atac.shape[1]
        self.batche_ids = None

        self.atac_obs, self.atac_var = atac.obs.copy(), atac.var.copy()
        del atac.obs, atac.var
        self.atac_X = atac.X.copy()
        del atac.X, atac

        self.mrna_obs, self.mrna_var = mrna.obs.copy(), mrna.var.copy()
        del mrna.obs, mrna.var
        self.mrna_X = mrna.X.copy()
        del mrna.X, mrna

        self.emb = emb
        del emb

    def __len__(self):
        return self.n_cells

    def __getitem__(self, index):
        a_x = self.atac_X[index].toarray().flatten()
        m_x = self.mrna_X[index].toarray().flatten()
        emb_x = self.emb[index].flatten()

        # chrom, start, end = self.atac_var["chr"][index], self.atac_var["start"][index], self.atac_var["end"][index]
        # mid = (int(start) + int(end)) // 2
        # left, right = mid - self.seq_len // 2, mid + self.seq_len // 2
        # left_pad, right_pad = 0, 0
        # if left < 0:
        #     left_pad = -left_pad
        #     left = 0
        # if right > self.genome[chrom].shape[0]:
        #     right_pad = right - self.genome[chrom].shape[0]
        #     right = self.genome[chrom].shape[0]
        # seq = self.genome[chrom][left:right]
        # if len(seq) < self.seq_len:
        #     seq = np.concatenate((
        #         np.full(left_pad, -1, dtype=seq.dtype),
        #         seq,
        #         np.full(right_pad, -1, dtype=seq.dtype),
        #     ))
        # # if strand == '-':
        # #     seq = get_reverse_strand(seq, integer=True)

        return emb_x, a_x, m_x
        # return a_x, m_x
    
