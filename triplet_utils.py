import sys
sys.path.append('/home/chenjn/rna2adt')

import numpy as np
from tqdm import tqdm

from mnn_utils import *

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GEODataLoader

from dae import TriAE, AutoEncoder
    

def train_triplet(adata, hidden_dims=[128, 64], n_epochs=1000, lr=0.001, key_added='triplet',
                             gradient_clipping=5., weight_decay=0.0001, margin=1.0, verbose=False,
                             random_seed=666, iter_comb=None, knn_neigh=10, Batch_list=None,
                             device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                             batch_key='Batch', label_key='celltype.l2'):
    """\
    Train graph attention auto-encoder and use spot triplets across slices to perform batch correction in the embedding space.
    To deal with large-scale data with multiple slices and reduce GPU memory usage, each slice is considered as a subgraph for training.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added + '_emb'].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    margin
        Margin is used in triplet loss to enforce the distance between positive and negative pairs.
        Larger values result in more aggressive correction.
    iter_comb
        For multiple slices integration, we perform iterative pairwise integration. iter_comb is used to specify the order of integration.
        For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
    knn_neigh
        The number of nearest neighbors when constructing MNNs. If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    section_ids = np.array(adata.obs[batch_key].unique())

    data_list = []
    for adata_tmp in Batch_list:
        edge_index = np.nonzero(adata_tmp.uns['adj'])
        data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_index[0], edge_index[1]])),
                              prune_edge_index=torch.LongTensor(np.array([])),
                              x=torch.FloatTensor(adata_tmp.X)))

    model = AutoEncoder(adata.shape[1], adata.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)

    adata.obsm['triplet'] = adata.X
    model = model.to(device)

    print('Train with triplet loss...')
    loop = tqdm(range(n_epochs))
    for epoch in loop:
        if epoch == 0:
            if verbose:
                print('Update spot triplets at epoch ' + str(epoch))

            pair_data_list = []
            for comb in iter_comb:
                i, j = comb[0], comb[1]
                batch_pair = adata[adata.obs[batch_key].isin([section_ids[i], section_ids[j]])]
                mnn_dict = create_dictionary_mnn(batch_pair, use_rep='triplet', batch_name=batch_key, k=knn_neigh, iter_comb=None, verbose=0, batch=True)

                batchname_list = batch_pair.obs[batch_key]
                cellname_by_batch_dict = dict()
                for batch_id in range(len(section_ids)):
                    cellname_by_batch_dict[section_ids[batch_id]] = batch_pair.obs_names[
                        batch_pair.obs[batch_key] == section_ids[batch_id]].values

                anchor_list = []
                positive_list = []
                negative_list = []
                for batch_pair_name in mnn_dict.keys():  # pairwise compare for multiple batches
                    for anchor in mnn_dict[batch_pair_name].keys():
                        for v in range(len(mnn_dict[batch_pair_name][anchor])):
                            anchor_list.append(anchor)
                            positive_spot = mnn_dict[batch_pair_name][anchor][v]
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                            negative_list.append(cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])


                batch_as_dict = dict(zip(list(batch_pair.obs_names), range(0, batch_pair.shape[0])))
                anchor_ind = list(map(lambda _: batch_as_dict[_], anchor_list))
                positive_ind = list(map(lambda _: batch_as_dict[_], positive_list))
                negative_ind = list(map(lambda _: batch_as_dict[_], negative_list))

                edge_list_1 = np.nonzero(Batch_list[i].uns['adj'])
                max_ind = edge_list_1[0].max()
                edge_list_2 = np.nonzero(Batch_list[j].uns['adj'])
                edge_list_2 = (edge_list_2[0] + max_ind + 1, edge_list_2[1] + max_ind + 1)
                edge_list = [edge_list_1, edge_list_2]
                edge_pairs = [np.append(edge_list[0][0], edge_list[1][0]), np.append(edge_list[0][1], edge_list[1][1])]
                pair_data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
                                           anchor_ind=torch.LongTensor(np.array(anchor_ind)),
                                           positive_ind=torch.LongTensor(np.array(positive_ind)),
                                           negative_ind=torch.LongTensor(np.array(negative_ind)),
                                           x=adata.X))

            pair_loader = GEODataLoader(pair_data_list, batch_size=1, shuffle=True)


        model.train()
        for batch in pair_loader:
            optimizer.zero_grad()

            batch.x = torch.FloatTensor(batch.x[0])
            batch = batch.to(device)
            z, out = model(batch.x)
            mse_loss = F.mse_loss(batch.x, out)

            anchor_arr = z[batch.anchor_ind,]
            positive_arr = z[batch.positive_ind,]
            negative_arr = z[batch.negative_ind,]

            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='sum')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

            # loss = mse_loss # + 0.5 * tri_output * (mse_loss.item() / tri_output.item())
            loss = mse_loss + tri_output

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            loop.set_postfix(loss=loss.item(), mseloss=mse_loss.item())
        

    test_loader = torch.utils.data.DataLoader(adata.X, batch_size=1024, shuffle=False)
    model.eval()
    with torch.no_grad():
        z_list = []
        o_list = []
        for batch in test_loader:
            batch = batch.to(device)
            z, o = model(batch)

            z_list.append(z.cpu().detach().numpy())
            o_list.append(o.cpu().detach().numpy())
    adata.obsm[key_added + '_emb'] = np.concatenate(z_list, axis=0)
    adata.obsm[key_added + '_out'] = np.concatenate(o_list, axis=0)

    return adata