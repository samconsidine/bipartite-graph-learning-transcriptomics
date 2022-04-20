import torch
import torch_geometric
from torch_geometric.utils import dense_to_sparse, to_undirected
from torch_geometric.data import Data
import scanpy as sc
from sklearn.model_selection import train_test_split
import pandas as pd

from torch.utils.data import DataLoader

from typing import List

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def random_mask(size, proportion):
    return torch.randperm(size) < size * proportion


def convert_to_undigraph(edge_index, edge_attr):
    new_edges = torch.cat((edge_index, torch.flip(edge_index, [0])), dim=-1)
    new_attrs = torch.cat((edge_attr, edge_attr), dim=-1)
    return new_edges.to(device), new_attrs.to(device)


def bipartite_graph_dataloader(X, y, include_fc_data=False, undirected=True):
    n_cells = X.shape[0]
    n_genes = X.shape[1]

    adj, edge_attr = adata_to_bipartite_adj(X)
    if undirected:
        mask = adj[0] >= n_cells
        adj = adj[:, mask]
        edge_attr = edge_attr[mask]

    mask = torch.cat([torch.ones(n_cells), torch.zeros(n_genes)]).to(torch.bool)
    train_mask = random_mask(n_cells, 0.8)
    train_mask = torch.cat([train_mask, torch.zeros(n_genes).to(torch.bool)])
    y = torch.tensor(y.cat.codes.values)
    y = torch.cat([y, torch.zeros(n_genes)]).long().to(device)
    
    inputs = torch.cat([torch.zeros([n_cells, n_genes]), torch.eye(n_genes)], dim=0).to(device)

    from_nodes = torch.repeat_interleave(torch.arange(n_cells), n_genes)
    to_nodes = torch.repeat_interleave(torch.arange(n_cells, n_genes+n_cells), n_cells)
    fc_edge_index = torch.stack([from_nodes, to_nodes]).long()

    fc_edge_attr = torch.tensor(X.values.flatten())
    assert fc_edge_attr.shape[0] == fc_edge_index.shape[1]
    
    #fc_edge_index, fc_edge_attr = convert_to_undigraph(fc_edge_index, fc_edge_attr)

    return Data(
        x=inputs,
        y=y,
        n_cells=n_cells,
        n_genes=n_genes,
        mask=mask,
        train_mask=train_mask,
        edge_index=adj,
        edge_attr=edge_attr,
        fc_edge_index=fc_edge_index,
        fc_edge_attr=fc_edge_attr
    )


def batched_bipartite_graph(X, y, batch_size: int) -> List[Data]:
    batched = []

    for idx in range(0, X.shape[0], batch_size):
        batched.append(bipartite_graph_dataloader(X[idx:(idx+batch_size)], y[idx:(idx+batch_size)]))

    return batched



def adata_to_bipartite_adj(df):
    n_cells = df.shape[0]
    n_genes = df.shape[1]
    n_nodes = n_cells + n_genes

    edge_weights = torch.zeros((n_nodes, n_nodes)).to(device)

    edge_weights[:n_cells, n_cells:] = torch.tensor(df.values)
    edge_weights[n_cells:, :n_cells] = torch.tensor(df.values).T

    return dense_to_sparse(edge_weights)
 

def load_data(config):
    data = sc.datasets.paul15()
    data.X = data.X.astype('float32')
    if config.n_genes is not None:
        sc.pp.recipe_zheng17(data, n_top_genes=config.n_genes)
    df = data.to_df()
    df['target'] = data.obs['paul15_clusters']
    df = df.dropna(subset=['target'])
    df['target'] = df.target.astype('category')
    train, test = train_test_split(df, random_state=42)
    train_X = train[train.columns[:-1]]
    train_y = train['target']
    test_X = test[test.columns[:-1]]
    test_y = test['target']
    return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    data = load_data()
    
    adj, edge_index = sc_rna_seq_to_bipartite_adj(data)

    print(f"{edge_index=}, {adj=}")
    print(f"{edge_index.max()=}")
