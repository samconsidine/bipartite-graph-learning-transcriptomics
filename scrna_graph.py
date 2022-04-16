import torch
import torch_geometric
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import scanpy as sc


def random_mask(size, proportion):
    return torch.randperm(size) < size * proportion


def bipartite_graph_dataloader(data):
    n_cells = data.X.shape[0]
    n_genes = data.X.shape[1]

    adj, edge_attr = adata_to_bipartite_adj(data)

    mask = torch.cat([torch.ones(n_cells), torch.zeros(n_genes)]).to(torch.bool)
    train_mask = random_mask(n_cells, 0.8)
    train_mask = torch.cat([train_mask, torch.zeros(n_genes).to(torch.bool)])
    y = torch.tensor(data.obs["paul15_clusters"].cat.codes.values)
    y = torch.cat([y, torch.zeros(n_genes)]).long()
    
    x = torch.cat([torch.zeros(n_cells, n_genes), torch.eye(n_genes)], axis=0)

    return Data(
        x=x,
        y=y,
        n_cells=n_cells,
        n_genes=n_genes,
        mask=mask,
        train_mask=train_mask,
        edge_index=adj,
        edge_attr=edge_attr.unsqueeze(0).T
    )


def adata_to_bipartite_adj(data):
    df = data.to_df()
    n_cells = df.shape[0]
    n_genes = df.shape[1]
    n_nodes = n_cells + n_genes

    edge_weights = torch.zeros((n_nodes, n_nodes))

    edge_weights[:n_cells, n_cells:] = torch.tensor(df.values)
    edge_weights[n_cells:, :n_cells] = torch.tensor(df.values).T
    
    return dense_to_sparse(edge_weights)
 

def load_data():
    data = sc.datasets.paul15()
    return data


if __name__ == "__main__":
    data = load_data()
    
    adj, edge_index = sc_rna_seq_to_bipartite_adj(data)

    print(f"{edge_index=}, {adj=}")
    print(f"{edge_index.max()=}")