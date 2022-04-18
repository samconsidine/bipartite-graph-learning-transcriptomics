import torch
import torch_geometric
from torch_geometric.utils import dense_to_sparse, to_undirected
from torch_geometric.data import Data
import scanpy as sc


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def random_mask(size, proportion):
    return torch.randperm(size) < size * proportion

def convert_to_undigraph(edge_index, edge_attr):
    new_edges = torch.cat((edge_index, torch.flip(edge_index, [0])), dim=-1)
    new_attrs = torch.cat((edge_attr, edge_attr), dim=-1)
    return new_edges, new_attrs

def bipartite_graph_dataloader(data, include_gene_idx, include_fc_data=False):
    n_cells = data.X.shape[0]
    n_genes = data.X.shape[1]

    adj, edge_attr = adata_to_bipartite_adj(data)

    mask = torch.cat([torch.ones(n_cells), torch.zeros(n_genes)]).to(torch.bool)
    train_mask = random_mask(n_cells, 0.8)
    train_mask = torch.cat([train_mask, torch.zeros(n_genes).to(torch.bool)])
    y = torch.tensor(data.obs["paul15_clusters"].cat.codes.values)
    y = torch.cat([y, torch.zeros(n_genes)]).long().to(device)
    
    if include_gene_idx:
        x = torch.cat([torch.zeros(n_cells, n_genes), torch.eye(n_genes)], axis=0).to(device)
    else:
        x = torch.zeros((n_cells+n_genes,1)).to(device)

    from_nodes = torch.repeat_interleave(torch.arange(n_cells), n_genes)
    to_nodes = torch.repeat_interleave(torch.arange(n_cells, n_genes+n_cells), n_cells)
    fc_edge_index = torch.stack([from_nodes, to_nodes]).long()

    fc_edge_attr = torch.tensor(data.X.flatten())
    assert fc_edge_attr.shape[0] == fc_edge_index.shape[1]

    # if include_fc_data:
    #     fc_edge_index, fc_edge_attr = to_undirected(fc_edge_index, fc_edge_attr)
    # else:
    #     fc_edge_index, fc_edge_attr = None, None

    fc_edge_index, fc_edge_attr = convert_to_undigraph(fc_edge_index, fc_edge_attr)

    return Data(
        x=x,
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


def adata_to_bipartite_adj(data):
    df = data.to_df()
    n_cells = df.shape[0]
    n_genes = df.shape[1]
    n_nodes = n_cells + n_genes

    edge_weights = torch.zeros((n_nodes, n_nodes)).to(device)

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
