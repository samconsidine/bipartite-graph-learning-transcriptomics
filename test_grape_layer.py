import torch
from torch_geometric.utils import grid
from custom_sage import GrapeLayer, GrapeModule

def gen_data(n):
    return torch.normal(0, 1, (n, 10))


def test_normal_graph():
    x = gen_data(20)
    n_nodes = x.shape[0]
    edge_idx = grid(n_nodes, n_nodes)[1].T.long()
    edge_attr = torch.ones((n_nodes * n_nodes)).float().unsqueeze(0).T
    layer = GrapeLayer(10, 2, 1)
    print(f"{n_nodes=}, {x.shape=}, {edge_attr.shape=}, {edge_idx.shape=}")
    print(layer(x, edge_attr, edge_idx))

    print("Testing Module")
    print("="*80)
    module = GrapeModule(10, 6, 2, 1, 4)
    print(module(x, edge_attr, edge_idx))

def test_bipartite():
    x_1 = gen_data(20)
    x_2 = gen_data(8)
    edge_idx = torch.tensor([[x, y] for x in range(20) for y in range(8)]).T
    edge_attr = torch.ones(edge_idx.shape[1]).float().unsqueeze(0).T
    layer = GrapeLayer(10, 2, 1)
    print(f"{x_1.shape=}, {x_2.shape=}, {edge_attr.shape=}, {edge_idx.shape=}")
    print(layer((x_1, x_2), edge_attr, edge_idx, shape=(20, 8)))


if __name__=="__main__":
    test_normal_graph()
