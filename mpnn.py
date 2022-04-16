import torch
from torch.nn import Module, Linear, Sequential, BatchNorm1d, ReLU

import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import grid
from torch_scatter import scatter



class MPNNModel(Module):
    def __init__(self, num_layers, emb_dim, in_dim, edge_dim, out_dim):
        super().__init__()
        self.lin_in = Linear(in_dim, emb_dim)
        
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # self.pool = global_mean_pool

        self.lin_pred = Linear(emb_dim, out_dim)
        
    def forward(self, data):
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)

        # h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h) # (batch_size, d) -> (batch_size, 1)

        return out


class EdgeRegressionModel(Module):
    def __init__(self, num_layers, emb_dim, in_dim, edge_dim):
        super().__init__()
        self.lin_in = Linear(in_dim, emb_dim)
        self.edg_in = Linear(edge_dim, emb_dim)
        
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(EdgeRegressionLayer(emb_dim, emb_dim, aggr='add'))
        
        # self.pool = global_mean_pool

        self.lin_pred = Linear(emb_dim, 1)  # Out dim = 1 cause edge prediction
        
    def forward(self, data):
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        e = self.edg_in(data.fc_edge_attr.unsqueeze(1))
        
        for conv in self.convs:
            h_next, e_next = conv(h, data.fc_edge_index, e)
            h = h + h_next
            e = e + e_next

        # h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(e) # (batch_size, d) -> (batch_size, 1)

        return out


class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )
        
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), 
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class EdgeRegressionLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )
        
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), 
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr), inputs

    def update(self, aggr_out, h):
        node_emb, edge_emb = aggr_out
        upd_out = torch.cat([h, node_emb], dim=-1)
        return self.mlp_upd(upd_out), edge_emb

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')

