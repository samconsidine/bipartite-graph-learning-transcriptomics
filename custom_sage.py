import torch
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing

from torch_scatter import scatter
from torch.nn import Module, Linear, ReLU, Sequential, ModuleList


class GrapeModule(Module):
    def __init__(self, in_dim, emb_dim, n_layers, edge_dim, out_dim):
        super().__init__()
        self.edge_mlp = Linear(2*emb_dim, out_dim)
        
        self.emb = Linear(in_dim, emb_dim)

        ## GNN Layers
        self.convs = ModuleList()
        for _ in range(n_layers):
            self.convs.append(GrapeLayer(emb_dim, emb_dim, edge_dim))

        ## Node prediction MLP
        self.node_pred_fn = Linear(emb_dim, out_dim)

        ## Edge update layers
        edge_update_fn = Sequential(
            Linear(2*in_dim + edge_dim, edge_dim),
            ReLU(),
        )
        self.edge_update_fns = ModuleList([edge_update_fn])
        for _ in range(1,n_layers):
            edge_update_fn = Sequential(
                Linear(2*emb_dim + edge_dim, edge_dim),
                ReLU()
            )
            self.edge_update_fns.append(edge_update_fn)

        self.edge_prediction_fn = Sequential(
            Linear(2*emb_dim + edge_dim, 1),
        )

    def forward(self, x, edge_attr, edge_index):
        #edge_attr = self.update_edges(x, edge_attr, edge_index, self.edge_update_fns[0])
        x = self.emb(x)
        for conv, edge_update in zip(self.convs, self.edge_update_fns[1:]):
            x = conv(x, edge_attr, edge_index)
            edge_attr = self.update_edges(x, edge_attr, edge_index, edge_update)

        node_prediction = self.node_pred_fn(x)
        edge_prediction = self.predict_edges(x, edge_attr, edge_index)
        return node_prediction, edge_prediction

    @staticmethod
    def update_edges(x, edge_attr, edge_index, update_fn):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = update_fn(torch.cat((x_i, x_j, edge_attr), dim=-1))
        return edge_attr

    def predict_edges(self, x, edge_attr, edge_index):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = torch.cat((x_i, x_j, edge_attr), dim=-1)
        return self.edge_prediction_fn(edge_attr)


class GrapeLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, aggr='sum'):
        super().__init__()

        self.message_fn = Sequential(
            Linear(in_dim + edge_dim, out_dim),
            ReLU(),
        )
        self.embedding_fn = Sequential(
            Linear(out_dim + in_dim, out_dim),
            ReLU(),
        )

    def forward(self, x, edge_attr, edge_index):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)#, size=shape)

    def message(self, x_i, x_j, edge_attr, edge_index, size):
        message = torch.cat((x_j, edge_attr), dim=-1)
        return self.message_fn(message)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, x):
        aggr = torch.cat((aggr_out, x), dim=-1)
        return self.embedding_fn(aggr)

