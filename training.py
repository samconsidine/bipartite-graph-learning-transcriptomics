from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import pandas as pd

import torch
from torch_geometric.utils import to_dense_adj

from models.grape import GrapeModule
from dataprocessing import bipartite_graph_dataloader, load_data


def train_grape(model: GrapeModule, X: pd.DataFrame, y: pd.DataFrame) -> GrapeModule:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = bipartite_graph_dataloader(X, y)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    node_loss_fn = torch.nn.CrossEntropyLoss() 
    edge_loss_fn = torch.nn.BCEWithLogitsLoss()
    #edge_target = to_dense_adj(data.fc_edge_index)[:data.n_cells, :data.n_genes].flatten()
    edge_target = (data.edge_attr > 0).float()
    node_target = data.y

    def node_accuracy(pred, target):
        return (pred.max(1)[1] == target).float().mean().item()

    def edge_accuracy(pred, target):
        class_choice = (pred.flatten() > 0.5).long()
        return (target == class_choice).float().mean().item()

    model.train()
    prev_acc = 0
    for epoch in range(2000):
        optimizer.zero_grad()
        node_pred, edge_pred = model(data.x, data.edge_attr.unsqueeze(0).T, data.edge_index)
        node_loss = node_loss_fn(node_pred[data.mask], node_target[data.mask])
        #edge_loss = edge_loss_fn(edge_pred.flatten(), edge_target)
        loss = node_loss# + edge_loss
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            with torch.no_grad():
                node_acc = node_accuracy(node_pred[data.mask], node_target[data.mask])
                print(f"Round finished, {node_loss.item()=}")#, {edge_loss.item()=}")
                print("Node accuracy = ", node_acc)
                #print("Edge accuracy = ", edge_accuracy(edge_pred, edge_target))
                if (prev_acc == node_acc) and (node_acc < 0.5):
                    return train_grape(model.reset(), X, y)
                prev_acc = node_acc
                if node_acc > 0.95:
                    return model

    return model


def eval_grape(model: GrapeModule, X: pd.DataFrame, y:pd.DataFrame) -> float:
    model.eval()
    data = bipartite_graph_dataloader(X, y)

    def node_accuracy(pred, target):
        return (pred.max(1)[1] == target).float().mean().item()

    pred = model(data.x, data.edge_attr.unsqueeze(0).T, data.edge_index)

    return node_accuracy(pred, y)


def train_logistic_regression(
        model: LogisticRegression,
        X: pd.DataFrame,
        y: pd.DataFrame,
        **pca_kwargs 
    ) -> LogisticRegression:

    dim_reduction = PCA(n_components=10)
    inputs = dim_reduction.fit_transform(X, y)

    model.fit(X, y)
    return model, model.score(X, y)


def eval_logistic_regression(model: LogisticRegression, X: pd.DataFrame, y: pd.DataFrame) -> float:
    return model.score(X, y)

