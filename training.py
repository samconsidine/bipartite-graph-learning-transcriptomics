from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import pandas as pd

import torch
from torch_geometric.utils import to_dense_adj

from models.grape import GrapeModule
from dataprocessing import bipartite_graph_dataloader, load_data, batched_bipartite_graph


def train_grape(model: torch.nn.Module, X: pd.DataFrame, y: pd.DataFrame, X_val, y_val, P) -> torch.nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batches = batched_bipartite_graph(X, y, batch_size=X.shape[0])
    val_data = bipartite_graph_dataloader(X_val, y_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    node_loss_fn = torch.nn.CrossEntropyLoss() 
    edge_loss_fn = torch.nn.BCEWithLogitsLoss()
    #edge_target = to_dense_adj(data.fc_edge_index)[:data.n_cells, :data.n_genes].flatten()
    #edge_target = (data.edge_attr > 0).float()
    val_target = val_data.y

    def node_accuracy(pred, target):
        return (pred.max(1)[1] == target).float().mean().item()

    def edge_accuracy(pred, target):
        class_choice = (pred.flatten() > 0.5).long()
        return (target == class_choice).float().mean().item()

    model.train()
    prev_acc = 0
    for epoch in range(200):
        for data in batches:
            optimizer.zero_grad()
            node_pred, edge_pred = model(data.x, data.edge_attr.unsqueeze(0).T, data.edge_index)
            node_loss = node_loss_fn(node_pred[data.mask], data.y[data.mask])
            #edge_loss = edge_loss_fn(edge_pred.flatten(), edge_target)
            loss = node_loss# + edge_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            node_pred, edge_pred = model(val_data.x,
                                         val_data.edge_attr.unsqueeze(0).T, 
                                         val_data.edge_index)
            node_acc = node_accuracy(node_pred[val_data.mask], val_target[val_data.mask])
            print("Val accuracy = ", node_acc)

            prev_acc = node_acc
            if node_acc > 0.95:
                return model, node_acc

    return model, node_acc


def eval_grape(model: GrapeModule, X: pd.DataFrame, y:pd.DataFrame, *args) -> float:
    model.eval()
    data = bipartite_graph_dataloader(X, y)

    def node_accuracy(pred, target):
        return (pred.max(1)[1] == target).float().mean().item()

    pred, _ = model(data.x, data.edge_attr.unsqueeze(0).T, data.edge_index)

    return node_accuracy(pred[data.mask], data.y[data.mask])


def train_logistic_regression(
        model: LogisticRegression,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *args,
        **pca_kwargs 
    ) -> LogisticRegression:

    dim_reduction = PCA(n_components=10)
    inputs = dim_reduction.fit_transform(X, y)

    model.fit(X, y)
    return model, model.score(X, y)


def eval_logistic_regression(model: LogisticRegression, X: pd.DataFrame, y: pd.DataFrame, *args) -> float:
    return model.score(X, y)

