import torch
from custom_sage import GrapeModule
from scrna_graph import bipartite_graph_dataloader, load_data
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = bipartite_graph_dataloader(load_data(), include_gene_idx=True, include_fc_data=False)

model = GrapeModule(data.x.shape[1], 4, 2, 1, 19).to(device)

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
for epoch in range(20000):
    optimizer.zero_grad()
    node_pred, edge_pred = model(data.x, data.edge_attr.unsqueeze(0).T, data.edge_index)
    node_loss = node_loss_fn(node_pred[data.mask], node_target[data.mask])
    #edge_loss = edge_loss_fn(edge_pred.flatten(), edge_target)
    loss = node_loss# + edge_loss
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        with torch.no_grad():
            print(f"Round finished, {node_loss.item()=}")#, {edge_loss.item()=}")
            print("Node accuracy = ", node_accuracy(node_pred[data.mask], node_target[data.mask]))
            print("Edge accuracy = ", edge_accuracy(edge_pred, edge_target))

