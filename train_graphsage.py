import torch
from custom_sage import GrapeModule
from scrna_graph import bipartite_graph_dataloader, load_data
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GrapeModule(1, 4, 2, 1, 19).to(device)
data = bipartite_graph_dataloader(load_data(), include_gene_idx=False, include_fc_data=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

node_loss_fn = torch.nn.CrossEntropyLoss() 
edge_loss_fn = torch.nn.BCEWithLogitsLoss()
edge_target = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)[:data.n_cells, :data.n_genes].flatten() #This is wrong
#edge_target = to_dense_adj(data.fc_edge_index)[:data.n_cells, :data.n_genes].flatten()
edge_target = (data.fc_edge_attr > 0).float()
node_target = data.y

model.train()

def node_accuracy(pred, target):
    print(pred.max(1)[1])
    print(target)
    return (pred.max(1)[1] == target).float().mean().item()

def edge_accuracy(pred, target):
    class_choice = (pred.flatten() > 0.5).long()
    print(target.shape)
    print(class_choice.shape)
    return (target == class_choice).float().mean().item()


for epoch in range(20000):
    optimizer.zero_grad()
    print(f"{data.fc_edge_attr.shape=}")
    node_pred, edge_pred = model(data.x, data.fc_edge_attr.unsqueeze(0).T, data.fc_edge_index)
    node_loss = node_loss_fn(node_pred, node_target)  
    edge_loss = edge_loss_fn(edge_pred.flatten(), edge_target)
    print(f"Round finished, {node_loss.item()=}, {edge_loss.item()=}")
    loss = node_loss + edge_loss
    loss.backward()
    optimizer.step()

    print(f"{edge_pred.shape=}")
    print(f"{edge_target.shape=}")
    with torch.no_grad():
        print("Node accuracy = ", node_accuracy(node_pred, node_target))
        print("Edge accuracy = ", edge_accuracy(edge_pred, edge_target))
