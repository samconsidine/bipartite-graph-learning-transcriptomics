import torch
from custom_sage import GrapeModule
from scrna_graph import bipartite_graph_dataloader, load_data
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GrapeModule(1, 4, 2, 1, 19).to(device)
data = bipartite_graph_dataloader(load_data(), include_gene_idx=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

node_loss_fn = torch.nn.CrossEntropyLoss() 
edge_loss_fn = torch.nn.MSELoss()
edge_target = to_dense_adj(data.edge_index)[:data.n_cells, :data.n_genes].flatten()
node_target = data.y
print(node_target)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    node_pred, edge_pred = model(data.x, data.edge_attr, data.edge_index)
    print(node_pred)
    print(f"{node_pred.shape}")
    print(f"{node_target.shape}")
    loss  = node_loss_fn(node_pred, node_target)  
    print("Node Loss", loss.item())
    #loss += edge_loss_fn(edge_pred, edge_target) 
    #print("Node + edge Loss", loss.item())
    loss.backward()
    optimizer.step()
    #print(loss.item())
