import torch
from mpnn import EdgeRegressionModel
from scrna_graph import bipartite_graph_dataloader, load_data
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EdgeRegressionModel(2, 4, 1, 1).to(device)
data = bipartite_graph_dataloader(load_data(), include_gene_idx=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

loss_fn = torch.nn.MSELoss() 
target = data.fc_edge_attr.float()# to_dense_adj(data.edge_index)[:data.n_cells, :data.n_genes].flatten()

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out, target)  # Need to figure out the masking here
    loss.backward()
    optimizer.step()
    print(loss.item())

    print(out)
