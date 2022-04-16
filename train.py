import torch
from mpnn import MPNNModel
from scrna_graph import bipartite_graph_dataloader, load_data
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MPNNModel(4, 16, 3451, 1, 19).to(device)
data = bipartite_graph_dataloader(load_data())

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

loss_fn = torch.nn.CrossEntropyLoss() 

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print(loss.item())

    print(out)
