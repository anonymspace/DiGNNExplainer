
from ba3motif_dataset import BA3Motif
import argparse
import torch
from torch_geometric.loader import DataLoader
import easydict

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

parser = argparse.ArgumentParser()
args = easydict.EasyDict({
    "dataset": 'BA3',
    "batch_size": 128,
    "hidden_channels": 64,
    "lr": 0.0005,
    #"lr":1e-3,
    "epochs": 3000,
})
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(4, args.hidden_channels)
        self.conv2 = GCNConv(args.hidden_channels, args.hidden_channels)
        self.conv3 = GCNConv(args.hidden_channels, args.hidden_channels)
        self.lin = Linear(args.hidden_channels, 3)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)

        return x

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        #print(data.batch)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        #softmax = out.softmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

test_dataset = BA3Motif('data/BA3', mode="testing")
val_dataset = BA3Motif('data/BA3', mode="evaluation")
train_dataset = BA3Motif('data/BA3', mode="training")


test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = GCN(hidden_channels=args.hidden_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

best_test_acc = 0
start_patience = patience = 100
for epoch in range(1, 300 + 1):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if epoch%100==0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

    if best_test_acc <= test_acc:
        #print('saving....')
        patience = start_patience
        best_test_acc = test_acc
        #print('best acc is', best_test_acc)

    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break

