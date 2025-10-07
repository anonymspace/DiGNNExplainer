from gnninterpreter import *
import torch
from tqdm.auto import trange

treecycle = TreeCycledataset(seed=12345)

tc_model = GCNClassifierNC(node_features=2,
                            num_classes=2,
                            hidden_channels=64,
                            num_layers=3)
#Early stopping
batch_size = len(treecycle.data.y)
best_val_acc = 0
start_patience = patience = 100
for epoch in trange(2000):
    train_loss = treecycle.fit_model_nc(tc_model, batch_size=batch_size, lr=0.001)
    train_acc, val_acc, test_acc = treecycle.test_nc(tc_model, batch_size=batch_size)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch:03d}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}, '
              f'Test Acc: {test_acc}'
            )
    if best_val_acc <= val_acc:
        print('saving....')
        patience = start_patience
        best_val_acc = val_acc
        print('best acc is', best_val_acc)
        torch.save(tc_model.state_dict(), 'ckpts/treecycle_ckpt.pt')
    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break


