from gnninterpreter import *
import torch
from tqdm.auto import trange


bashapes = BAshapesdataset(seed=12345)

bashapes_model = GCNClassifierNC(node_features=4,
                            num_classes=4,
                            hidden_channels=64,
                            num_layers=3)
#Early stopping
batch_size = len(bashapes.data.y)
best_val_acc = 0
start_patience = patience = 100
for epoch in trange(2000):
    train_loss = bashapes.fit_model_nc(bashapes_model, batch_size=batch_size, lr=0.001)
    train_acc, val_acc, test_acc = bashapes.test_nc(bashapes_model, batch_size=batch_size)

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
        torch.save(bashapes_model.state_dict(), 'ckpts/bashapes_ckpt.pt')

    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break

