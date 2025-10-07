from gnninterpreter import *
import torch
from tqdm.auto import trange

mutag = MUTAGDataset(seed=12345)
mutag_train, mutag_test = mutag.train_test_split(k=10)
mutag_model = GCNClassifier(node_features=7,
                            num_classes=2,
                            hidden_channels=64,
                            num_layers=3)


# for epoch in trange(128):
#     train_loss = mutag_train.fit_model(mutag_model, lr=0.001)
#     train_f1 = mutag_train.evaluate_model(mutag_model)
#     val_f1 = mutag_val.evaluate_model(mutag_model)
#     print(f'Epoch: {epoch:03d}, '
#           f'Train Loss: {train_loss:.4f}, '
#           f'Train F1: {train_f1}, '
#           f'Test F1: {val_f1}')

#torch.save(mutag_model.state_dict(), 'ckpts/mutag_ckpt.pt')
# #Early stopping
test_size = len(mutag_test)
best_test_acc = 0
start_patience = patience = 100

for epoch in trange(2000):
    train_loss = mutag_train.fit_model(mutag_model, lr=0.001)
    test_acc = mutag_test.test(mutag_model, test_size)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch:03d}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Test Acc: {test_acc}'
            )

    if best_test_acc <= test_acc:
        print('saving....')
        patience = start_patience
        best_test_acc = test_acc
        print('best acc is', best_test_acc)
        torch.save(mutag_model.state_dict(), 'ckpts/mutag_ckpt.pt')
    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break


