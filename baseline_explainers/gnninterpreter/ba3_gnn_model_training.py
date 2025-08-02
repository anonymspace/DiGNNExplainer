from gnninterpreter import *
import torch
from tqdm.auto import trange

ba3 = Ba3dataset(seed=12345)

ba3_train, ba3_test = ba3.train_test_split(k=10)
ba3_model = GCNClassifier(node_features=4,
                            num_classes=3,
                            hidden_channels=64,
                            num_layers=3)
#Early stopping
test_size = len(ba3_test)
best_test_acc = 0
start_patience = patience = 100
for epoch in trange(300):
    train_loss = ba3_train.fit_model(ba3_model, lr=0.001)
    test_acc = ba3_test.test(ba3_model,test_size)

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
        torch.save(ba3_model.state_dict(), 'ckpts/ba3_ckpt.pt')
    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break


