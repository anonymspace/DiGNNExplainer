import torch
import torch_geometric
from torch_geometric.data import DataLoader

from tqdm import tqdm


from gvae import GraphVAE


from utils import (count_parameters, gvae_loss)
from config import DEVICE as device
from config import (MAX_NODES, SAMPLES, BATCH_SIZE, EPOCHS)

loss_fn = gvae_loss

def train(data_loader, type, epoch, model):


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    kl_beta = 0.5
    all_losses = []
    all_kldivs = []
    for _, batch in enumerate(tqdm(data_loader)):
        try:
            batch.to(device)
            optimizer.zero_grad()
            # Call model

            triu_logits, node_logits, feature_logits, mu, logvar = model(batch.x.float(),
                                            batch.edge_index,
                                            batch.batch)
            # Calculate loss and backpropagate
            loss, kl_div = loss_fn(triu_logits,node_logits,feature_logits,
                                   batch.edge_index, batch.node_types,batch.x, mu, logvar, batch.batch, kl_beta)
            print(f"{type} epoch {epoch} loss: ",loss)
            if type == "Train":
                loss.backward()
                optimizer.step()
                # Store loss and metrics
                all_losses.append(loss.detach().cpu().numpy())

                all_kldivs.append(kl_div.detach().cpu().numpy())
        except IndexError as error:
            print("Error: ", error)

    # Perform sampling
    if type == "Test":
        model.sample_graphs(num=SAMPLES)



def main():

    file_path = 'real_pubmed_vae/' + str(MAX_NODES) + '.pt'
    adjs, node_types, node_feats = torch.load(file_path)
    data_list = []
    for i, adj in enumerate(adjs):

        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)

        data = torch_geometric.data.Data(x=torch.tensor(node_feats[i]), node_types=torch.tensor(node_types[i]),
                                         edge_index=edge_index)
        data_list.append(data)

    train_loader = DataLoader(data_list[:140], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(data_list[:60], batch_size=BATCH_SIZE, shuffle=True)

    model = GraphVAE()
    model = model.to(device)
    print("Model parameters: ", count_parameters(model))

    for epoch in range(EPOCHS):
        model.train()
        train(train_loader, type="Train", epoch=epoch,model=model)

        if epoch % 5 == 0:
            print("Start test epoch...")
            model.eval()
            train(test_loader, type="Test",epoch=epoch,model=model)

if __name__ == '__main__':
    main()