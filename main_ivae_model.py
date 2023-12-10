import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse
import joblib
from torch import optim
from ivae_exogenous_model import iVAE

class custom_data(Dataset):
    def __init__(self, data):
        self.data = data.reshape(-1, data_np.shape[2])
        self.state = self.data[:, 0:17]
        self.action = self.data[:, 17:23]
        self.next_state = self.data[:, :17]
        
    def __getitem__(self, index):
        state = self.state[index]
        action = self.action[index]
        next_state = self.next_state[index+1]
        return state, action, next_state

    def __len__(self):
        return len(self.data)-1
    
    def get_dims(self):
        return self.state.shape[-1], self.action.shape[-1]




def vae_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=None, help='path to data file in .npy format. (default None)')
    parser.add_argument('-cf', '--cluster_file', default=None, help='path to model file in .pkl format. (default None)')
    parser.add_argument('-cn', '--cluster_num', default=20, help='path to model file in .pkl format. (default None)')
    parser.add_argument('-ad', '--aux_dim', default=5, help='path to model file in .pkl format. (default None)')
    parser.add_argument('-b', '--batch-size', type=int, default=1024, help='batch size (default 64)')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='number of epochs (default 20)')
    parser.add_argument('-t', '--task', type=str, default=None, help='use annealing in learning')
    parser.add_argument('-g', '--hidden-dim', type=int, default=50, help='hidden dim of the networks (default 50)')
    parser.add_argument('-d', '--depth', type=int, default=3, help='depth (n_layers) of the networks (default 3)')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate (default 1e-3)')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('-c', '--cuda', action='store_true', default=True, help='train on gpu')
    parser.add_argument('-a', '--anneal', action='store_true', default=True, help='use annealing in learning')
 #   parser.add_argument('-q', '--log-freq', type=int, default=25, help='logging frequency (default 25).')
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    
    args = vae_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.task = 'cheetah_run'
    args.cluster_num = 20
    args.file = '../dataset/dm_control_suite/cheetah_run/total_{}_dataset_episode.npy'.format(args.task)
    args.cluster_file = '{}_cluster_{}.pkl'.format(args.task, args.cluster_num)
    

    device = torch.device('cuda' if args.cuda else 'cpu')
    data_np = np.load(args.file)

    # load data
    train_loader = DataLoader(dataset=custom_data(data_np), batch_size=args.batch_size, shuffle=True)
    state_dim, action_dim = custom_data(data_np).get_dims()
    args.N = train_loader.__len__()
    
    aux_dim = args.aux_dim
    # define model and optimizer
    model = iVAE(latent_dim=state_dim, state_dim=state_dim, action_dim=action_dim, aux_dim=aux_dim, aux_max=args.cluster_num, activation='lrelu', device=device, hidden_dim=args.hidden_dim,
                 anneal=args.anneal) 

    #cluster model
    cluster = joblib.load(args.cluster_file)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, verbose=True)


    print('Beginning training for task: {}'.format(args.task))

    # training loop
    model.train()
    model.cuda()
    for epoch in tqdm(range(args.epochs)):
        total_elbo_train = []
        for batch_idx, (state, action, next_state) in enumerate(train_loader):
            model.anneal(args.N, args.epochs, epoch+1)
            optimizer.zero_grad()
            u = torch.tensor(cluster.predict(next_state.cpu().numpy())).long().cuda()
            embed_u = model.cat_embedding(u).cuda()
            state, action, next_state = state.cuda(), action.cuda(), next_state.cuda()
            elbo, z_est = model.elbo(next_state, embed_u, state, action)
            elbo.mul(-1).backward()
            optimizer.step()
            total_elbo_train.append(-elbo.item())
        mean_elbo = sum(total_elbo_train) / len(total_elbo_train)
        #scheduler.step(mean_elbo)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (state, action, next_state) in enumerate(train_loader):
                u = torch.tensor(cluster.predict(next_state.cpu().numpy())).long().cuda()
                embed_u = model.cat_embedding(u).cuda()
                state, action, next_state = state.cuda(), action.cuda(), next_state.cuda()
                elbo, z_est = model.elbo(next_state, embed_u, state, action)


        print('Epoch {} done; \tloss: {};'.format(epoch + 1, mean_elbo))
        torch.save(model.state_dict(), 'learned_models/ivae_'+ args.task +'.pt')