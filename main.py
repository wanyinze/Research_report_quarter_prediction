import torch
import argparse
import numpy as np
import pandas as pd
from torch import optim
from vae import VAE, vae_loss_function
from mlp import MLP, mlp_loss_function

parser = argparse.ArgumentParser(description='Predict Revenue')
parser.add_argument('--x-dim', type=int, default=4, metavar='N',
                    help='column dimension of the input X')
parser.add_argument('--i-dim', type=int, default=13, metavar='N',
                    help='column dimension of the input I')
parser.add_argument('--z-dim', type=int, default=16, metavar='N',
                    help='column dimension of the input Z')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
args = parser.parse_args()


vae = VAE(args.x_dim, args.i_dim, args.z_dim)
mlp = MLP(args.z_dim, args.i_dim)
vae_optimizer = optim.Adam(vae.parameters(), lr=1e-2)
mlp_optimizer = optim.Adam(mlp.parameters(), lr=1e-2)


def train_vae(epoch):
    vae.train()
    train_loss = 0
    for batch_idx in range(vae_batch_num):
        data = vae_samples[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
        x = data[:, :args.x_dim]
        vae_optimizer.zero_grad()
        recon_x, mu, logvar = vae(data)
        loss = vae_loss_function(recon_x, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        vae_optimizer.step()
        if (batch_idx+1) % (vae_batch_num/5) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * args.batch_size, vae_samples.size(0),
                100. * (batch_idx+1) / vae_batch_num,
                loss.item() / args.batch_size))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / vae_samples.size(0)))


def train_mlp(epoch):
    mlp.train()
    train_loss = 0
    for batch_idx in range(mlp_batch_num):
        data = mlp_samples[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
        target = mlp_target[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
        mlp_optimizer.zero_grad()
        prediction = mlp(data)
        loss = mlp_loss_function(prediction, target)
        loss.backward()
        train_loss += loss.item()
        mlp_optimizer.step()
        if (batch_idx+1) % (mlp_batch_num/5) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * args.batch_size, mlp_samples.size(0),
                100. * (batch_idx+1) / mlp_batch_num,
                loss.item() / args.batch_size))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / mlp_samples.size(0)))


if __name__ == "__main__":

    # samples = pd.read_csv("./data/train_data.csv")
    # vae_samples = torch.tensor(np.matrix(samples.iloc[:, 2:2+args.x_dim+args.i_dim]), dtype=torch.float32)
    vae_samples = torch.randn(args.batch_size*5, args.x_dim+args.i_dim)  # 随机生成的数据可以训练
    vae_batch_num = vae_samples.size(0)//args.batch_size
    for epoch in range(1, 100+1):
        train_vae(epoch)

    print("="*100)

    Z = vae.sample_z(vae_samples).clone().detach_()  # 利用训练好的VAE对vae_samples进行encode生成隐变量Z
    I = vae_samples[:, -args.i_dim:]
    mlp_samples = torch.cat((Z, I), 1)
    # mlp_target = torch.tensor(np.matrix(samples.loc[:, '下一期营收']), dtype=torch.float32).view(-1, 1)  # 公司下一个季度的Revenue
    mlp_target = torch.randn(args.batch_size*5, 1)
    mlp_batch_num = mlp_samples.size(0)//args.batch_size

    for epoch in range(1, 100+1):
        train_mlp(epoch)
