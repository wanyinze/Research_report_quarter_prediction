import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, x_dim, i_dim, z_dim):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.i_dim = i_dim
        self.z_dim = z_dim
        xi_features_num = 32  # X和I进行合并时的特征个数
        zi_features_num = 16  # Z和I进行合并时的特征个数

        # encoder部分
        # input_X => feature_X
        self.extract_x_features = nn.Sequential(
            nn.Linear(x_dim, 64),
            nn.ReLU(),
            nn.Linear(64, xi_features_num),
            nn.ReLU()
        )
        # input_I => feature_I
        self.extract_i_features_1 = nn.Sequential(
            nn.Linear(i_dim, 64),
            nn.ReLU(),
            nn.Linear(64, xi_features_num),
            nn.ReLU()
        )
        # feature_X, feature_I => mu and variance
        self.get_mu_variance = nn.Sequential(
            nn.Linear(2*xi_features_num, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 2*z_dim)  # 前z_dim为mu，后z_dim为log_var
        )

        # decoder部分
        # sampled Z => feature_Z
        self.extract_z_features = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ReLU(),
            nn.Linear(32, zi_features_num),
            nn.ReLU()
        )
        # input_I => feature_I
        self.extract_i_features_2 = nn.Sequential(
            nn.Linear(i_dim, 64),
            nn.ReLU(),
            nn.Linear(64, zi_features_num),
            nn.ReLU()
        )
        # feature_Z, feature_I => X
        self.reconstruct_x = nn.Sequential(
            nn.Linear(2*zi_features_num, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, x_dim)
        )

    def encode(self, X, I):
        """提取X与I的特征再进行合并，向前传播生成mu与logvar"""
        x_features = self.extract_x_features(X)
        i_features = self.extract_i_features_1(I)
        xi = torch.cat((x_features, i_features), 1)
        mu, logvar = self.get_mu_variance(xi).chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return mu + eps*std

    def decode(self, Z, I):
        """提取Z与I的特征再进行合并，向前传播重构X"""
        z_features = self.extract_z_features(Z)
        i_features = self.extract_i_features_2(I)
        zi = torch.cat((z_features, i_features), 1)
        recon_x = self.reconstruct_x(zi)
        return recon_x

    def forward(self, inputs):
        X = inputs[:, :self.x_dim]
        I = inputs[:, self.x_dim:]
        mu, logvar = self.encode(X, I)
        Z = self.reparameterize(mu, logvar)
        recon_x = self.decode(Z, I)
        return recon_x, mu, logvar

    def sample_z(self, inputs):
        """通过输入X与I来取样Z"""
        X = inputs[:, :self.x_dim]
        I = inputs[:, self.x_dim:]
        mu, logvar = self.encode(X, I)
        Z = self.reparameterize(mu, logvar)
        return Z


def vae_loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence
