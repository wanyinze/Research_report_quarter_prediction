import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, z_dim, i_dim):
        super(MLP, self).__init__()

        self.z_dim = z_dim
        self.i_dim = i_dim
        zi_features_num = 16  # Z和I进行合并时的特征个数

        # input_I => feature_I
        self.extract_i_features = nn.Sequential(
            nn.Linear(i_dim, 64),
            nn.ReLU(),
            nn.Linear(64, zi_features_num),
            nn.ReLU()
        )
        # input_Z => feature_Z
        self.extract_z_features = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ReLU(),
            nn.Linear(32, zi_features_num),
            nn.ReLU()
        )
        # feature_Z, feature_I => revenue variable
        self.predict_revenue = nn.Sequential(
            nn.Linear(2*zi_features_num, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, inputs):
        Z = inputs[:, :self.z_dim]
        I = inputs[:, self.z_dim:]
        z_features = self.extract_z_features(Z)
        i_features = self.extract_i_features(I)
        zi = torch.cat((z_features, i_features), 1)
        out = self.predict_revenue(zi)
        return out


def mlp_loss_function(prediction, real):
    return F.mse_loss(prediction, real, reduction='sum')
