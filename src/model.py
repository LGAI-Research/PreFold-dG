import torch
from torch import nn


class PreFolddG(nn.Module):
    def __init__(self,
                 si_encoder_hidden_dim,
                 si_encoder_dropout,
                 s_encoder_hidden_dim,
                 s_encoder_dropout,
                 z_encoder_dropout,
                 embed_dim,
                 hidden_dim):

        super().__init__()

        self.si = nn.Sequential(
            nn.Linear(384 ** 2, si_encoder_hidden_dim),
            nn.BatchNorm1d(si_encoder_hidden_dim),
            nn.Softplus(),
            nn.Dropout(si_encoder_dropout),
            nn.Linear(si_encoder_hidden_dim, embed_dim),
        )
        self.s = nn.Sequential(
            nn.Linear(384 ** 2, s_encoder_hidden_dim),
            nn.BatchNorm1d(s_encoder_hidden_dim),
            nn.Softplus(),
            nn.Dropout(s_encoder_dropout),
            nn.Linear(s_encoder_hidden_dim, embed_dim),
        )
        self.z = nn.Sequential(
            nn.Dropout(z_encoder_dropout),
            nn.Linear(128, embed_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def embedding(self, x):
        si = self.si(x[:, :384 ** 2])
        s = self.s(x[:, 384 ** 2:2 * (384 ** 2)])
        z = self.z(x[:, 2 * (384 ** 2):2 * (384 ** 2) + 128])

        return torch.stack([si, s, z]).mean(dim=0)

    def forward(self, x):
        emb = self.embedding(x)
        return self.head(emb)
