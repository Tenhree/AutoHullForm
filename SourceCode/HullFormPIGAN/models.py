import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig, TrainConfig, PathConfig
import pandas as pd
import os


class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        c = base_channels

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),


            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),


            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),


            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),


            nn.Conv2d(c * 8, c * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )

        self.feature_h = 3
        self.feature_w = 7
        self.feature_c = c * 8
        self.feature_dim = self.feature_c * self.feature_h * self.feature_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return torch.flatten(h, start_dim=1)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, base_channels: int):
        super().__init__()
        c = base_channels

        self.init_h = 3
        self.init_w = 7
        self.init_c = c * 8

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, self.init_c * self.init_h * self.init_w),
            nn.ReLU(inplace=True),
            nn.Linear(self.init_c * self.init_h * self.init_w, self.init_c * self.init_h * self.init_w),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(

            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),


            nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, cond_embed], dim=1)
        h = self.fc(h)
        h = h.view(h.shape[0], self.init_c, self.init_h, self.init_w)
        h = self.deconv(h)
        h = F.interpolate(h, size=(20, 50), mode="bilinear", align_corners=False)
        raw = self.head(h)
        return raw


class TwoStageCVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 32,
        cond_dim: int = 2,
        hidden_dim: int = 256,
        base_channels: int = 32,
        use_structured_output: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.use_structured_output = use_structured_output

        self.encoder = ConvEncoder(in_channels, base_channels)
        self.cond_embed = ConditionEmbedding(cond_dim, hidden_dim)

        enc_in_dim = self.encoder.feature_dim + hidden_dim
        self.fc_mu = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = ConvDecoder(latent_dim, hidden_dim, base_channels)

    def encode(self, x: torch.Tensor, cond: torch.Tensor):
        x_feat = self.encoder(x)
        c_feat = self.cond_embed(cond)
        h = torch.cat([x_feat, c_feat], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def structured_output(self, raw: torch.Tensor):

        if not self.use_structured_output:
            return torch.sigmoid(raw)

        x_raw = raw[:, 0:1, :, :]
        y_raw = raw[:, 1:2, :, :]

        # X 通道：沿最后一维(宽度=50)做递增累计，再归一化到 [0,1]
        dx = F.softplus(x_raw) + 1e-6
        x_cum = torch.cumsum(dx, dim=-1)
        x_out = x_cum / x_cum[..., -1:].clamp(min=1e-6)

        # Y 通道：sigmoid 到 [0,1]
        y_out = torch.sigmoid(y_raw)

        # 第一行和第一列强制为 0
        mask = torch.ones_like(y_out)
        mask[:, :, 0, :] = 0.0
        mask[:, :, :, 0] = 0.0
        y_out = y_out * mask

        return torch.cat([x_out, y_out], dim=1)

    def decode(self, z: torch.Tensor, cond: torch.Tensor):
        c_feat = self.cond_embed(cond)
        raw = self.decoder(z, c_feat)
        out = self.structured_output(raw)
        return out

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, cond)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, cond: torch.Tensor, device: str):
        path_cfg = PathConfig()

        os.makedirs(path_cfg.inference_path, exist_ok=True)
        z_csv_path = os.path.join(path_cfg.inference_path, "z.csv")

        if os.path.exists(z_csv_path):

            z_np = pd.read_csv(z_csv_path, header=None).values
            z = torch.tensor(z_np, dtype=torch.float32, device=device)
            print(f"Loaded z from {z_csv_path}")
        else:

            z = torch.randn(num_samples, self.latent_dim, device=device)


            z_np = z.detach().cpu().numpy()
            pd.DataFrame(z_np).to_csv(
                z_csv_path,
                index=False,
                header=False
            )
        return self.decode(z, cond)