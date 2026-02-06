import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model.gru import SimpleGRUNet, minGRU


class MLP(nn.Module):
    def __init__(self, in_dim, n_dim):
        super().__init__()
        
        self.rain_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, x, rain):
        rain = rain.unsqueeze(-1)
        return self.rain_mlp(rain).squeeze(-1)


class GRU(nn.Module):
    def __init__(self, in_dim, n_dim):
        super().__init__()
        
        self.D = 24
        self.rain_mlp = nn.Sequential(
            nn.Linear(self.D, 16),
            nn.GELU(),
            minGRU(16, 16),
            nn.GELU(),
            nn.Linear(16, self.D),
        )
        
    def forward(self, x, rain):
        N, S, L = rain.shape
        rain = rain.view(N, S, L // self.D, self.D)
        return self.rain_mlp(rain).view(N, S, L)
    

class CGRU(nn.Module):
    def __init__(self, in_dim, n_dim):
        super().__init__()
        
        self.D = 24
        self.rain_mlp = nn.Sequential(
            nn.Linear(self.D, 8),
            nn.GELU(),
            SimpleGRUNet(8, 8),
            nn.GELU(),
            nn.Linear(8, self.D),
        )
        
    def forward(self, x, rain):
        N, S, L = rain.shape
        rain = rain.contiguous().view(N, S, L // self.D, self.D).contiguous().view(-1, self.D)
        out = self.rain_mlp(rain)
        return out.contiguous().view(N, S, L // self.D, self.D).contiguous().view(N, S, L)


class SpatialBlock(nn.Module):
    """Spatial attention across stations for each time chunk (shared across all chunks)."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out


class MinimalSpatialTemporalMixer(nn.Module):
    def __init__(self, d_model=24, n_heads=4, num_layers=4):
        super().__init__()
        self.D = d_model
        self.in_rain_mlp = nn.Sequential(
            nn.Linear(self.D, 8),
            nn.GELU(),
        )
        self.out = nn.Linear(8, self.D)
        self.spatial = SpatialBlock(8, n_heads)
        self.temporal = nn.Sequential(
            SimpleGRUNet(8, 8),
            nn.GELU(),
        )
        self.num_layers = num_layers

    def forward(self, x, rain):
        # x: unused (keep for compatibility)
        # rain: (N, S, L)
        N, S, L = rain.shape
        assert L % self.D == 0, "L must be divisible by D"
        num_chunks = L // self.D

        rain_chunks = rain.contiguous().view(N, S, num_chunks, self.D)
        rain_chunks = rain_chunks.permute(0, 2, 1, 3).view(N * num_chunks, S, self.D)
        rain_feat = self.in_rain_mlp(rain_chunks)

        # Alternating spatial/temporal mixing, sharing weights
        for _ in range(self.num_layers):
            rain_feat = self.spatial(rain_feat) + rain_feat  # Spatial block
            rain_feat = self.temporal(rain_feat) + rain_feat # Temporal block

        rain_feat = rain_feat.view(N, num_chunks, S, 8).permute(0, 2, 1, 3)
        return self.out(rain_feat).contiguous().view(N, S, num_chunks * self.D)  # (N, S, L)


class CGRU_Diffusion(nn.Module):
    def __init__(self, in_dim, n_dim, num_timesteps=1000):
        super().__init__()
        self.D = 24
        self.num_timesteps = num_timesteps
        
        self.rain_mlp = nn.Sequential(
            nn.Linear(self.D, 8),
            nn.GELU(),
            SimpleGRUNet(8, 8),
            nn.GELU(),
            nn.Linear(8, self.D),
        )
        # Timestep embedding
        self.time_embed = nn.Embedding(num_timesteps, self.D)

    def forward(self, x_noisy, rain, t):
        N, S, L = rain.shape
        assert L % self.D == 0, "L must be divisible by D"
        num_chunks = L // self.D

        rain = rain.contiguous().view(N, S, num_chunks, self.D).contiguous().view(-1, self.D)
        rain_feat = self.rain_mlp(rain)
        rain_feat = rain_feat.contiguous().view(N, S, num_chunks, self.D).contiguous().view(N, S, L)

        # Timestep embedding: shape (N, D) → (N, 1, 1, D) → (N, S, num_chunks, D) → (N, S, L)
        t_emb = self.time_embed(t).unsqueeze(1).unsqueeze(2)
        t_emb = t_emb.expand(N, S, num_chunks, self.D).contiguous().view(N, S, L)

        h = x_noisy + rain_feat + t_emb
        return h


class MLP2(nn.Module):
    def __init__(self, in_dim, n_dim):
        super().__init__()
        
        self.rain_mlp = nn.Sequential(
            nn.Linear(168, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 168),
        )

    def forward(self, x, rain):
        return self.rain_mlp(rain).squeeze(-1)
    


# ---------- Utility: Beta Scheduler ----------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


# ---------- Diffusion Forward Process ----------
def q_sample(x_start, t, noise, alphas_cumprod):
    # x_start: (N, S, L)
    # t: (N,)
    sqrt_alphas_cumprod_t = alphas_cumprod[t].sqrt().view(-1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# ---------- Synthetic Dummy Dataset for Demo ----------
def get_synthetic_dataloader(batch_size=8, S=1, L=48, D=24, num_batches=100):
    # Generate random water levels and rainfall (normalized)
    xs = []
    rains = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, S, L)
        rain = torch.randn(batch_size, S, L)
        xs.append(x)
        rains.append(rain)
    dataset = TensorDataset(torch.cat(xs), torch.cat(rains))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------- Training Loop ----------
def train_diffusion(model, dataloader, num_timesteps=10, epochs=5, device="cpu"):
    betas = linear_beta_schedule(num_timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for x, _, _, _, _, rain, _, _ in pbar:
            x = x.to(device)
            rain = rain.to(device)
            N = x.shape[0]
            t = torch.randint(0, num_timesteps, (N,), device=x.device).long()
            noise = torch.randn_like(x)
            x_noisy = q_sample(x, t, noise, alphas_cumprod)
            pred_noise = model(x_noisy, rain, t)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
    print("Training complete.")

# ---------- Testing: Reverse (Sampling) Process ----------
@torch.no_grad()
def sample_diffusion(model, rain, num_timesteps=10, device="cpu"):
    model.eval()
    betas = linear_beta_schedule(num_timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

    N, S, L = rain.shape
    x = torch.randn(N, S, L, device=device)  # Start from noise
    for t_idx in reversed(range(num_timesteps)):
        t = torch.full((N,), t_idx, device=device, dtype=torch.long)
        pred_noise = model(x, rain, t)
        alpha = alphas[t_idx]
        alpha_cumprod = alphas_cumprod[t_idx]
        alpha_cumprod_prev = alphas_cumprod_prev[t_idx]
        beta = betas[t_idx]

        x0_pred = (x - (1 - alpha_cumprod).sqrt() * pred_noise) / alpha_cumprod.sqrt()
        if t_idx > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = (alpha_cumprod_prev.sqrt() * x0_pred +
             (1 - alpha_cumprod_prev).sqrt() * noise)
    return x0_pred  # Final denoised prediction (water level)