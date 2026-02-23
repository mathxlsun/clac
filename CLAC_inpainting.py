# -*- coding: utf-8 -*-
import os
from datetime import datetime
import numpy as np
import scipy.io as scio
import torch
from torch import nn, optim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

def create_coordinate_grid(spatial_size, batch_size=1, normalize=True):
    height, width = spatial_size
    if normalize:
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, height),
            torch.linspace(0, 1, width),
            indexing="ij",
        )
    else:
        y_coords, x_coords = torch.meshgrid(
            torch.arange(0, height),
            torch.arange(0, width),
            indexing="ij",
        )
    coords = torch.stack([x_coords, y_coords], dim=-1)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return coords.type(dtype).to(device)

def calculate_metrics(original, denoised):
    original_np = original.cpu().detach().numpy() if torch.is_tensor(original) else original
    denoised_np = denoised.cpu().detach().numpy() if torch.is_tensor(denoised) else denoised
    data_range = original_np.max() - original_np.min()
    psnr_value = skimage_psnr(original_np, denoised_np, data_range=data_range)
    ssim_value = skimage_ssim(
        original_np, denoised_np,
        data_range=data_range,
        win_size=11,
        multichannel=True,
        gaussian_weights=True,
        sigma=1.5
    )
    return psnr_value, ssim_value

class NoFC_3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(NoFC_3, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, out_channel, bias=False),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.layer(x)

class AdaptiveFourierEncoding(nn.Module):
    def __init__(self, coord_dim, num_harmonics=16, learnable=True):
        super(AdaptiveFourierEncoding, self).__init__()
        self.num_harmonics = num_harmonics
        self.learnable = learnable
        self.coord_dim = coord_dim
        freqs = torch.logspace(0, 3, num_harmonics, dtype=dtype)
        if learnable:
            self.frequencies = nn.Parameter(freqs)
        else:
            self.register_buffer("frequencies", freqs)
        self.encode_weight = nn.Parameter(torch.ones(coord_dim, num_harmonics * 2, dtype=dtype))
    def forward(self, coords):
        B, H, W, D = coords.shape
        K = self.num_harmonics
        freq = self.frequencies.view(1, 1, 1, 1, K).to(coords.device)
        c = coords.unsqueeze(-1)
        angles = 2 * torch.pi * c * freq
        enc = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        w = self.encode_weight.view(1, 1, 1, D, 2 * K).to(coords.device)
        enc = enc * w
        return enc.reshape(B, H, W, D * 2 * K)

class EnhancedCoordinateToTensor(nn.Module):
    def __init__(self, coord_dim, hidden_dims, output_channels, spatial_size, use_positional_encoding=True):
        super(EnhancedCoordinateToTensor, self).__init__()
        self.spatial_size = spatial_size
        self.output_channels = output_channels
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoding = AdaptiveFourierEncoding(coord_dim=coord_dim, num_harmonics=12, learnable=True)
            coord_dim = coord_dim * self.pos_encoding.num_harmonics * 2
        layers = []
        input_dim = coord_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_channels))
        self.mlp = nn.Sequential(*layers)
    def forward(self, coords):
        if self.use_positional_encoding:
            coords = self.pos_encoding(coords)
        coords_flat = coords.reshape(-1, coords.shape[-1])
        features_flat = self.mlp(coords_flat)
        features = features_flat.reshape(coords.shape[0], coords.shape[1], coords.shape[2], self.output_channels)
        return features

class EnhancedMultiScaleDynamicExpertConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5), num_experts=4):
        super(EnhancedMultiScaleDynamicExpertConv, self).__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.experts = nn.ModuleList()
        for k_size in kernel_sizes:
            padding = k_size // 2
            self.experts.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=k_size,
                        padding=padding,
                        groups=in_channels,
                        bias=False,
                        padding_mode="circular",
                    )
                )
            )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_experts, kernel_size=1, bias=False),
            nn.Softmax(dim=1),
        )
    def forward(self, x):
        attn_weights = self.attention(x)
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out.unsqueeze(1))
        expert_outputs = torch.cat(expert_outputs, dim=1)
        attn_weights = attn_weights.unsqueeze(2)
        dynamic_out = (expert_outputs * attn_weights).sum(dim=1)
        return dynamic_out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out

class EnhancedCoordinateCLAC(nn.Module):
    def __init__(self, n_4, n_3, coord_dim=2, hidden_dims=(128, 256, 512), spatial_size=(200, 200), num_experts=4):
        super(EnhancedCoordinateCLAC, self).__init__()
        self.n_4 = n_4
        self.n_3 = n_3
        self.spatial_size = spatial_size
        self.coord_to_tensor = EnhancedCoordinateToTensor(
            coord_dim=coord_dim,
            hidden_dims=list(hidden_dims),
            output_channels=n_4,
            spatial_size=spatial_size,
            use_positional_encoding=True,
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(n_4) for _ in range(2)])
        self.g_part1 = nn.Sequential(NoFC_3(n_4, n_4))
        self.dynamic_conv1 = EnhancedMultiScaleDynamicExpertConv(
            in_channels=n_4,
            out_channels=n_4,
            kernel_sizes=(3, 5),
            num_experts=num_experts,
        )
        self.dynamic_conv2 = EnhancedMultiScaleDynamicExpertConv(
            in_channels=n_4,
            out_channels=n_4,
            kernel_sizes=(3, 5),
            num_experts=num_experts,
        )
        self.g_part2 = nn.Sequential(NoFC_3(n_4, n_3))
    def forward(self, coords):
        coord_features = self.coord_to_tensor(coords)
        x = coord_features.permute(0, 3, 1, 2).contiguous()
        x_res = self.residual_blocks(x)
        x = self.dynamic_conv1(x + x_res)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.g_part1(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dynamic_conv2(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.g_part2(x)
        return coord_features, x

def main():
    original_data_path = "Balloons.mat"
    missing_data_path = "Balloons1.mat"

    max_iter = 6000
    num_experts = 2
    print_every = 100
    F_norm = nn.MSELoss()

    mat_original = scio.loadmat(original_data_path)
    X_original_np = mat_original["Ohsi"].astype(np.float32)
    mat_missing = scio.loadmat(missing_data_path)
    X_missing_np = mat_missing["Nhsi"].astype(np.float32)

    mask_np = np.ones_like(X_missing_np, dtype=np.float32)
    mask_np[X_missing_np == 0] = 0.0

    X_original = torch.from_numpy(X_original_np).to(device, dtype)
    X_missing = torch.from_numpy(X_missing_np).to(device, dtype)
    mask = torch.from_numpy(mask_np).to(device, dtype)

    n_3 = X_original_np.shape[2]
    n_4 = n_3 * 2
    spatial_size = (X_original_np.shape[0], X_original_np.shape[1])
    coords = create_coordinate_grid(spatial_size, batch_size=1, normalize=True)

    X_original_b = X_original.unsqueeze(0)
    X_missing_b = X_missing.unsqueeze(0)
    mask_b = mask.unsqueeze(0)

    model = EnhancedCoordinateOnlyCLAC(
        n_4=n_4,
        n_3=n_3,
        coord_dim=2,
        hidden_dims=(256, 256, 256),
        spatial_size=spatial_size,
        num_experts=num_experts,
    ).to(device)

    try:
        model = torch.compile(model)
    except Exception as e:
        pass

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=1e-8,
        fused=(device.type == "cuda"),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
    use_amp = (device.type == "cuda")
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)


    final_X_filled = None
    for it in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            _, X_Out = model(coords)
            loss = F_norm(X_Out * mask_b, X_missing_b * mask_b)
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()

        if (it % print_every == 0) or (it == max_iter - 1):
            lr = scheduler.get_last_lr()[0]
            print(f"Iter: {it:5d}/{max_iter} | Loss: {loss.item():.6f} | LR: {lr:.6f}")

        if it == max_iter - 1:
            with torch.no_grad():
                final_X_filled = X_Out * (1 - mask_b) + X_missing_b * mask_b

    final_recovered = final_X_filled.detach().cpu().numpy()[0]
    psnr_value, ssim_value = calculate_metrics(X_original_np, final_recovered)
    print(f"PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f}")

if __name__ == "__main__":
    main()