"""
BC-VMamba-CUBS — Config A1 (architecture-only).

Komponen (sesuai PDF R1):
  - VSSBlock        : bidirectional Mamba SSM (O(N)) untuk long-range boundary context
  - TiFusionModule  : local depthwise-separable conv (speckle) + VSSBlock (global) + channel gate
  - EdgeAttention   : Sobel-guided attention untuk fokus tepi LIB/MAB
  - MWFFDDecoder    : boundary-aware decoder dengan learnable weight skip/upsample + EdgeAttention
  - BCVMambaCUBS_A1 : U-shape encoder-decoder, head = Conv2d(c1, num_classes, 1) tanpa sigmoid
                       sehingga compatible dengan SimpleLoss(softmax)+segmentation_metrics(argmax) baseline.

Catatan implementasi:
  - TiFusion HANYA dipakai di level dalam (64x64 dan 32x32) + bottleneck (16x16) untuk menjaga
    memori T4 16GB pada batch=8. Top-2 level (256, 128) pakai ConvBlock saja: jumlah token Mamba
    di level 256x256 = 65536 dengan dim=32 akan memboroskan VRAM tanpa kontribusi besar
    pada konteks long-range. Ini konsisten dengan praktik VM-UNet & M3-UNet (Mamba di deeper layers).
  - VSSBlock pakai 2-way scan (forward + flipped) — implementasi paling umum di VSS literature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as _e:
    raise ImportError("mamba_ssm not installed. Run: pip install mamba-ssm causal-conv1d") from _e


# ----- Building blocks --------------------------------------------------------
class ConvBlock(nn.Module):
    """Plain double-conv block used di top-2 encoder levels (top-resolution)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
    def forward(self, x): return self.block(x)


class VSSBlock(nn.Module):
    """Bidirectional Mamba SSM untuk 2D feature map (O(N) long-range modeling).

    forward + flipped scan (2-way) sepanjang flatten(H,W) sequence.
    Output di-proyeksikan kembali ke dim semula + residual.
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_f = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_b = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(2 * dim, dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)         # (B, HW, C)
        x_norm = self.norm(x_flat)
        y_f = self.mamba_f(x_norm)
        y_b = self.mamba_b(x_norm.flip(1)).flip(1)
        y   = self.proj(torch.cat([y_f, y_b], dim=-1)) + self.skip_scale * x_flat
        return y.transpose(1, 2).reshape(B, C, H, W)


class TiFusionModule(nn.Module):
    """Local depthwise-separable CNN (speckle) + Global VSSBlock + channel attention gate.

    PDF: f_fused = gate(concat(F_local, F_global)). Projected back ke ch dim.
    """
    def __init__(self, ch: int, d_state: int = 16, r: int = 4):
        super().__init__()
        # Local branch: depthwise separable conv (speckle texture)
        self.local = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),   # depthwise
            nn.Conv2d(ch, ch, 1, bias=False),                          # pointwise
            nn.BatchNorm2d(ch), nn.GELU(),
        )
        # Global branch: VSS Mamba
        self.global_mamba = VSSBlock(ch, d_state=d_state)
        # Channel attention gate over [F_local; F_global]
        hid = max(1, (ch * 2) // r)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch * 2, hid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid, ch * 2, 1), nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(ch * 2, ch, 1)

    def forward(self, x):
        fl  = self.local(x)
        fg  = self.global_mamba(x)
        cat = torch.cat([fl, fg], dim=1)
        return self.proj(cat * self.gate(cat))


class EdgeAttention(nn.Module):
    """Sobel-guided attention untuk fokus tepi LIB/MAB."""
    def __init__(self, ch: int):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sx.transpose(-1, -2).contiguous())
        self.attn = nn.Sequential(nn.Conv2d(ch + 1, ch, 1), nn.Sigmoid())

    def forward(self, x):
        gray = x.mean(1, keepdim=True)
        ex   = F.conv2d(gray, self.sobel_x, padding=1).abs()
        ey   = F.conv2d(gray, self.sobel_y, padding=1).abs()
        edge = ex + ey
        return x * self.attn(torch.cat([x, edge], dim=1))


class MWFFDDecoder(nn.Module):
    """Mixed-Weight Fine Feature Decoder.
    Learnable scalar weight per stream (skip vs upsampled), refine conv, lalu EdgeAttention.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.w_sk = nn.Parameter(torch.ones(1))
        self.w_up = nn.Parameter(torch.ones(1))
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.edge_attn = EdgeAttention(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([self.w_up * x, self.w_sk * skip], dim=1)
        return self.edge_attn(self.refine(fused))


# ----- Full model -------------------------------------------------------------
class BCVMambaCUBS_A1(nn.Module):
    """A1 config: arsitektur saja, output logits 2-channel compatible dengan SimpleLoss baseline.

    Encoder channel layout: [base, base*2, base*4, base*8] + bottleneck base*16.
    Default base=32 -> [32, 64, 128, 256, 512]. Total ~9-15M params (cek di sanity).
    """
    def __init__(self, in_ch: int = 1, num_classes: int = 2, base_ch: int = 32, d_state: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        cb             = base_ch * 16

        # ---------------- Encoder ----------------
        self.stem  = nn.Conv2d(in_ch, c1, 3, padding=1)
        # Level 1 & 2: ConvBlock saja (top resolution, Mamba terlalu boros memori)
        self.enc1  = ConvBlock(c1, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c1, c2, 1, bias=False))
        self.enc2  = ConvBlock(c2, c2)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c2, c3, 1, bias=False))
        # Level 3 & 4: TiFusion (local + global Mamba)
        self.enc3  = TiFusionModule(c3, d_state=d_state)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c3, c4, 1, bias=False))
        self.enc4  = TiFusionModule(c4, d_state=d_state)
        self.down4 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c4, cb, 1, bias=False))

        # ---------------- Bottleneck ----------------
        self.bottleneck = TiFusionModule(cb, d_state=d_state)

        # ---------------- Decoder ----------------
        self.dec4 = MWFFDDecoder(cb, c4, c4)
        self.dec3 = MWFFDDecoder(c4, c3, c3)
        self.dec2 = MWFFDDecoder(c3, c2, c2)
        self.dec1 = MWFFDDecoder(c2, c1, c1)

        # ---------------- Segmentation head (logits, no sigmoid) ----------------
        self.seg_head = nn.Conv2d(c1, num_classes, 1)

        print(f"  [BCVMambaCUBS_A1] base_ch={base_ch}, d_state={d_state}, num_classes={num_classes}")

    def forward(self, x):
        x  = self.stem(x)                       # (B, c1,  H,    W)
        e1 = self.enc1(x)                       # (B, c1,  H,    W)
        e2 = self.enc2(self.down1(e1))          # (B, c2,  H/2,  W/2)
        e3 = self.enc3(self.down2(e2))          # (B, c3,  H/4,  W/4)
        e4 = self.enc4(self.down3(e3))          # (B, c4,  H/8,  W/8)
        b  = self.bottleneck(self.down4(e4))    # (B, cb,  H/16, W/16)
        d4 = self.dec4(b,  e4)                  # (B, c4,  H/8,  W/8)
        d3 = self.dec3(d4, e3)                  # (B, c3,  H/4,  W/4)
        d2 = self.dec2(d3, e2)                  # (B, c2,  H/2,  W/2)
        d1 = self.dec1(d2, e1)                  # (B, c1,  H,    W)
        return self.seg_head(d1)                # (B, num_classes, H, W)  -- logits
