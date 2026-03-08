"""
vit.py
------
Vision Transformer implementations for acoustic leak detection.

Two variants are provided:

1. ``VisionTransformer``   — ViT trained from scratch.
2. ``PretrainedViT``       — ViT-Base/16 (ImageNet weights via timm) with a
                             custom projection head for scalogram / spectrogram input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Building blocks (shared)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split a 2-D image into non-overlapping patches and linearly embed them.

    Parameters
    ----------
    img_size : tuple of (H, W)
    patch_size : tuple of (ph, pw)
    in_channels : int
    embed_dim : int
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ViT from scratch
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) trained from scratch.

    Parameters
    ----------
    img_size : tuple of (H, W)
        Spatial size of the input time-frequency image.
    patch_size : tuple of (ph, pw)
        Patch resolution.
    in_channels : int
        1 for grayscale scalograms / spectrograms, 3 for RGB.
    num_classes : int
        Number of output classes.
    is_binary : bool
        If ``True``, outputs a single logit (BCEWithLogitsLoss).
    embed_dim : int
        Transformer hidden dimension.
    depth : int
        Number of Transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        MLP expansion ratio.
    drop_rate : float
        Dropout applied to embeddings and MLP layers.
    attn_drop_rate : float
        Dropout applied to attention weights.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (50, 512),
        patch_size: Tuple[int, int] = (5, 8),
        in_channels: int = 1,
        num_classes: int = 5,
        is_binary: bool = False,
        embed_dim: int = 768,
        depth: int = 10,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.15,
        attn_drop_rate: float = 0.05,
    ):
        super().__init__()
        self.is_binary = is_binary

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, attn_drop_rate, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        out_dim = 1 if is_binary else num_classes
        self.head = nn.Linear(embed_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x[:, 0])  # CLS token


# ---------------------------------------------------------------------------
# Pretrained ViT (timm / ImageNet weights)
# ---------------------------------------------------------------------------

class PretrainedViT(nn.Module):
    """
    ViT-Base/16 fine-tuned on time-frequency images.

    Uses ``timm``'s ``vit_base_patch16_224`` backbone with ImageNet weights.
    A projection head adapts the backbone for variable-size inputs and
    a custom number of output classes.

    Parameters
    ----------
    input_shape : tuple of (H, W)
        Spatial dimensions of the time-frequency image **before** resizing.
    num_classes : int
    is_binary : bool
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (50, 2048),
        num_classes: int = 5,
        is_binary: bool = False,
    ):
        super().__init__()
        import timm

        self.is_binary = is_binary
        # timm.create_model works in all timm versions >= 0.4
        # num_classes=0 removes the head, keeping the backbone features only
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )

        hidden_dim = self.backbone.embed_dim  # 768 for ViT-B/16
        out_dim = 1 if is_binary else num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize to 224×224 (required by ViT-B/16)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Repeat grayscale channel to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.backbone(x)
        return self.head(features)


# ---------------------------------------------------------------------------
# Image size registry  (one place to change if you tweak your transforms)
# ---------------------------------------------------------------------------

# Maps transform name → (img_size, patch_size)
# img_size   = (H, W) of the time-frequency image fed to the ViT
# patch_size = (ph, pw) — must divide img_size exactly
TRANSFORM_CONFIGS = {
    "cwt":  {"img_size": (50, 2048),  "patch_size": (5, 16)},
    "stft": {"img_size": (272, 112),  "patch_size": (8, 8)},
}

# Maps task name → number of output neurons
TASK_CONFIGS = {
    "binary":     {"num_classes": 1,  "is_binary": True},
    "multiclass": {"num_classes": 5,  "is_binary": False},
}


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_model(
    model_type: str = "scratch",
    transform: str = "cwt",
    task: str = "binary",
    device: torch.device | None = None,
) -> nn.Module:
    """
    Instantiate the correct ViT variant for a given experiment configuration.

    Parameters
    ----------
    model_type : ``'scratch'`` | ``'pretrained'``
        ``'scratch'``    — ViT trained from scratch with custom architecture.
        ``'pretrained'`` — ViT-Base/16 fine-tuned from ImageNet weights (timm).
    transform : ``'cwt'`` | ``'stft'``
        Determines the spatial size of the input image and patch grid:
          - ``'cwt'``  → image (50 × 2048), patches (5 × 16)
          - ``'stft'`` → image (272 × 112),  patches (8 × 8)
    task : ``'binary'`` | ``'multiclass'``
        ``'binary'``     — 1 output neuron, BCEWithLogitsLoss.
        ``'multiclass'`` — 5 output neurons (no-leak + 4 pressure levels),
                           CrossEntropyLoss.

    Notes
    -----
    ``dataset`` (branched / looped) is intentionally NOT a parameter here.
    The topology only affects which data file you load — not the model
    architecture. Pass the correct data path to ``make_dataloaders()`` instead.

    Returns
    -------
    nn.Module moved to ``device``.

    Examples
    --------
    >>> model = build_model("scratch", "cwt", "binary")
    >>> model = build_model("pretrained", "stft", "multiclass")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Validate inputs ───────────────────────────────────────────────────
    if transform not in TRANSFORM_CONFIGS:
        raise ValueError(f"transform must be one of {list(TRANSFORM_CONFIGS)}. Got '{transform}'.")
    if task not in TASK_CONFIGS:
        raise ValueError(f"task must be one of {list(TASK_CONFIGS)}. Got '{task}'.")
    if model_type not in ("scratch", "pretrained"):
        raise ValueError(f"model_type must be 'scratch' or 'pretrained'. Got '{model_type}'.")

    # ── Resolve configs ───────────────────────────────────────────────────
    img_size   = TRANSFORM_CONFIGS[transform]["img_size"]
    patch_size = TRANSFORM_CONFIGS[transform]["patch_size"]
    num_classes = TASK_CONFIGS[task]["num_classes"]
    is_binary   = TASK_CONFIGS[task]["is_binary"]

    # ── Build model ───────────────────────────────────────────────────────
    if model_type == "pretrained":
        model = PretrainedViT(
            input_shape=img_size,
            num_classes=num_classes,
            is_binary=is_binary,
        )
    else:
        model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            num_classes=num_classes,
            is_binary=is_binary,
        )

    print(
        f"[build_model] {model_type.upper()} ViT | "
        f"transform={transform} → img {img_size} patches {patch_size} | "
        f"task={task} → {num_classes} output(s)"
    )
    return model.to(device)
