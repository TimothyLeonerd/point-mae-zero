# models/Point_MAE_IAE.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from .build import MODELS
from .Point_MAE import Group, Encoder, TransformerEncoder
from utils.logger import print_log


@MODELS.register_module()
class Point_MAE_IAE(nn.Module):
    """
    Point-MAE style encoder + global latent + implicit SDF head.

    Inputs:
        pts:          (B, N, 3)  encoder input point cloud
        query_points: (B, M, 3)  SDF query locations

    API:
        latent = model.encode_inputs(pts)          # (B, latent_dim)
        sdf = model.decode(query_points, latent)   # (B, M)
        sdf = model(pts, query_points)             # convenience
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        print_log("[Point_MAE_IAE] init", logger="Point_MAE_IAE")

        # Backbone config (same style as PointTransformer)
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        # Grouping + local patch encoder
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # Project encoder dims to transformer dim if needed
        if self.encoder_dims != self.trans_dim:
            self.encoder2trans = nn.Linear(self.encoder_dims, self.trans_dim)
        else:
            self.encoder2trans = None

        # CLS token + positional embeddings (same pattern as PointTransformer)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        # Transformer encoder over tokens
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            drop_path_rate=dpr,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        # ---- Implicit SDF head ----
        sdf_hidden_dim = getattr(config, "sdf_hidden_dim", 256)
        sdf_num_layers = getattr(config, "sdf_num_layers", 4)
        sdf_activation = getattr(config, "sdf_activation", "relu").lower()

        # Global latent = concat(CLS token, max-pooled patches)
        self.latent_dim = 2 * self.trans_dim

        act_cls = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "leaky_relu": nn.LeakyReLU,
        }.get(sdf_activation, nn.ReLU)

        mlp_layers = []
        in_dim = self.latent_dim + 3  # latent + 3D coordinates

        # (sdf_num_layers - 1) hidden layers, then output
        for _ in range(max(sdf_num_layers - 1, 1)):
            mlp_layers.append(nn.Linear(in_dim, sdf_hidden_dim))
            if sdf_activation == "gelu":
                mlp_layers.append(act_cls())
            else:
                mlp_layers.append(act_cls(inplace=True))
            in_dim = sdf_hidden_dim

        mlp_layers.append(nn.Linear(in_dim, 1))  # scalar SDF
        self.sdf_head = nn.Sequential(*mlp_layers)

        # init tokens
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

    # ---------------- Encoder: pts -> latent ----------------
    def encode_inputs(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B, N, 3)
        returns latent: (B, 2 * trans_dim)
        """
        # Group into local neighborhoods
        neighborhood, center = self.group_divider(pts)    # (B, G, S, 3), (B, G, 3)

        # Local patch encoder -> group tokens
        group_tokens = self.encoder(neighborhood)         # (B, G, encoder_dims)

        if self.encoder2trans is not None:
            group_tokens = self.encoder2trans(group_tokens)  # (B, G, trans_dim)

        B, G, C = group_tokens.shape

        # CLS token + position
        cls_tokens = self.cls_token.expand(B, 1, -1)      # (B, 1, C)
        cls_pos = self.cls_pos.expand(B, 1, -1)           # (B, 1, C)

        pos = self.pos_embed(center)                      # (B, G, C)

        x = torch.cat((cls_tokens, group_tokens), dim=1)  # (B, 1+G, C)
        pos = torch.cat((cls_pos, pos), dim=1)            # (B, 1+G, C)

        x = self.blocks(x, pos)
        x = self.norm(x)                                  # (B, 1+G, C)

        cls_feat = x[:, 0]                                # (B, C)
        if x.size(1) > 1:
            patch_max = x[:, 1:].max(dim=1)[0]            # (B, C)
        else:
            patch_max = cls_feat

        latent = torch.cat([cls_feat, patch_max], dim=-1) # (B, 2C)
        return latent

    # ---------------- Decoder: (query_points, latent) -> SDF ----------------
    def decode(self, query_points: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        query_points: (B, M, 3)
        latent:       (B, latent_dim)
        returns sdf:  (B, M)
        """
        B, M, _ = query_points.shape

        if latent.dim() != 2 or latent.size(0) != B:
            raise ValueError(f"latent must be (B, latent_dim), got {latent.shape}")

        latent_expanded = latent.unsqueeze(1).expand(-1, M, -1)  # (B, M, latent_dim)
        x = torch.cat([query_points, latent_expanded], dim=-1)   # (B, M, latent_dim+3)
        x = x.reshape(B * M, -1)                                 # (B*M, latent_dim+3)

        sdf = self.sdf_head(x)                                   # (B*M, 1)
        #sdf = torch.tanh(sdf)
        sdf = sdf.view(B, M)                                     # (B, M)
        return sdf

    # ---------------- Convenience forward ----------------
    def forward(
        self,
        pts: torch.Tensor,
        query_points: torch.Tensor = None,
        return_latent: bool = False,
    ):
        """
        If query_points is provided:
            -> returns SDF predictions (and optionally latent).
        If query_points is None and return_latent=True:
            -> returns only latent.
        """
        latent = self.encode_inputs(pts)

        if query_points is None:
            if return_latent:
                return latent
            raise ValueError(
                "query_points is None. Provide query_points or call with "
                "return_latent=True to get only the latent."
            )

        sdf = self.decode(query_points, latent)

        if return_latent:
            return sdf, latent
        return sdf
