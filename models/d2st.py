import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, einsum
import clip
import math
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = self.norm(x)
        x = rearrange(x, 'b t h w c -> b c t h w')
        return x

class ViT_DeformAttention(nn.Module):
    def __init__(self, cfg, dim, heads, groups, kernel_size, stride, padding):
        super().__init__()
        self.args = cfg
        self.dim = dim
        self.heads = heads
        self.head_channels = dim // heads
        self.scale = self.head_channels ** -0.5
        self.groups = groups
        self.group_channels = self.dim // self.groups
        self.group_heads = self.heads // self.groups
        self.factor = 2.0

        self.conv_offset = nn.Sequential(
            nn.Conv3d(in_channels=self.group_channels, out_channels=self.group_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.group_channels),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv3d(in_channels=self.group_channels, out_channels=3, kernel_size=(1, 1, 1), bias=False)
        )

        self.proj_q = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_k = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_v = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_out = nn.Linear(in_features=self.dim, out_features=self.dim)

    @torch.no_grad()
    def _get_ref_points(self, T, H, W, B, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T - 0.5, T, dtype=dtype, device=device),
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)
        ref[..., 0].div_(T).mul_(2).sub_(1)
        ref[..., 1].div_(H).mul_(2).sub_(1)
        ref[..., 2].div_(W).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.groups, -1, -1, -1, -1)  # B * g T H W 3

        return ref

    def forward(self, x):
        # hw+1 bt c
        n, BT, C = x.shape
        T = self.args.DATA.NUM_INPUT_FRAMES
        B = BT // T
        H = round(math.sqrt(n - 1))
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = rearrange(q[1:, :, :], '(h w) (b t) c -> b c t h w', h=H, t=T)
        q_off = rearrange(q_off, 'b (g c) t h w -> (b g) c t h w', g=self.groups, c=self.group_channels)
        offset = self.conv_offset(q_off)  # B * g 3 Tp Hp Wp
        Tp, Hp, Wp = offset.size(2), offset.size(3), offset.size(4)
        n_sample = Tp * Hp * Wp

        offset_range = torch.tensor([min(1.0, self.factor / Tp), min(1.0, self.factor / Hp), min(1.0, self.factor / Wp)], device=device).reshape(1, 3, 1, 1, 1)
        offset = offset.tanh().mul(offset_range)
        offset = rearrange(offset, 'b p t h w -> b t h w p')
        reference = self._get_ref_points(Tp, Hp, Wp, B, dtype, device)
        pos = offset + reference

        x_sampled = rearrange(x[1:, :, :], '(h w) (b t) c -> b c t h w', h=H, t=T)
        x_sampled = rearrange(x_sampled, 'b (g c) t h w -> (b g) c t h w', g=self.groups)
        x_sampled = F.grid_sample(input=x_sampled, grid=pos[..., (2, 1, 0)], mode='bilinear', align_corners=True)  # B * g, Cg, Tp, Hp, Wp
        x_sampled = rearrange(x_sampled, '(b g) c t h w -> b (g c) t h w', g=self.groups)
        x_sampled = rearrange(x_sampled, 'b c t h w -> b (t h w) c')

        q = rearrange(q, 'n (b t) c -> b c (t n)', b=B)
        q = rearrange(q, 'b (h c) n -> (b h) c n', h=self.heads)

        k = self.proj_k(x_sampled)
        k = rearrange(k, 'b n (h c) -> (b h) c n', h=self.heads)

        v = self.proj_v(x_sampled)
        v = rearrange(v, 'b n (h c) -> (b h) c n', h=self.heads)

        attn = einsum(q, k, 'b c m, b c n -> b m n')
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=-1)

        out = einsum(attn, v, 'b m n, b c n -> b c m')
        out = rearrange(out, '(b h) c n -> b (h c) n', h=self.heads)
        out = rearrange(out, 'b c (t n) -> n (b t) c', t=T)
        out = self.proj_out(out)

        return out

class ViT_D2ST_Adapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.in_channels = cfg.ADAPTER.WIDTH
        self.out_channels = cfg.ADAPTER.WIDTH
        self.adapter_channels = int(cfg.ADAPTER.WIDTH * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Linear(in_features=self.in_channels, out_features=self.adapter_channels)
        self.gelu1 = nn.GELU()

        self.pos_embed = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=self.adapter_channels)
        self.s_ln = nn.LayerNorm(normalized_shape=self.adapter_channels)
        self.s_attn = ViT_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=4, groups=4, kernel_size=(4, 5, 5), stride=(4, 3, 3), padding=(0, 0, 0))
        self.t_ln = nn.LayerNorm(normalized_shape=self.adapter_channels)
        self.t_attn = ViT_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=4, groups=4, kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        self.gelu = nn.GELU()

        self.up = nn.Linear(in_features=self.adapter_channels, out_features=self.out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        # hw+1 bt c
        n, bt, c = x.shape
        H = round(math.sqrt(n - 1))
        x_in = x

        x = self.down(x)
        x = self.gelu1(x)

        cls = x[0, :, :].unsqueeze(0)
        x = x[1:, :, :]

        x = rearrange(x, '(h w) (b t) c -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES, h=H)
        x = x + self.pos_embed(x)
        x = rearrange(x, 'b c t h w -> (h w) (b t) c')

        x = torch.cat([cls, x], dim=0)

        # Spatial Deformable Attention
        xs = x + self.s_attn(self.s_ln(x))

        # Temporal Deformable Attention
        xt = x + self.t_attn(self.t_ln(x))

        x = (xs + xt) / 2
        x = self.gelu(x)

        x = self.up(x)
        x = self.gelu2(x)

        x += x_in
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResNet_DeformAttention(nn.Module):
    """Exact deformable attention for ResNet backbone - matches ViT implementation"""
    def __init__(self, cfg, dim, heads, groups, kernel_size, stride, padding):
        super().__init__()
        self.args = cfg
        self.dim = dim
        self.heads = heads
        self.head_channels = dim // heads
        self.scale = self.head_channels ** -0.5
        self.groups = groups
        self.group_channels = self.dim // self.groups
        self.group_heads = self.heads // self.groups
        self.factor = 2.0

        self.conv_offset = nn.Sequential(
            nn.Conv3d(in_channels=self.group_channels, out_channels=self.group_channels, 
                     kernel_size=kernel_size, stride=stride, padding=padding, groups=self.group_channels),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv3d(in_channels=self.group_channels, out_channels=3, kernel_size=(1, 1, 1), bias=False)
        )

        self.proj_q = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=1)
        self.proj_k = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=1)  
        self.proj_v = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=1)

    @torch.no_grad()
    def _get_ref_points(self, T, H, W, B, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T - 0.5, T, dtype=dtype, device=device),
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)
        ref[..., 0].div_(T).mul_(2).sub_(1)
        ref[..., 1].div_(H).mul_(2).sub_(1)
        ref[..., 2].div_(W).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.groups, -1, -1, -1, -1)  # B * g T H W 3
        return ref

    def forward(self, x):
        # Input: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        dtype, device = x.dtype, x.device

        # Generate queries, keys, values
        q = self.proj_q(x)  # (B, C, T, H, W)
        
        # Prepare for offset computation
        q_off = rearrange(q, 'b (g c) t h w -> (b g) c t h w', g=self.groups, c=self.group_channels)
        offset = self.conv_offset(q_off)  # (B*g, 3, Tp, Hp, Wp)
        Tp, Hp, Wp = offset.size(2), offset.size(3), offset.size(4)
        
        # Compute offset range and normalize
        offset_range = torch.tensor([min(1.0, self.factor / Tp), min(1.0, self.factor / Hp), min(1.0, self.factor / Wp)], 
                                   device=device).reshape(1, 3, 1, 1, 1)
        offset = offset.tanh().mul(offset_range)
        offset = rearrange(offset, 'bg p t h w -> bg t h w p')
        
        # Get reference points
        reference = self._get_ref_points(Tp, Hp, Wp, B, dtype, device)
        pos = offset + reference

        # Sample from input using deformable positions
        x_sampled = rearrange(x, 'b (g c) t h w -> (b g) c t h w', g=self.groups)
        x_sampled = F.grid_sample(input=x_sampled, grid=pos[..., (2, 1, 0)], 
                                mode='bilinear', align_corners=True)  # (B*g, Cg, Tp, Hp, Wp)
        x_sampled = rearrange(x_sampled, '(b g) c t h w -> b (g c) t h w', g=self.groups)

        # Reshape for attention computation
        q = rearrange(q, 'b (h c) t h_dim w -> (b h) c (t h_dim w)', h=self.heads)
        
        k = self.proj_k(x_sampled)
        k = rearrange(k, 'b (h c) t h_dim w -> (b h) c (t h_dim w)', h=self.heads)
        
        v = self.proj_v(x_sampled)
        v = rearrange(v, 'b (h c) t h_dim w -> (b h) c (t h_dim w)', h=self.heads)

        # Compute attention
        attn = einsum(q, k, 'b c m, b c n -> b m n')
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = einsum(attn, v, 'b m n, b c n -> b c m')
        out = rearrange(out, '(b h) c (t h_dim w) -> b (h c) t h_dim w', 
                       h=self.heads, t=T, h_dim=H, w=W)
        out = self.proj_out(out)

        return out

class ResNet_D2ST_Adapter(nn.Module):
    """D2ST Adapter for ResNet backbone - exact implementation matching ViT version"""
    def __init__(self, cfg, dim, num_frames):
        super().__init__()
        self.args = cfg
        self.num_frames = num_frames
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)
        
        # Dimension reduction
        self.down = nn.Conv3d(self.in_channels, self.adapter_channels, kernel_size=1)
        self.gelu1 = nn.GELU()
        
        # Positional embedding
        self.pos_embed = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, 
                                  kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=self.adapter_channels)
        
        # Layer norms for dual pathways - use 3D layer norms
        self.s_ln = LayerNormProxy(self.adapter_channels)
        self.t_ln = LayerNormProxy(self.adapter_channels)
        
        # Spatial and temporal deformable attention pathways - EXACT configuration from original
        # Different head/group configurations based on feature dimension (matching original D2ST)
        if dim == self.args.ADAPTER.WIDTH // 8:
            self.s_attn = ResNet_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=1, groups=1, 
                                               kernel_size=(4, 7, 7), stride=(4, 7, 7), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=1, groups=1, 
                                               kernel_size=(1, 14, 14), stride=(1, 14, 14), padding=(0, 0, 0))
        elif dim == self.args.ADAPTER.WIDTH // 4:
            self.s_attn = ResNet_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=2, groups=2, 
                                               kernel_size=(4, 7, 7), stride=(4, 7, 7), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=2, groups=2, 
                                               kernel_size=(1, 14, 14), stride=(1, 14, 14), padding=(0, 0, 0))
        elif dim == self.args.ADAPTER.WIDTH // 2:
            self.s_attn = ResNet_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=4, groups=4, 
                                               kernel_size=(4, 5, 5), stride=(4, 3, 3), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=4, groups=4, 
                                               kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        else:  # dim == self.args.ADAPTER.WIDTH (final stage)
            self.s_attn = ResNet_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=8, groups=8, 
                                               kernel_size=(4, 4, 4), stride=(4, 3, 3), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=8, groups=8, 
                                               kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        
        self.gelu = nn.GELU()
        
        # Dimension restoration
        self.up = nn.Conv3d(self.adapter_channels, self.out_channels, kernel_size=1)
        self.gelu2 = nn.GELU()
        
        # Initialize adapter weights to zero (residual learning)
        nn.init.constant_(self.up.weight, 0)
        nn.init.constant_(self.up.bias, 0)
        
    def forward(self, x):
        # Input: (B*T, C, H, W) -> reshape to (B, C, T, H, W)
        x_in = x
        B_T, C, H, W = x.shape
        B = B_T // self.num_frames
        
        # Only apply adapter if we have the expected number of frames
        if B_T % self.num_frames == 0:
            x = x.reshape(B, self.num_frames, C, H, W).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # Down-project
            x = self.down(x)
            x = self.gelu1(x)
            
            # Add positional encoding
            x = x + self.pos_embed(x)
            
            # Spatial Deformable Attention
            xs = x + self.s_attn(self.s_ln(x))
            
            # Temporal Deformable Attention  
            xt = x + self.t_attn(self.t_ln(x))
            
            # Fuse pathways - exact same as ViT version
            x = (xs + xt) / 2
            x = self.gelu(x)
            
            # Up-project
            x = self.up(x)
            x = self.gelu2(x)
            
            # Back to original format
            x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(B_T, C, H, W)  # Back to (B*T, C, H, W)
            
            return x + x_in
        else:
            # If frames don't match, just return identity (no adaptation)
            return x_in

class ResidualAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.ADAPTER.WIDTH
        n_head = cfg.ADAPTER.HEADS
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.Adapter = ViT_D2ST_Adapter(cfg)

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x):
        # x shape [HW+1, BT, C]
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = self.Adapter(x)
        return x

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(cfg) for _ in range(cfg.ADAPTER.LAYERS)])

    def forward(self, x):
        return self.resblocks(x)

class D2STConfig:
    def __init__(self, num_frames=8, train_crop_size=224, num_classes=5, backbone="ViT-B/16"):
        self.DATA = type('DATA', (), {})()
        self.DATA.NUM_INPUT_FRAMES = num_frames
        self.DATA.TRAIN_CROP_SIZE = train_crop_size
        
        self.ADAPTER = type('ADAPTER', (), {})()
        
        # Auto-configure based on backbone type
        if "ViT" in backbone or "CLIP" in backbone:
            # ViT-CLIP backbone configuration
            self.ADAPTER.NAME = "ViT_CLIP"
            self.ADAPTER.PRETRAINED = backbone
            
            # ViT model-specific parameters
            if "B/16" in backbone:
                self.ADAPTER.WIDTH = 768
                self.ADAPTER.PATCH_SIZE = 16
                self.ADAPTER.LAYERS = 12
                self.ADAPTER.HEADS = 12
            elif "B/32" in backbone:
                self.ADAPTER.WIDTH = 768
                self.ADAPTER.PATCH_SIZE = 32
                self.ADAPTER.LAYERS = 12
                self.ADAPTER.HEADS = 12
            elif "L/14" in backbone:
                self.ADAPTER.WIDTH = 1024
                self.ADAPTER.PATCH_SIZE = 14
                self.ADAPTER.LAYERS = 24
                self.ADAPTER.HEADS = 16
            else:  # Default ViT-B/16
                self.ADAPTER.WIDTH = 768
                self.ADAPTER.PATCH_SIZE = 16
                self.ADAPTER.LAYERS = 12
                self.ADAPTER.HEADS = 12
                
        else:
            # ResNet backbone configuration
            self.ADAPTER.NAME = "ResNet"
            
            if "resnet18" in backbone.lower():
                self.ADAPTER.LAYERS = 18
                self.ADAPTER.WIDTH = 512
            elif "resnet34" in backbone.lower():
                self.ADAPTER.LAYERS = 34
                self.ADAPTER.WIDTH = 512
            elif "resnet101" in backbone.lower():
                self.ADAPTER.LAYERS = 101
                self.ADAPTER.WIDTH = 2048
            elif "resnet152" in backbone.lower():
                self.ADAPTER.LAYERS = 152
                self.ADAPTER.WIDTH = 2048
            else:  # Default ResNet50
                self.ADAPTER.LAYERS = 50
                self.ADAPTER.WIDTH = 2048
                
            # ResNet doesn't use these parameters, but set defaults
            self.ADAPTER.PATCH_SIZE = None
            self.ADAPTER.HEADS = None
        
        # Common adapter parameters
        self.ADAPTER.ADAPTER_SCALE = 0.25
        
        # Training configuration - auto-adjust based on backbone
        self.TRAIN = type('TRAIN', (), {})()
        
        # ViT models typically use higher classification value
        if "ViT" in backbone or "CLIP" in backbone:
            self.TRAIN.USE_CLASSIFICATION_VALUE = 2.0
        else:
            self.TRAIN.USE_CLASSIFICATION_VALUE = 1.0
            
        self.TRAIN.NUM_CLASS = num_classes
        
        # Store backbone info for easy access
        self.backbone_type = backbone

class D2ST(nn.Module):
    def __init__(self, config, disc=False, gc=False, dp=0.1):
        super(D2ST, self).__init__()
        
        # Initialize config from FSOSAR format
        self.config = config
        self.num_frames = config["seq_len"]
        self.disc = disc
        self.gc = gc
        
        # Get backbone from config
        backbone = config["backbone"]
        
        # Create D2ST config in original format
        self.args = D2STConfig(self.num_frames, 224, config.get("num_classes", 5), backbone)
        
        # Initialize based on backbone type
        if self.args.ADAPTER.NAME == "ViT_CLIP":
            self._init_vit_backbone()
        else:
            self._init_resnet_backbone()
        
        # Discriminator for open-set recognition (if needed)
        if self.disc:
            from utils import BinaryClassificationModelSAFSAR
            self.discriminator = BinaryClassificationModelSAFSAR(self.width)
        
        # Debug counter
        self.debug_samples_counter = 0

    def _init_vit_backbone(self):
        """Initialize ViT-B/16 CLIP backbone"""
        self.backbone_type = "ViT"
        self.pretrained = self.args.ADAPTER.PRETRAINED
        self.width = self.args.ADAPTER.WIDTH
        self.patch_size = self.args.ADAPTER.PATCH_SIZE
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.width, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        scale = self.width ** -0.5
        self.layers = self.args.ADAPTER.LAYERS
        self.class_embedding = nn.Parameter(scale * torch.randn(self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((self.args.DATA.TRAIN_CROP_SIZE // self.patch_size) ** 2 + 1, self.width))
        self.ln_pre = LayerNorm(self.width)
        self.num_frames = self.args.DATA.NUM_INPUT_FRAMES
        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames, self.width))
        self.transformer = Transformer(self.args)
        self.ln_post = LayerNorm(self.width)
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            self.classification_layer = nn.Linear(self.width, int(self.args.TRAIN.NUM_CLASS))
        
        # Initialize weights
        self.init_vit_weights()
        
        # Freeze backbone weights - only train adapters and specific components
        self.freeze_backbone_weights()

    def _init_resnet_backbone(self):
        """Initialize ResNet backbone with D2ST adapters"""
        self.backbone_type = "ResNet"
        self.width = self.args.ADAPTER.WIDTH
        
        # ResNet backbone
        if self.args.ADAPTER.LAYERS == 18:
            backbone = models.resnet18(pretrained=True)
            self.feature_dim = 512
        else:  # ResNet50
            backbone = models.resnet50(pretrained=True)
            self.feature_dim = 2048
            
        # Split ResNet into stages
        self.stage1 = nn.Sequential(*list(backbone.children())[:5])  # Up to layer1
        self.stage2 = list(backbone.children())[5]  # layer2
        self.stage3 = list(backbone.children())[6]  # layer3  
        self.stage4 = list(backbone.children())[7]  # layer4
        self.stage5 = nn.Sequential(*list(backbone.children())[8:-1])  # avgpool (excluding final FC)
        
        # D2ST Adapters for each stage
        adapter_scale = self.args.ADAPTER.ADAPTER_SCALE
        self.adapter1 = ResNet_D2ST_Adapter(self.args, self.feature_dim // 8, self.num_frames)
        self.adapter2 = ResNet_D2ST_Adapter(self.args, self.feature_dim // 4, self.num_frames)
        self.adapter3 = ResNet_D2ST_Adapter(self.args, self.feature_dim // 2, self.num_frames)
        self.adapter4 = ResNet_D2ST_Adapter(self.args, self.feature_dim, self.num_frames)
        
        # Freeze backbone weights - only train adapters
        self.freeze_backbone_weights()

    def init_vit_weights(self):
        logger.info(f'load model from: {self.pretrained}')
        # Load OpenAI CLIP pretrained weights
        clip_model, _ = clip.load(self.pretrained, device="cpu")
        pretrain_dict = clip_model.visual.state_dict()
        del clip_model
        if 'proj' in pretrain_dict:
            del pretrain_dict['proj']
        msg = self.load_state_dict(pretrain_dict, strict=False)
        logger.info('Missing keys: {}'.format(msg.missing_keys))
        logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        torch.cuda.empty_cache()
        # zero-initialize Adapters
        for n1, m1 in self.named_modules():
            if 'Adapter' in n1:
                for n2, m2 in m1.named_modules():
                    if 'up' in n2:
                        logger.info('init:  {}.{}'.format(n1, n2))
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)

    def freeze_backbone_weights(self):
        """Freeze backbone weights and only train adapters and specific components"""
        logger.info("Freezing backbone weights - only training adapters and specific components")
        
        for name, param in self.named_parameters():
            # Only train these components (following original D2ST implementation)
            if ('class_embedding' not in name and 
                'temporal_embedding' not in name and 
                'Adapter' not in name and 
                'adapter' not in name and  # For ResNet adapters
                'ln_post' not in name and 
                'classification_layer' not in name):
                param.requires_grad = False
                
        # Log trainable parameters
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
        
        # Log which parameters are trainable
        logger.info("Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                logger.info(f'  {name}: {param.numel()} parameters')

    def extract_class_indices(self, labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def get_feat_vit(self, x):
        """ViT feature extraction"""
        x = self.conv1(x)  # b*t c h w
        x = rearrange(x, 'b c h w -> b (h w) c')
        # b*t h*w+1 c
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        # n = h*w+1
        n = x.shape[1]

        x = rearrange(x, '(b t) n c -> (b n) t c', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t c -> (b t) n c', n=n)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x[:, 0, :]
        return x

    def get_feat_resnet(self, x):
        """ResNet feature extraction with D2ST adapters"""
        # x shape: (B*T, C, H, W)
        
        # Stage 1: conv1, bn1, relu, maxpool, layer1
        x = self.stage1(x)
        x = self.adapter1(x)
        
        # Stage 2: layer2  
        x = self.stage2(x)
        x = self.adapter2(x)
        
        # Stage 3: layer3
        x = self.stage3(x)
        x = self.adapter3(x)
        
        # Stage 4: layer4
        x = self.stage4(x)
        x = self.adapter4(x)
        
        # Stage 5: avgpool (following original implementation)
        x = self.stage5(x)
        x = x.squeeze()  # Remove spatial dimensions, following original
        
        return x

    def set_train(self):
        """Set training mode"""
        self.train()
        
    def set_eval(self):
        """Set evaluation mode"""  
        self.eval()
    
    def get_debug_data(self):
        """Return debug data for logging"""
        return {}
    
    def compute_additional_metrics(self, **kwargs):
        """Compute additional metrics for logging"""
        return {}

    def extract_features(self, x):
        """Extract features using the configured backbone"""
        # x shape: (B, T, C, H, W) -> reshape to (B*T, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        
        if self.backbone_type == "ViT":
            return self.get_feat_vit(x)
        else:
            return self.get_feat_resnet(x)

    def compute_similarity_matrix(self, support_features, query_features, support_labels):
        """Compute similarity matrix using original D2ST Bi-MHM approach"""
        # Both ViT and ResNet use the same similarity computation in original D2ST
        
        # Reshape to include temporal dimension - BOTH use ADAPTER.WIDTH as feature dim
        if self.backbone_type == "ViT":
            feat_dim = self.args.ADAPTER.WIDTH
        else:
            feat_dim = self.args.ADAPTER.WIDTH  # Not self.feature_dim!
            
        support_features = support_features.reshape(-1, self.num_frames, feat_dim)
        query_features = query_features.reshape(-1, self.num_frames, feat_dim)
        
        unique_labels = torch.unique(support_labels)

        # Compute class prototypes - EXACT same as original
        support_features_per_class = [torch.mean(torch.index_select(support_features, 0, self.extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features_per_class)

        support_num = support_features.shape[0]
        query_num = query_features.shape[0]

        support_features = support_features.unsqueeze(0).repeat(query_num, 1, 1, 1)
        support_features = rearrange(support_features, 'q s t c -> q (s t) c')

        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(query_features, dim=2).permute(0, 2, 1)).reshape(query_num, support_num, self.num_frames, self.num_frames)
        dist = 1 - frame_sim

        # Bi-MHM - EXACT same for both ViT and ResNet
        class_dist = dist.min(3)[0].sum(2) + dist.min(2)[0].sum(2)

        return -class_dist

    def forward(self, support_set, support_labels, query_set, batch_class_list):
        """Forward pass for few-shot learning"""
        
        # Extract features
        support_features = self.extract_features(support_set)
        query_features = self.extract_features(query_set)
        
        # Compute similarity matrix using D2ST approach
        similarity_matrix = self.compute_similarity_matrix(
            support_features, query_features, support_labels
        )
        
        logits = {"similarity_matrix": similarity_matrix}
        
        # Add discriminator output if needed
        if self.disc:
            # Concatenate support and query features for discriminator
            all_features = torch.cat([support_features, query_features], dim=0)
            # Reshape for discriminator: (batch, frames, features)
            if self.backbone_type == "ViT":
                disc_input = all_features.view(-1, self.num_frames, self.width)
            else:
                disc_input = all_features.view(-1, self.num_frames, self.feature_dim)
            disc_prob = self.discriminator(disc_input)
            # Only return query discriminator probabilities
            query_disc_prob = disc_prob[support_features.shape[0]:]
            logits["disc_prob"] = query_disc_prob
            
        return logits
    
    def compute_known_losses(self, similarity_matrix, target_labels, support_labels, 
                           batch_class_list, **kwargs):
        """Compute known class losses (cross-entropy)"""
        losses = {}
        
        if target_labels is not None:
            # Only compute loss for known samples (target_labels != -1)
            known_mask = target_labels != -1
            if known_mask.sum() > 0:
                known_similarity = similarity_matrix[known_mask]
                known_targets = target_labels[known_mask]
                losses["known_loss"] = F.cross_entropy(known_similarity, known_targets)
            else:
                losses["known_loss"] = torch.tensor(0.0, device=similarity_matrix.device)
        else:
            losses["known_loss"] = torch.tensor(0.0, device=similarity_matrix.device)
            
        return losses