import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
import logging
import copy

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'geovig': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)
    
logger = logging.getLogger(__name__)

class Stem(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.GELU):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU()   
        )
        
    def forward(self, x):
        return self.stem(x)

class GraphAdapter(nn.Module):
    """
    Graph-Aware Adapter: Lightweight 3x3 DW Conv + GroupNorm + 1x1 Conv + GroupNorm.
    Aligns graph features with grid-based detection heads.
    """
    def __init__(self, channels):
        super().__init__()
        self.adapter = nn.Sequential(
            # Depthwise 3x3
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            # GroupNorm (num_groups=1 is equivalent to LayerNorm but works with 4D tensors)
            nn.GroupNorm(1, channels), 
            # Pointwise 1x1
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels), # GroupNorm for stability
            nn.GELU()
        )

    def forward(self, x):
        return self.adapter(x)

    

class MLP(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x

    
class InvertedResidual(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x
    

def img_to_graph(x):
    """
    Converts (B, C, H, W) -> (B, N, C) and generates a grid adjacency (8-neighbor-ish).
    For efficiency this returns a single edge_index shared for the batch.
    """
    B, C, H, W = x.shape
    device = x.device
    N = H * W

    # Flatten features: (B, C, H, W) -> (B, N, C)
    x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

    # Generate grid edges (simplified 8-neighbor connectivity)
    idx = torch.arange(N, device=device).view(H, W)
    edges = []

    # Right neighbors
    if W > 1:
        edges.append(torch.stack([idx[:, :-1].flatten(), idx[:, 1:].flatten()], dim=0))
        # Left (will be mirrored later)
    # Down neighbors
    if H > 1:
        edges.append(torch.stack([idx[:-1, :].flatten(), idx[1:, :].flatten()], dim=0))
    # Diagonals
    if H > 1 and W > 1:
        edges.append(torch.stack([idx[:-1, :-1].flatten(), idx[1:, 1:].flatten()], dim=0))
        edges.append(torch.stack([idx[1:, :-1].flatten(), idx[:-1, 1:].flatten()], dim=0))

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.cat(edges, dim=1)
        # Make undirected by mirroring
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    return x_flat, edge_index


class GraphMRConv(nn.Module):
    """
    Graph version of Max-Relative Graph Convolution.
    Operates on (B, N, C) and edge_index (2, E).
    Uses PyTorch's index_reduce_ with 'amax' for true max aggregation.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # OPTIMIZATION: Removed concatenation (2*C). Now just Linear(C -> C).
        # This reduces parameters significantly and speeds up the projection.
        self.nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU()
        )

    def forward(self, x, edge_index):
        # x: (B, N, C)
        B, N, C = x.shape
        device = x.device

        # Reshape to (B*N, C) for batched processing
        x_flat = x.view(-1, C)

        if edge_index.numel() == 0:
            # No edges: fallback to identity-like behavior with zeros
            # Return processed x_flat reshaped, or zeros if safer
            # Ideally graph conv with no edges is just local projection of 0-vector (since it's rel diff)
            # max(x_j) would be empty -> -inf -> replaced by 0
            # then -x_i
            # so effectively -x_i passed through nn
            return self.nn(-x).view(B, N, -1)

        row, col = edge_index  # (E,), (E,)
        num_edges = row.size(0)

        # Create batched edge index: (2, B*E)
        batch_offsets = torch.arange(B, device=device) * N
        edge_offsets = batch_offsets.view(-1, 1).repeat(1, num_edges).view(-1)
        
        row_batch = row.repeat(B) + edge_offsets
        col_batch = col.repeat(B) + edge_offsets

        # Gather features
        x_flat_c = x_flat.contiguous() 
        # Optimization: Don't compute x_j - x_i explicitly (huge tensor).
        # max(x_j - x_i) = max(x_j) - x_i
        x_j = x_flat_c[col_batch]  # (B*E, C)

        # Initialize aggregation buffer
        aggr = torch.full_like(x_flat, -1e9)  # (B*N, C)
        
        # Vectorized Max Aggregation of Neighbors
        index_expanded = row_batch.view(-1, 1).expand(-1, C)
        aggr.scatter_reduce_(0, index_expanded, x_j, reduce='amax', include_self=True)
        
        # Replace -1e9 with 0 
        aggr = torch.where(aggr == -1e9, torch.zeros_like(aggr), aggr)
        
        # Now apply the "- x_i" part of "max(x_j) - x_i"
        aggr = aggr - x_flat

        # Output only the aggregated update
        return self.nn(aggr).view(B, N, -1)


class GraphMobileViGBlock(nn.Module):
    """
    Graph MobileViG block: GraphMRConv + FFN for (B,N,C) representations.
    """
    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.graph_conv = GraphMRConv(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, edge_index):
        # x: (B, N, C)
        shortcut = x
        x = self.norm1(x)
        x = self.graph_conv(x, edge_index)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + shortcut
        return x


class SpreadEdgePool(nn.Module):
    """
    Geometry-Aware SpreadEdgePool
    """
    def __init__(self, in_channels, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.in_channels = in_channels

    def forward(self, x, edge_index, shape=None):
        B, N, C = x.shape
        device = x.device

        # 2D Scaling Path (For Detection) - STABLE MODE
        # Skips distance-based weighting to avoid instability in detection fine-tuning
        if shape is not None:
            H, W = shape
            if H * W == N:
                x_2d = x.transpose(1, 2).view(B, C, H, W)
                
                # Force Stride 2 for Detection (Target Stride 32 from Stride 16 input)
                # ratio=0.25 implies 1/4 area -> stride 2
                stride = 2 if self.ratio <= 0.25 else 1 
                
                x_pooled_2d = F.avg_pool2d(x_2d, kernel_size=stride, stride=stride)
                
                # Regenerate Grid Graph
                x_pooled, new_edge_index = img_to_graph(x_pooled_2d)
                return x_pooled, new_edge_index

        # Default 1D Graph Pooling (original classification logic)
        num_keep = max(1, int(N * self.ratio))
        
        # Only compute weights if strictly needed (1D path)
        if edge_index.numel() > 0:
            row, col = edge_index
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)
            sq_i = x_sq[:, row, :]
            sq_j = x_sq[:, col, :]
            x_row = x[:, row, :]
            x_col = x[:, col, :]
            dot = (x_row * x_col).sum(dim=-1, keepdim=True)
            dist_sq = sq_i + sq_j - 2 * dot
            dist = torch.sqrt(F.relu(dist_sq) + 1e-6).squeeze(-1)
            avg_edge_scores = dist.mean(dim=0)
            node_importance = torch.zeros(N, device=device)
            node_importance.index_add_(0, row, avg_edge_scores)
            node_weights = torch.sigmoid(node_importance).view(1, N, 1)
        else:
            node_weights = torch.ones((1, N, 1), device=device)

        x_weighted = x * node_weights
        num_keep = max(1, int(N * self.ratio))
        
        x_pooled = F.adaptive_avg_pool1d(x_weighted.transpose(1, 2), num_keep).transpose(1, 2)

        if edge_index.numel() == 0:
            idx = torch.arange(num_keep, device=device)
            new_edge_index = torch.stack([idx[:-1], idx[1:]], dim=0) if num_keep > 1 else torch.zeros((2, 0), dtype=torch.long, device=device)
            return x_pooled, new_edge_index

        idx = torch.arange(num_keep, device=device)
        if num_keep > 1:
            left = idx[:-1]
            right = idx[1:]
            new_edge_index = torch.cat([torch.stack([left, right], dim=0), torch.stack([right, left], dim=0)], dim=1)
        else:
            new_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        return x_pooled, new_edge_index


class Downsample(nn.Module):
    """
    Convolution-based downsample
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class GeoViG(nn.Module):
    """
    Hybrid Grid->Graph GeoViG model. Can be used for classification or as a backbone for detection.
    """
    def __init__(self,
                 local_blocks=[2, 2],
                 local_channels=[42, 84],
                 graph_blocks=[2],
                 graph_channels=[168],
                 num_classes=1000,
                 pool_ratio=0.5,
                 distillation=False,
                 features_only=False,
                 out_indices=None,
                 pretrained=None,
                 use_detect_adapter=False,
                 **kwargs):
        super().__init__()

        self.features_only = features_only
        self.pool_ratio = pool_ratio
        self.out_indices = out_indices
        self.distillation = distillation
        self.pretrained = pretrained
        self.use_detect_adapter = use_detect_adapter

        # Stage 0: stem
        self.stem = Stem(input_dim=3, output_dim=local_channels[0])

        # Grid stages
        self.grid_stages = nn.ModuleList()
        in_c = local_channels[0]
        
        self.feature_info = [dict(num_chs=in_c, reduction=2, module=f'stem')]

        for i, out_c in enumerate(local_channels):
            layers = []
            if i > 0:
                layers.append(Downsample(in_c, out_c))
                in_c = out_c
            for _ in range(local_blocks[i]):
                layers.append(InvertedResidual(dim=out_c, mlp_ratio=4, drop_path=0.0))
            self.grid_stages.append(nn.Sequential(*layers))
            self.feature_info.append(dict(num_chs=out_c, reduction=2**(i+2), module=f'grid_stages.{i}'))

        # Transition: grid -> graph
        self.to_graph_pool = SpreadEdgePool(in_channels=in_c, ratio=pool_ratio)
        self.graph_dim = in_c

        # Graph stages
        self.graph_proj = nn.Linear(self.graph_dim, graph_channels[0]) if graph_channels else nn.Identity()
        self.graph_dim = graph_channels[0] if graph_channels else self.graph_dim

        self.graph_stages = nn.ModuleList()
        for _ in range(graph_blocks[0] if len(graph_blocks) > 0 else 0):
            self.graph_stages.append(GraphMobileViGBlock(dim=self.graph_dim))
        
        self.feature_info.append(dict(num_chs=self.graph_dim, reduction=2**(len(local_channels)+1), module='graph_stages'))

        if not self.features_only:
            # Head for classification
            self.norm = nn.LayerNorm(self.graph_dim)
            self.head = nn.Linear(self.graph_dim, num_classes)
            if self.distillation:
                self.head_dist = nn.Linear(self.graph_dim, num_classes)
        else:
            # For torchvision FPN
            self._out_channels = []
            if self.out_indices:
                for i in self.out_indices:
                    # The indices for feature_info are shifted by 1 compared to grid_stages
                    # features_info: [stem(idx0), grid0(idx1), grid1(idx2), grid2(idx3), graph(idx4)]
                    # grid_stages: [grid0, grid1, grid2] (indices 0, 1, 2)
                    pass

        # Detection Adapter Initialization
        if self.use_detect_adapter and self.features_only:
            # Determine channels automatically
            # Stem: local_channels[0] (but usually we tap grid stages)
            # Grid Stages output: local_channels
            # Graph Stage output: graph_channels[0]
            
            # Map indices to channels
            # We assume out_indices=[0, 1, 2, 3] corresponds to [grid0, grid1, grid2, graph]
            # Verify this mapping convention from forward loop
            self.detect_adapters = nn.ModuleList()
            
            # This is a bit dynamic because out_indices can vary, but assuming standard GeoViG-M/B
            # and standard out_indices=[0, 1, 2, 3]
            
            current_channels = []
            # Indices refer to stages
            # 0 -> grid_stages[0] (local_channels[0])
            # 1 -> grid_stages[1] (local_channels[1])
            # 2 -> grid_stages[2] (local_channels[2])
            # 3 -> graph_stage (graph_channels[0]) (if present)
            
            all_channels = local_channels[:]
            if graph_channels:
                all_channels.append(graph_channels[0])
                
            for idx in (self.out_indices or []):
                if idx < len(all_channels):
                    self.detect_adapters.append(GraphAdapter(all_channels[idx]))
                else:
                    # Fallback or error
                    pass

        self.init_weights()
        if pretrained:
            self._load_pretrained_weights()
        
        # self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def init_weights(self):
        if self.pretrained and isinstance(self.pretrained, str):
            logger = get_root_logger()
            print(f"Pretrained weights being being loaded from {self.pretrained}")
            logger.warn(f'Pretrained weights being loaded from {self.pretrained}')
            self._load_pretrained_weights()
        else:
            # Init random weights if needed or rely on default init
            pass

    def _load_pretrained_weights(self):
        """Load pretrained weights from checkpoint."""
        if self.pretrained and isinstance(self.pretrained, str):
            logging.info(f"Loading pretrained weights from {self.pretrained}")
            try:
                # Use mmcv loader if available
                ckpt = _load_checkpoint(self.pretrained, logger=logging.getLogger(), map_location='cpu')
            except:
                ckpt = torch.load(self.pretrained, map_location='cpu')

            print("ckpt keys: ", ckpt.keys())
            if 'state_dict_ema' in ckpt:
                state_dict = ckpt['state_dict_ema']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            model_dict = self.state_dict()
            
            # Simple permissive loading
            pretrained_dict = {}
            for k, v in state_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        # Exclude classification head logic
                        if 'head' not in k:
                            pretrained_dict[k] = v
                    else:
                        print(f"Shape mismatch for {k}: {v.shape} vs {model_dict[k].shape}")

            missing, unexpected = self.load_state_dict(pretrained_dict, strict=False)
            print("missing_keys: ", missing)
            print("unexpected_keys: ", unexpected)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        # For detection, keep BatchNorm in eval mode
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, stage in enumerate(self.grid_stages):
            x = stage(x)
            if self.features_only and self.out_indices and i in self.out_indices:
                outs.append(x)
        
        # Graph processing
        B, C, H, W = x.shape
        x_nodes, edge_index = img_to_graph(x)
        # We need to know the stride used by SpreadEdgePool to reconstruct the shape later
        # Logic matches SpreadEdgePool: stride 2 if ratio <= 0.25 else 1
        pool_stride = 2 if self.pool_ratio <= 0.25 else 1
        
        x_nodes, edge_index = self.to_graph_pool(x_nodes, edge_index, shape=(H, W))
        x_nodes = self.graph_proj(x_nodes)
        
        for block in self.graph_stages:
            x_nodes = block(x_nodes, edge_index)
        
        if self.features_only:
            graph_stage_idx = len(self.grid_stages)
            if self.out_indices and graph_stage_idx in self.out_indices:
                # Calculate correct new dimensions
                new_h = H // pool_stride
                new_w = W // pool_stride
                
                # Verify shape matches
                if new_h * new_w != x_nodes.shape[1]:
                    logger.warning(
                        "Shape mismatch in Graph Stage reshape. Expected %dx%d=%d, got %d",
                        new_h, new_w, new_h * new_w, x_nodes.shape[1]
                    )
                    # Fallback to square if totally broken (should not happen with stable pooling)
                    num_nodes = x_nodes.shape[1]
                    new_h = int(num_nodes**0.5)
                    new_w = int(num_nodes**0.5)
                    if new_h * new_w != num_nodes:
                         x_nodes = x_nodes[:, :new_h*new_w, :]

                x_graph = x_nodes.transpose(1, 2).view(B, self.graph_dim, new_h, new_w)
                outs.append(x_graph)
            
            # Apply Adapters and Clamping
            final_outs = []
            for i, feat in enumerate(outs):
                # Stability Fix: Clamp
                feat = torch.clamp(feat, min=-10.0, max=10.0)
                
                # Adaptation: Apply GraphAdapter if enabled
                if self.use_detect_adapter and hasattr(self, 'detect_adapters') and i < len(self.detect_adapters):
                     feat = self.detect_adapters[i](feat)
                
                final_outs.append(feat)
            
            return tuple(final_outs)

        # Global readout (mean) for classification
        x_global = x_nodes.mean(dim=1)
        x_out = self.norm(x_global)
        
        if self.distillation:
            x_main = self.head(x_out)
            x_dist = self.head_dist(x_out)
            if self.training:
                return x_main, x_dist
            else:
                return (x_main + x_dist) / 2
        else:
            return self.head(x_out)


if has_mmdet:
    @det_BACKBONES.register_module()
    def geovig_m_feat(pretrained=False, **kwargs):
        # GeoViG-M
        # Hack: Filter out ResNet50 pretrained string if it leaks from config
        if pretrained == 'torchvision://resnet50':
            print("Warning: Detected 'torchvision://resnet50' in pretrained arg. Overriding with GeoViG-M default.")
            pretrained = True
            
        real_pretrained = '../../geovig_m_5e4_8G_300_80_70/checkpoint.pth'
        
        # Use passed pretrained if it's a specific path, else use default if True
        load_path = real_pretrained if pretrained is True else pretrained
        
        model = GeoViG(local_blocks=[3, 3, 9], local_channels=[64, 128, 256],
                   graph_blocks=[3], graph_channels=[384],
                   pool_ratio=0.25,
                   features_only=True,
                   out_indices=[0, 1, 2, 3],
                   pretrained=load_path)
        model.default_cfg = default_cfgs['geovig']
        return model

    @det_BACKBONES.register_module()
    def geovig_b_feat(pretrained=False, **kwargs):
        # GeoViG-B
        if pretrained == 'torchvision://resnet50':
            print("Warning: Detected 'torchvision://resnet50' in pretrained arg. Overriding with GeoViG-B default.")
            pretrained = True
        
        real_pretrained = '../../geovig_b_5e4_8G_300_82_38/checkpoint.pth'

        load_path = real_pretrained if pretrained is True else pretrained
        
        model = GeoViG(local_blocks=[5, 5, 9], local_channels=[80, 160, 320],
                   graph_blocks=[4], graph_channels=[512],
                   pool_ratio=0.25,
                   features_only=True,
                   out_indices=[0, 1, 2, 3],
                   pretrained=load_path)
        model.default_cfg = default_cfgs['geovig']
        return model

