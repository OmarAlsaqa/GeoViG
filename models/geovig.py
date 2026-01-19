import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
import logging

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath
from timm.models import register_model

'''
@article{han2022vision,
  title={Vision GNN: An Image is Worth Graph of Nodes},
  author={Han, Kai and Wang, Yunhe and Guo, Jianyuan and Tang, Yehui and Wu, Enhua},
  journal={arXiv preprint arXiv:2206.00272},
  year={2022}
}
'''

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
    'mobilevig': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}

    
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
    

# -------------------- New graph utilities and GeoViG additions --------------------

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
            # No edges: fallback to concatenation with zeros
            # If no edges, no aggregation, so the update is zero.
            # We apply the non-linearity to a zero tensor to maintain shape and type.
            return self.nn(torch.zeros_like(x_flat)).view(B, N, -1)

        row, col = edge_index  # (E,), (E,)
        num_edges = row.size(0)

        # Create batched edge index: (2, B*E)
        batch_offsets = torch.arange(B, device=device) * N
        edge_offsets = batch_offsets.view(-1, 1).repeat(1, num_edges).view(-1)
        
        row_batch = row.repeat(B) + edge_offsets
        col_batch = col.repeat(B) + edge_offsets

        # Gather features
        # Gather features
        x_flat_c = x_flat.contiguous() # Ensure memory layout for scatter
        # Optimization: Don't compute x_j - x_i explicitly (huge tensor).
        # max(x_j - x_i) = max(x_j) - x_i
        x_j = x_flat_c[col_batch]  # (B*E, C)

        # Initialize aggregation buffer
        aggr = torch.full_like(x_flat, -1e9)  # (B*N, C)
        
        # Vectorized Max Aggregation of Neighbors
        index_expanded = row_batch.view(-1, 1).expand(-1, C)
        aggr.scatter_reduce_(0, index_expanded, x_j, reduce='amax', include_self=True)
        
        # Replace -1e9 with 0 (although strictly if we subtract x_i later, we need to handle no-neighbor case carefully)
        # If include_self=True, then x_i is included in x_j? No. 
        # include_self=True in scatter_reduce_ means "reduction includes the initial value of self (aggr)".
        # Our initial value is -1e9. So it works as a safe initializer.
        
        aggr = torch.where(aggr == -1e9, torch.zeros_like(aggr), aggr)
        
        # Now apply the "- x_i" part of "max(x_j) - x_i"
        # aggr is now max(neighbors). 
        # Result = max(neighbors) - center
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
    Geometry-Aware SpreadEdgePool: uses edge scores (diffusion distance proxy) as 
    attention weights to guide feature pooling. Ensures nodes with high structural 
    diversity contribute more to the pooled representation.
    """
    def __init__(self, in_channels, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.in_channels = in_channels

    def forward(self, x, edge_index):
        """
        x: (B, N, C)
        edge_index: (2, E)
        Returns:
            x_pooled: (B, N_new, C) with geometry-aware weighting
            new_edge_index: (2, E_new)
        """
        B, N, C = x.shape
        device = x.device
        num_keep = max(1, int(N * self.ratio))

        # Fallback if no edges
        if edge_index.numel() == 0:
            x_pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), num_keep).transpose(1, 2)
            idx = torch.arange(num_keep, device=device)
            new_edge_index = torch.stack([idx[:-1], idx[1:]], dim=0) if num_keep > 1 else torch.zeros((2, 0), dtype=torch.long, device=device)
            return x_pooled, new_edge_index

        row, col = edge_index  # (E,), (E,)

        # Geometry-Aware Scoring: Diffusion distance proxy via feature space
        # Optimization: Calculate distance WITHOUT materializing (B, E, C) difference tensor.
        # |A - B|^2 = |A|^2 + |B|^2 - 2(A . B)
        
        # Precompute norms: (B, N, 1)
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)
        
        # Gather norms: (B, E, 1)
        sq_i = x_sq[:, row, :]
        sq_j = x_sq[:, col, :]
        
        # Gather features just for dot (this still uses memory but dot reduces it immediately)
        x_row = x[:, row, :]  # (B, E, C)
        x_col = x[:, col, :]  # (B, E, C)
        dot = (x_row * x_col).sum(dim=-1, keepdim=True) # (B, E, 1)
        
        # Combine
        dist_sq = sq_i + sq_j - 2 * dot
        dist = torch.sqrt(F.relu(dist_sq) + 1e-6).squeeze(-1) # (B, E)

        # Spread score: high score = high distance = high structural diversity
        # Original: "similarity = torch.exp(-dist)", "edge_scores = 1.0 / (similarity + 1e-6)"
        # This means edge_scores = exp(dist). So higher distance means higher score.
        # Using distance directly achieves this.
        edge_scores = dist # Linear distance directly. 
        
        # Map edge scores to node importance: sum scores of incident edges
        # Average over batch for global topology understanding
        avg_edge_scores = edge_scores.mean(dim=0)  # (E,)
        
        # Aggregate edge importance to nodes via index_add_
        node_importance = torch.zeros(N, device=device)
        node_importance.index_add_(0, row, avg_edge_scores)  # Sum incoming edge scores per node
        
        # Normalize importance scores to [0, 1] via sigmoid
        node_weights = torch.sigmoid(node_importance).view(1, N, 1)  # (1, N, 1) for broadcasting

        # Geometry-Aware Pooling: Weight features by their geometric importance before aggregation
        x_weighted = x * node_weights  # (B, N, C), broadcast multiply
        
        # Adaptive pooling now prioritizes geometrically important features
        x_pooled = F.adaptive_avg_pool1d(x_weighted.transpose(1, 2), num_keep).transpose(1, 2)  # (B, num_keep, C)

        # Build heuristic new adjacency: simple chain / local ring
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
                 **kwargs):
        super().__init__()

        self.features_only = features_only
        self.out_indices = out_indices
        self.distillation = distillation
        self.pretrained = pretrained

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
                    self._out_channels.append(self.feature_info[i+1]['num_chs'])

        if self.pretrained:
            self._load_pretrained_weights()

    @property
    def out_channels(self):
        if not self.features_only:
            raise ValueError("out_channels is only available in features_only mode.")
        return self._out_channels

    def _load_pretrained_weights(self):
        """Load pretrained weights from checkpoint."""
        if self.pretrained and isinstance(self.pretrained, str):
            logging.info(f"Loading pretrained weights from {self.pretrained}")
            state_dict = torch.load(self.pretrained, map_location='cpu', weights_only=False)
            
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            model_dict = self.state_dict()
            
            # Filter out keys that are not in the current model, have different shapes, or belong to the classification head
            pretrained_dict = {
                k: v for k, v in state_dict.items() 
                if k in model_dict and v.shape == model_dict[k].shape and 'head' not in k
            }
            
            model_dict.update(pretrained_dict) 
            self.load_state_dict(model_dict, strict=False)
            logging.info(f"Loaded {len(pretrained_dict)} keys from pretrained weights.")

    def train(self, mode=True):
        super().train(mode)
        # For detection, keep BatchNorm in eval mode
        if self.features_only:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, x):
        outs = {}
        x = self.stem(x)

        for i, stage in enumerate(self.grid_stages):
            x = stage(x)
            if self.features_only and self.out_indices and i in self.out_indices:
                outs[str(i)] = x
        
        # Graph processing
        B, C, H, W = x.shape
        x_nodes, edge_index = img_to_graph(x)
        x_nodes, edge_index = self.to_graph_pool(x_nodes, edge_index)
        x_nodes = self.graph_proj(x_nodes)
        
        for block in self.graph_stages:
            x_nodes = block(x_nodes, edge_index)
        
        if self.features_only:
            graph_stage_idx = len(self.grid_stages)
            if self.out_indices and graph_stage_idx in self.out_indices:
                num_nodes = x_nodes.shape[1]
                new_h = int(num_nodes**0.5)
                new_w = int(num_nodes**0.5)
                if new_h * new_w != num_nodes:
                    new_h = new_w = int(num_nodes**0.5)
                    x_nodes = x_nodes[:, :new_h*new_w, :]
                
                x_graph = x_nodes.transpose(1, 2).view(B, self.graph_dim, new_h, new_w)
                outs[str(graph_stage_idx)] = x_graph
            
            return outs

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


# Register a small GeoViG variant for timm-style usage
@register_model
def geovig_ti(pretrained=False, **kwargs):
    # Target: ~4.0M Params, Lower MACs
    model = GeoViG(local_blocks=[3, 3, 3], local_channels=[48, 96, 192], 
                   graph_blocks=[3], graph_channels=[256], 
                   pool_ratio=0.5,
                   pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['mobilevig']
    return model

@register_model
def geovig_s(pretrained=False, **kwargs):
    # Target: ~5.1M Params
    model = GeoViG(local_blocks=[3, 3, 4], local_channels=[56, 112, 224], 
                   graph_blocks=[3], graph_channels=[288], 
                   pool_ratio=0.5,
                   pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['mobilevig']
    return model

@register_model
def geovig_m(pretrained=False, **kwargs):
    # Target: ~10M Params (MobileViG-M: 14M)
    model = GeoViG(local_blocks=[3, 3, 9], local_channels=[64, 128, 256],
                   graph_blocks=[3], graph_channels=[384],
                   pool_ratio=0.5,
                   pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['mobilevig']
    return model

@register_model
def geovig_b(pretrained=False, **kwargs):
    # Target: ~19M Params (MobileViG-B: 26.7M)
    model = GeoViG(local_blocks=[5, 5, 9], local_channels=[80, 160, 320],
                   graph_blocks=[4], graph_channels=[512],
                   pool_ratio=0.5,
                   pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['mobilevig']
    return model
