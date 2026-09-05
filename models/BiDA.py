import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio, drop, attn_drop):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=drop, activation='gelu',
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)

    def forward(self, x):
        return self.encoder(x)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, adj, x):
        h = torch.matmul(adj, x)
        h = self.linear(h)
        return self.act(h)


class VisionGraph(nn.Module):
    """Graph convolution over the token sequence (source domain branch)."""
    def __init__(self, n_tokens, dim, drop):
        super().__init__()
        self.register_buffer('adj', self._build_chain_adj(n_tokens))
        self.gcn1 = GCNLayer(dim, dim)
        self.gcn2 = GCNLayer(dim, dim)
        self.drop = nn.Dropout(drop)

    @staticmethod
    def _build_chain_adj(n):
        A = torch.zeros(n, n)
        for i in range(n):
            A[i, i] = 1.0
            if i - 1 >= 0:
                A[i, i - 1] = 1.0
            if i + 1 < n:
                A[i, i + 1] = 1.0
        A = A + torch.eye(n)
        D = A.sum(dim=1, keepdim=True)
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        A_norm = D_inv_sqrt * A * D_inv_sqrt
        return A_norm

    def forward(self, x):
        residual = x
        h = self.drop(self.gcn1(self.adj, x))
        h = self.drop(self.gcn2(self.adj, h))
        return h + residual


class CNNEncoder(nn.Module):
    """1-D CNN over the token sequence (target domain branch)."""
    def __init__(self, dim, drop):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        # LayerNorm (not BatchNorm): avoids batch-dependent stats that would
        # be stale/broken in the EMA teacher (which runs in train mode with
        # un-updated running stats).
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = h.transpose(1, 2)
        h = self.drop(self.ln1(h))
        h = h.transpose(1, 2)
        h = F.relu(self.conv2(h))
        h = h.transpose(1, 2)
        h = self.drop(self.ln2(h))
        return h + residual


class AdaptiveGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, a, b):
        cat = torch.cat([a, b], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        return g * a + (1.0 - g) * b


class CrossAttentionRefine(nn.Module):
    def __init__(self, dim, heads, drop):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, kv):
        q = self.norm_q(query)
        k = self.norm_kv(kv)
        out, _ = self.attn(q, k, k, need_weights=False)
        return self.norm(query + out)


class TokenRefiner(nn.Module):
    """Per-domain token refiner.

    Each branch's internal flow:
        tokens -> transformer_encoder           (stream t)
        tokens -> secondary_encoder             (stream s: VisionGraph for source, CNN for target)
        fused   = adaptive_gate(t, s)
        final   = cross_attention(fused, mem=[t, s])

    The transformer_encoder / gate / cross_attention are shared across domains
    (keeping source and target token representations in a common space so the
    downstream triple-branch attention can still align them). Only the secondary
    encoder differs: VisionGraph for the source, CNNEncoder for the target.
    """
    def __init__(self, n_tokens, dim, heads, mlp_ratio, depth, drop, attn_drop, out_init=0.01):
        super().__init__()
        self.transformer_enc = TransformerEncoder(dim, depth, heads, mlp_ratio, drop, attn_drop)
        self.vision_graph = VisionGraph(n_tokens, dim, drop)
        self.cnn_encoder = CNNEncoder(dim, drop)
        self.gate = AdaptiveGate(dim)
        self.cross_attn = CrossAttentionRefine(dim, heads, attn_drop)
        # learnable output scale, small init -> the refiner starts near identity
        # (so training begins on the original BiDA trajectory) and wakes up
        # gradually instead of corrupting the shared triple-branch attention.
        self.out_scale = nn.Parameter(torch.full((1,), out_init))

    def forward(self, x, use_graph):
        t = self.transformer_enc(x)
        s = self.vision_graph(x) if use_graph else self.cnn_encoder(x)
        fused = self.gate(t, s)
        mem = torch.cat([t, s], dim=1)
        out = self.cross_attn(fused, mem)
        return out * self.out_scale


class Attention_triple_branches(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None

    def forward(self, x, x2, use_attn=True, inference_target_only=False):
        B, N, C = x2.shape
        if inference_target_only:
            qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

            attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            self.attn = attn2
            attn2 = self.attn_drop(attn2)
            
            x2 = ( attn2 @ v2 ) if use_attn else v2

            x2 = x2.transpose(1, 2).reshape(B, N, C)
            x2 = self.proj(x2)
            x2 = self.proj_drop(x2)
            x, x3, x4 = None, None, None
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
            q_st = torch.cat((q, q2), dim=0)
            k_st = torch.cat((k2, k), dim=0)
            v_st = torch.cat((v2, v), dim=0)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
            attn_st = (q_st @ k_st.transpose(-2, -1)) * self.scale
            
            attn = attn.softmax(dim=-1)
            attn2 = attn2.softmax(dim=-1)
            attn_st = attn_st.softmax(dim=-1)
            self.attn = attn
            attn = self.attn_drop(attn)
            attn2 = self.attn_drop(attn2)
            attn_st = self.attn_drop(attn_st)

            x = ( attn @ v ) if use_attn else v
            x2 = ( attn2 @ v2 ) if use_attn else v2
            x_st = ( attn_st @ v_st ) if use_attn else v_st

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            x2 = x2.transpose(1, 2).reshape(B, N, C)
            x2 = self.proj(x2)
            x2 = self.proj_drop(x2)
            
            x_st = x_st.transpose(1, 2).reshape(2*B, N, C)
            x_st = self.proj(x_st)
            x_st = self.proj_drop(x_st)
            x3, x4 = torch.split(x_st, B, dim=0)
        return x, x2, x3, x4

    
class Block_triple_branches(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_dim=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_triple_branches(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x, x2, x1_x2_fusion, inference_target_only=False):
        if inference_target_only:
            _, xa_attn2, _, _ = self.attn(None,self.norm1(x2), inference_target_only=inference_target_only)
            xb = x2 + self.drop_path(xa_attn2)
            xb = xb + self.drop_path(self.mlp(self.norm2(xb)))
            xa, xab, xba = None, None, None
        else:
            xa_attn, xa_attn2, xa_attn3, xa_attn4 = self.attn(self.norm1(x),self.norm1(x2), inference_target_only=inference_target_only)
            xa = x + self.drop_path(xa_attn)
            xa = xa + self.drop_path(self.mlp(self.norm2(xa)))

            xb = x2 + self.drop_path(xa_attn2)
            xb = xb + self.drop_path(self.mlp(self.norm2(xb)))

            xab = x1_x2_fusion + self.drop_path(xa_attn3)
            xab = xab + self.drop_path(self.mlp(self.norm2(xab)))

            xba = x + self.drop_path(xa_attn4)
            xba = xba + self.drop_path(self.mlp(self.norm2(xba)))
            
        return xa, xb, xab, xba

class BiDAnet(nn.Module):
    def __init__(self, n_bands=30, in_channels=1, num_classes=16, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8,  
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.,
                 use_refiner=1, ref_depth=1, ref_heads=8, ref_residual=1, ref_shared=1,
                 ref_init_scale=0.01):
        super(BiDAnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.token_len = num_tokens
        self.use_refiner = use_refiner
        self.ref_residual = ref_residual
        self.ref_shared = ref_shared
        self.conv_a = nn.Conv2d(dim, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*n_bands, out_channels=dim,
                      kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block_triple_branches(
                dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        
        self.to_cls_token = nn.Identity()

        n_seq = num_tokens + 1  # cls token + semantic tokens
        if self.use_refiner:
            if self.ref_shared:
                # one shared refiner core (VisionGraph + CNN encoder inside,
                # selected per-domain at forward time)
                self.refiner = TokenRefiner(
                    n_seq, dim, ref_heads, mlp_ratio, ref_depth,
                    drop_rate, attn_drop_rate, out_init=ref_init_scale)
                self.src_refiner = None
                self.tar_refiner = None
            else:
                self.refiner = None
                self.src_refiner = TokenRefiner(
                    n_seq, dim, ref_heads, mlp_ratio, ref_depth,
                    drop_rate, attn_drop_rate, out_init=ref_init_scale)
                self.tar_refiner = TokenRefiner(
                    n_seq, dim, ref_heads, mlp_ratio, ref_depth,
                    drop_rate, attn_drop_rate, out_init=ref_init_scale)
        else:
            self.refiner = None
            self.src_refiner = None
            self.tar_refiner = None

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _call_refiner(self, x, use_graph):
        if self.ref_shared:
            return self.refiner(x, use_graph=use_graph)
        if use_graph:
            return self.src_refiner(x, use_graph=True)
        return self.tar_refiner(x, use_graph=False)

    def _tokenize(self, x):
        x = self.conv3d_features(x)  # torch.Size([128, 1, 64, 13, 13])
        # torch.Size([128, 496, 11, 11])
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)  # torch.Size([128, 64, 9, 9])
        T = self._forward_semantic_tokens(x)
        return T
    
    def forward(self, x, x_tar, inference_target_only=False, return_feat_prob=False):
        T = self._tokenize(x)
        T_tar = self._tokenize(x_tar)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x_tar = torch.cat((cls_tokens, T_tar), dim=1)
        x += self.pos_embedding
        x_tar += self.pos_embedding
        x = self.dropout(x)  # torch.Size([128, 5, 64])
        x_tar = self.dropout(x_tar)  # torch.Size([128, 5, 64])

        if self.use_refiner:
            if self.ref_residual:
                def _ref(inp, use_graph):
                    return inp + self._call_refiner(inp, use_graph)
            else:
                def _ref(inp, use_graph):
                    return self._call_refiner(inp, use_graph)
            if self.training:
                x = _ref(x, use_graph=True)
                x_tar = _ref(x_tar, use_graph=False)
            else:
                x_tar = _ref(x_tar, use_graph=False)

        inference_target_only = not self.training
        x_fusion = x_tar
        for i, blk in enumerate(self.blocks):
            x, x_tar, x_fusion, x_fusion_src = blk(
                x, x_tar, x_fusion, inference_target_only=inference_target_only)
        if inference_target_only:
        # if inference_target_only:
            x_tar = self.norm(x_tar)
            out_x_tar = self.nn1(self.to_cls_token(x_tar[:, 0])) 
            if return_feat_prob:
                return None, out_x_tar, None, x_tar[:, 0]
            else:
                return None, out_x_tar, None
        else:
            x = self.norm(x)
            x_tar = self.norm(x_tar)
            x_fusion = self.norm(x_fusion)
            x_fusion_src = self.norm(x_fusion_src)
            out_x = self.nn1(self.to_cls_token(x[:, 0]))  # torch.Size([128, 64])
            out_x_tar = self.nn1(self.to_cls_token(x_tar[:, 0]))  # torch.Size([128, 64])
            out_x_fusion = self.nn1(self.to_cls_token(x_fusion[:, 0]))  # torch.Size([128, 64])
            out_fusion_src = self.nn1(self.to_cls_token(x_fusion_src[:, 0]))  # torch.Size([128, 64])
            return out_x, out_x_tar, out_x_fusion, out_fusion_src

def BiDA(dataset, opts):
    model = None
    use_refiner = getattr(opts, 'use_refiner', 1)
    ref_depth = getattr(opts, 'ref_depth', 1)
    ref_heads = getattr(opts, 'ref_heads', 8)
    ref_residual = getattr(opts, 'ref_residual', 1)
    ref_shared = getattr(opts, 'ref_shared', 1)
    ref_init_scale = getattr(opts, 'ref_init_scale', 0.01)
    if 'MJG' in dataset.split('_'):
        model = BiDAnet(n_bands=64, num_classes=5,
                         num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth,
                         use_refiner=use_refiner, ref_depth=ref_depth, ref_heads=ref_heads,
                         ref_residual=ref_residual, ref_shared=ref_shared,
                         ref_init_scale=ref_init_scale)
    elif dataset == 'Houston18' or dataset == 'Houston13':
        model = BiDAnet(n_bands=48, num_classes=7,
                         num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth,
                         use_refiner=use_refiner, ref_depth=ref_depth, ref_heads=ref_heads,
                         ref_residual=ref_residual, ref_shared=ref_shared,
                         ref_init_scale=ref_init_scale)
    elif dataset == 'Dioni' or dataset == 'Loukia':
        model = BiDAnet(n_bands=176, num_classes=12,
                         num_tokens=opts.num_tokens, dim=opts.dim, depth=opts.depth,
                         use_refiner=use_refiner, ref_depth=ref_depth, ref_heads=ref_heads,
                         ref_residual=ref_residual, ref_shared=ref_shared,
                         ref_init_scale=ref_init_scale)

    return model


if __name__ == '__main__':
    t = torch.randn(size=(3, 1, 204, 7, 7))
    print("input shape:", t.shape)
    net = BiDA(dataset='sa', patch_size=7)
    print("output shape:", net(t).shape)


