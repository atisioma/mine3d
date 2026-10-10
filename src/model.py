"""
模型定义 — 包含 PointNetSeg (轻量基线) 和 LitePT (CVPR 2026)
=============================================================
PointNetSeg: 简单 MLP + global pooling 语义分割, 无需额外依赖
LitePT:      来自 LitePT/litept/model.py 的完整实现
             依赖: spconv, flash_attn, torch_scatter, addict, timm
"""
import sys
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Part 1: PointNetSeg — 轻量语义分割基线
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PointNetSeg(nn.Module):
    """
    简化版 PointNet 语义分割
    输入: (B, N, 6) — x y z r g b
    输出: (B, N, num_classes)
    """

    def __init__(self, num_classes=3, in_ch=6, feat_ch=128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            ConvBNReLU(in_ch, 64),
            ConvBNReLU(64, 64),
            ConvBNReLU(64, feat_ch),
        )
        self.classifier = nn.Sequential(
            ConvBNReLU(feat_ch * 2, 256),
            ConvBNReLU(256, 128),
            ConvBNReLU(128, 64),
            nn.Conv1d(64, num_classes, 1),
        )

    def forward(self, x):
        B, N, _ = x.shape
        feat = x.permute(0, 2, 1)
        feat = self.point_mlp(feat)
        global_feat = feat.max(dim=2, keepdim=True).values
        global_feat = global_feat.expand(-1, -1, N)
        combined = torch.cat([feat, global_feat], dim=1)
        out = self.classifier(combined)
        return out.permute(0, 2, 1)


# ============================================================
# Part 2: LitePT — 完整实现 (源自 LitePT/litept/model.py)
# ============================================================
# 依赖: spconv, flash_attn, torch_scatter, addict, timm
# 可选: libs.pointrope (已有纯 PyTorch fallback)

_has_litept_deps = True
try:
    import spconv.pytorch as spconv
    import flash_attn
    import torch_scatter
    from addict import Dict as AddictDict
    from timm.layers import DropPath
    from src.libs.pointrope import PointROPE
except ImportError as e:
    _has_litept_deps = False
    _missing_dep = str(e)


def _check_litept_deps():
    if not _has_litept_deps:
        raise ImportError(
            f"LitePT 需要额外依赖 (缺失: {_missing_dep})\n"
            "请安装: pip install spconv-cu124 flash-attn torch-scatter addict timm\n"
            "并编译 libs/pointrope (或使用纯 PyTorch fallback)"
        )


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)

@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(AddictDict if _has_litept_deps else dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) is a dictionary that contains various properties of
    a batched point cloud:
    - "coord": original coordinate of point cloud
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling)
    Optional:
    - "offset": separator of point clouds in batched data
    - "batch": batch index of each point
    - "feat": feature of point cloud, default input of model
    - "grid_size": Grid size of point cloud (related to GridSampling)
    - "serialized_depth", "serialized_code", "serialized_order", "serialized_inverse"
    - "sparse_shape", "sparse_conv_feat"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        from src.utils.serialization import encode

        self["order"] = order
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max() + 1).bit_length()
        self["serialized_depth"] = depth
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        code = [
            encode(self.grid_coord, self.batch, depth, order=order_)
            for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        _check_litept_deps()
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    @torch.no_grad()
    def get_padding_and_inverse(self, patch_size):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in self.keys()
            or unpad_key not in self.keys()
            or cu_seqlens_key not in self.keys()
        ):
            offset = self.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(bincount + patch_size - 1, patch_size, rounding_mode="trunc")
                * patch_size
            )
            mask_pad = bincount > patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i]:_offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1] - patch_size + (bincount[i] % patch_size):_offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1] - 2 * patch_size + (bincount[i] % patch_size):_offset_pad[i + 1] - patch_size
                    ]
                pad[_offset_pad[i]:_offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i], _offset_pad[i + 1],
                        step=patch_size, dtype=torch.int32, device=offset.device,
                    )
                )
            self[pad_key] = pad
            self[unpad_key] = unpad
            self[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return self[pad_key], self[unpad_key], self[cu_seqlens_key]


class PointModule(nn.Module):
    r"""placeholder, all module subclass from this will take Point in PointSequential."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container for Point modules."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index out of range")
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        _check_litept_deps()
        for k, module in self._modules.items():
            if isinstance(module, PointModule):
                input = module(input)
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(input.feat)
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PointROPEAttention(PointModule):
    def __init__(self, channels, num_heads, patch_size, rope_freq,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, order_index=0):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.patch_size = patch_size
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rope = PointROPE(freq=rope_freq)

    def forward(self, point):
        _check_litept_deps()
        H = self.num_heads
        C = self.channels

        pad, unpad, cu_seqlens = point.get_padding_and_inverse(self.patch_size)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]
        pos = point.grid_coord[order].reshape(-1, 3).unsqueeze(0)

        q, k, v = qkv.half().chunk(3, dim=-1)
        q = q.reshape(-1, H, C // H).transpose(0, 1)[None]
        k = k.reshape(-1, H, C // H).transpose(0, 1)[None]

        q = self.rope(q.float(), pos).to(q.dtype)
        k = self.rope(k.float(), pos).to(k.dtype)

        qkv_rotated = torch.stack([
            q.squeeze(0).transpose(0, 1),
            k.squeeze(0).transpose(0, 1),
            v.reshape(-1, H, C // H)
        ], dim=1)

        feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv_rotated, cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.attn_drop if self.training else 0,
            softmax_scale=self.scale,
        ).reshape(-1, C)

        feat = feat.to(qkv.dtype)
        feat = feat[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class GridPooling(PointModule):
    def __init__(self, in_channels, out_channels, stride=2,
                 norm_layer=None, act_layer=None, reduce="max",
                 shuffle_orders=True, traceable=True,
                 re_serialization=False, serialization_order="z"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable
        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())
        self.re_serialization = re_serialization
        self.serialization_order = serialization_order

    def forward(self, point):
        _check_litept_deps()
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0], point.grid_size, rounding_mode="trunc"
            ).int()
        else:
            raise AssertionError("grid_coord or (coord, grid_size) required")
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48
        grid_coord, cluster, counts = torch.unique(
            grid_coord, sorted=True, return_inverse=True, return_counts=True, dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        point_dict = AddictDict(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        for key in ["origin_coord", "condition", "context", "name", "split", "color", "grid_size", "mask"]:
            if key in point.keys():
                if key == "grid_size":
                    point_dict[key] = point.grid_size * self.stride
                elif key == "mask":
                    point_dict[key] = torch_scatter.segment_csr(
                        point.mask[indices].float(), idx_ptr, reduce="mean"
                    ) > 0.5
                else:
                    point_dict[key] = torch_scatter.segment_csr(
                        point[key][indices], idx_ptr, reduce="mean"
                    )
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if hasattr(self, "norm"):
            point = self.norm(point)
        if hasattr(self, "act"):
            point = self.act(point)
        if self.re_serialization:
            point.serialization(order=self.serialization_order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        return point


class GridUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels,
                 norm_layer=None, act_layer=None, traceable=False):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())
        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat = point.feat

        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
            parent["unpooling_inverse"] = inverse
        return parent


class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(in_channels, embed_channels, kernel_size=5, padding=1, bias=False, indice_key="stem")
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point):
        return self.stem(point)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(self, channels, num_heads, patch_size=48, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 pre_norm=True, order_index=0, cpe_indice_key=None,
                 enable_conv=True, enable_attn=True, rope_freq=100.0):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.enable_conv = enable_conv
        self.enable_attn = enable_attn

        if self.enable_conv:
            self.conv = PointSequential(
                spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=cpe_indice_key),
                nn.Linear(channels, channels),
                norm_layer(channels),
            )
        else:
            self.norm0 = PointSequential(norm_layer(channels))

        if self.enable_attn:
            self.norm1 = PointSequential(norm_layer(channels))
            self.attn = PointROPEAttention(
                channels=channels, patch_size=patch_size, rope_freq=rope_freq,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=proj_drop, order_index=order_index,
            )
            self.norm2 = PointSequential(norm_layer(channels))
            self.mlp = PointSequential(MLP(
                in_channels=channels, hidden_channels=int(channels * mlp_ratio),
                out_channels=channels, act_layer=act_layer, drop=proj_drop,
            ))
            self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point):
        if self.enable_conv:
            shortcut = point.feat
            point = self.conv(point)
            point.feat = shortcut + point.feat
        else:
            point = self.norm0(point)

        if self.enable_attn:
            shortcut = point.feat
            if self.pre_norm:
                point = self.norm1(point)
            point = self.drop_path(self.attn(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm1(point)

            shortcut = point.feat
            if self.pre_norm:
                point = self.norm2(point)
            point = self.drop_path(self.mlp(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class LitePT(PointModule):
    """
    LitePT: Lighter Yet Stronger Point Transformer (CVPR 2026)

    U-Net encoder-decoder with serialization-based attention.
    Default config = LitePT-S (12.7M params).

    输入: data_dict 包含:
        - feat [N, in_channels]: 输入特征
        - grid_coord [N, 3]: 体素化坐标 (GridSampling 后)
        - offset [batch_size]: batch 分隔符
    输出: Point 对象 (包含 feat, coord 等)
    """

    def __init__(
        self,
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
    ):
        super().__init__()
        _check_litept_deps()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.enc_conv = enc_conv
        self.enc_attn = enc_attn
        self.dec_conv = dec_conv
        self.dec_attn = dec_attn

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels, embed_channels=enc_channels[0],
            norm_layer=bn_layer, act_layer=act_layer,
        )

        # Encoder
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]):sum(enc_depths[:s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1], out_channels=enc_channels[s],
                        stride=stride[s - 1], norm_layer=bn_layer, act_layer=act_layer,
                        re_serialization=enc_attn[s], serialization_order=self.order,
                    ), name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s], num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s], mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i], norm_layer=ln_layer,
                        act_layer=act_layer, pre_norm=pre_norm,
                        order_index=i % len(self.order), cpe_indice_key=f"stage{s}",
                        enable_conv=enc_conv[s], enable_attn=enc_attn[s],
                        rope_freq=enc_rope_freq[s],
                    ), name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # Decoder
        if not self.enc_mode:
            dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[sum(dec_depths[:s]):sum(dec_depths[:s + 1])]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1], skip_channels=enc_channels[s],
                        out_channels=dec_channels[s], norm_layer=bn_layer, act_layer=act_layer,
                    ), name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s], num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s], mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            attn_drop=attn_drop, proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i], norm_layer=ln_layer,
                            act_layer=act_layer, pre_norm=pre_norm,
                            order_index=i % len(self.order), cpe_indice_key=f"stage{s}",
                            enable_conv=dec_conv[s], enable_attn=dec_attn[s],
                            rope_freq=dec_rope_freq[s],
                        ), name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        """
        data_dict 至少包含:
        1. feat [N, in_channels]
        2. grid_coord [N, 3] 或 coord + grid_size
        3. offset [batch_size] 或 batch [N]
        """
        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point


# ============================================================
# 工厂函数
# ============================================================

def build_model(num_classes=3, in_ch=6, model_type="pointnet", **kwargs):
    """
    构建模型
    model_type: "pointnet" | "litept"
    """
    if model_type == "pointnet":
        return PointNetSeg(num_classes=num_classes, in_ch=in_ch, **kwargs)
    elif model_type == "litept":
        return LitePT(in_channels=in_ch, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
