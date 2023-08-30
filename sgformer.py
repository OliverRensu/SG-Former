import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from pos_embed import get_2d_sincos_pos_embed

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def local_conv(dim):
    return nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

class Attention(nn.Module):
    def __init__(self, dim, mask, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio=sr_ratio
        if sr_ratio>1:
            if mask:
                self.q = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
                if self.sr_ratio==8:
                    f1, f2, f3 = 14*14, 56, 28
                elif self.sr_ratio==4:
                    f1, f2, f3 = 49, 14, 7
                elif self.sr_ratio==2:
                    f1, f2, f3 = 2, 1, None
                self.f1 = nn.Linear(f1, 1)
                self.f2 = nn.Linear(f2, 1)
                if f3 is not None:
                    self.f3 = nn.Linear(f3, 1)
            else:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
                self.act = nn.GELU()

                self.q1 = nn.Linear(dim, dim//2, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.q2 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.lepe_linear = nn.Linear(dim, dim)
        self.lepe_conv = local_conv(dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask):
        B, N, C = x.shape
        lepe = self.lepe_conv(
            self.lepe_linear(x).transpose(1, 2).view(B, C, H, W)).view(B, C, -1).transpose(-1, -2)
        if self.sr_ratio > 1:
            if mask is None:
                # global
                q1 = self.q1(x).reshape(B, N, self.num_heads//2, C // self.num_heads).permute(0, 2, 1, 3)
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_1 = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_1 = self.act(self.norm(x_1))
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1] #B head N C

                attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale #B head Nq Nkv
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)

                global_mask_value = torch.mean(attn1.detach().mean(1), dim=1) # B Nk  #max ?  mean ?
                global_mask_value = F.interpolate(global_mask_value.view(B,1,H//self.sr_ratio,W//self.sr_ratio),
                                                  (H, W), mode='nearest')[:, 0]

                # local
                q2 = self.q2(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3) #B head N C
                kv2 = self.kv2(x_.reshape(B, C, -1).permute(0, 2, 1)).reshape(B, -1, 2, self.num_heads // 2,
                                                                          C // self.num_heads).permute(2, 0, 3, 1, 4)
                k2, v2 = kv2[0], kv2[1]
                q_window = 7
                window_size= 7
                q2, k2, v2 = window_partition(q2, q_window, H, W), window_partition(k2, window_size, H, W), \
                             window_partition(v2, window_size, H, W)
                attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
                # (B*numheads*num_windows, window_size*window_size, window_size*window_size)
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)

                x2 = (attn2 @ v2)  # B*numheads*num_windows, window_size*window_size, C   .transpose(1, 2).reshape(B, N, C)
                x2 = window_reverse(x2, q_window, H, W, self.num_heads // 2)

                local_mask_value = torch.mean(attn2.detach().view(B, self.num_heads//2, H//window_size*W//window_size, window_size*window_size, window_size*window_size).mean(1), dim=2)
                local_mask_value = local_mask_value.view(B, H // window_size, W // window_size, window_size, window_size)
                local_mask_value=local_mask_value.permute(0, 1, 3, 2, 4).contiguous().view(B, H, W)

                # mask B H W
                x = torch.cat([x1, x2], dim=-1)
                x = self.proj(x+lepe)
                x = self.proj_drop(x)
                # cal mask
                mask = local_mask_value+global_mask_value
                mask_1 = mask.view(B, H * W)
                mask_2 = mask.permute(0, 2, 1).reshape(B, H * W)
                mask = [mask_1, mask_2]
            else:
                q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                # mask [local_mask global_mask]  local_mask [value index]  value [B, H, W]
                # use mask to fuse
                mask_1, mask_2 = mask
                mask_sort1, mask_sort_index1 = torch.sort(mask_1, dim=1)
                mask_sort2, mask_sort_index2 = torch.sort(mask_2, dim=1)
                if self.sr_ratio == 8:
                    token1, token2, token3 = H * W // (14 * 14), H * W // 56, H * W // 28
                    token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                elif self.sr_ratio == 4:
                    token1, token2, token3 = H * W // 49, H * W // 14, H * W // 7
                    token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                elif self.sr_ratio == 2:
                    token1, token2 = H * W // 2, H * W // 1
                    token1, token2 = token1 // 2, token2 // 2
                if self.sr_ratio==4 or self.sr_ratio==8:
                    p1 = torch.gather(x, 1, mask_sort_index1[:, :H * W // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2 = torch.gather(x, 1, mask_sort_index1[:, H * W // 4:H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3 = torch.gather(x, 1, mask_sort_index1[:, H * W // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)], dim=-1).permute(0,2,1)  # B N C

                    x_ = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H * W, C)
                    p1_ = torch.gather(x_, 1, mask_sort_index2[:, :H * W // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 4:H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3_.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)], dim=-1).permute(0,2,1)  # B N C
                elif self.sr_ratio==2:
                    p1 = torch.gather(x, 1, mask_sort_index1[:, :H * W // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2 = torch.gather(x, 1, mask_sort_index1[:, H * W // 2:].unsqueeze(-1).repeat(1, 1, C))
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)], dim=-1).permute(0, 2, 1)  # B N C

                    x_ = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B, H * W, C)
                    p1_ = torch.gather(x_, 1, mask_sort_index2[:, :H * W // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 2:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)], dim=-1).permute(0, 2, 1)  # B N C

                kv1 = self.kv1(seq1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4) # kv B heads N C
                kv2 = self.kv2(seq2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv = torch.cat([kv1, kv2], dim=2)
                k, v = kv[0], kv[1]
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x+lepe)
                x = self.proj_drop(x)
                mask=None

        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x+lepe)
            x = self.proj_drop(x)
            mask=None

        return x, mask

def window_partition(x, window_size, H, W):
    B, num_heads, N, C = x.shape
    x = x.contiguous().view(B*num_heads, N, C).contiguous().view(B*num_heads, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C).\
        view(-1, window_size*window_size, C)
    return windows  #(B*numheads*num_windows, window_size, window_size, C)


def window_reverse(windows, window_size, H, W, head):
    Bhead = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(Bhead, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(Bhead, H, W, -1).view(Bhead//head, head, H, W, -1)\
        .contiguous().permute(0,2,3,1,4).contiguous().view(Bhead//head, H, W, -1).view(Bhead//head, H*W, -1)
    return x #(B, H, W, C)

class Block(nn.Module):

    def __init__(self, dim, mask, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, mask,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask):
        x_, mask = self.attn(self.norm1(x), H, W, mask)
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x, mask


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.GroupNorm(1, b)#torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    # @torch.no_grad()
    # def fuse(self):
    #     c, bn = self._modules.values()
    #     w = bn.weight / (bn.running_var + bn.eps)**0.5
    #     w = c.weight * w[:, None, None, None]
    #     b = bn.bias - bn.running_mean * bn.weight / \
    #         (bn.running_var + bn.eps)**0.5
    #     m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
    #         0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
    #     m.weight.data.copy_(w)
    #     m.bias.data.copy_(b)
    #     return m

class Head(nn.Module):
    def __init__(self, n):
        super(Head, self).__init__()
        self.conv = nn.Sequential(
            Conv2d_BN(3, n, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(n, n, 3, 1, 1),
            nn.GELU(),
            Conv2d_BN(n, n, 3, 2, 1),
        )
        self.norm = nn.LayerNorm(n)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H,W

class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()

        self.dim = dim
        self.out_dim = out_dim
        self.act = nn.GELU()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        # x B C H W
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W

class SGFormer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.num_patches = img_size//4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Head(embed_dims[0])  #
            else:
                patch_embed = PatchMerging(dim=embed_dims[i - 1],
                                           out_dim=embed_dims[i])
            block = nn.ModuleList([Block(
                dim=embed_dims[i], mask=True if (j%2==1 and i<num_stages-1) else False, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches*self.num_patches, embed_dims[0]))  # fixed sin-cos embedding

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches,
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        mask=None
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            if i==0:
                x+=self.pos_embed
            for blk in block:
                x, mask = blk(x, H, W, mask)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


@register_model
def sgformer_s(pretrained=False, **kwargs):
    model = SGFormer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 16, 1], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def sgformer_m(pretrained=False, **kwargs):
    model = SGFormer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 6, 28, 2], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def sgformer_b(pretrained=False, **kwargs):
    model = SGFormer(
        patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[4, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[4, 6, 24, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


