import sys
from subprocess import check_call
from dataclasses import dataclass
from typing import List
from functools import cached_property, partial
import concurrent.futures

import math
import numpy as np
import numba as nb
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch

import platform
from ctypes import Structure, CDLL, POINTER, c_int, c_double, c_char, sizeof, cast, byref

import scipy

EPS = 1e-10
T = 5
START_NODE = None

# HGS_TW_LIBRARY_FILEPATH = get_hgs_vrptw()
# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, units=32, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr, return_node_emb: bool = False):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))

        if return_node_emb:
            return w, x # edge_emb(E,d), node_emb(N,d)
        return w


# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn, output_space: str = "probs"):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.output_space = output_space
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                if self.output_space == "probs":
                    x = torch.sigmoid(x)  # last layer
        return x


# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu', output_space: str = "probs"):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn, output_space=output_space)

    def forward(self, x):
        return super().forward(x).squeeze(dim=-1)

class PomoEncoderLayer(nn.Module):
    """
    입력/출력: (B, N, D)
    """
    def __init__(self, embed_dim: int, n_heads: int, ff_hidden: int = 512):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.InstanceNorm1d(embed_dim, affine=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm2 = nn.InstanceNorm1d(embed_dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        """
        # self-attention
        attn_out, _ = self.mha(x, x, x)          # (B, N, D)
        x = x + attn_out                         # residual
        x_perm = x.permute(0, 2, 1)              # (B, D, N)
        x = self.norm1(x_perm).permute(0, 2, 1).contiguous()  # (B, N, D)

        # feed-forward
        ff_out = self.ff(x)                      # (B, N, D)
        x = x + ff_out                           # residual
        x_perm = x.permute(0, 2, 1)
        x = self.norm2(x_perm).permute(0, 2, 1).contiguous()
        return x
    
class PomoTransformerEncoder(nn.Module):
    """
    - 입력:
        (N, node_dim)   : 단일 그래프
        (B, N, node_dim): 배치 그래프 (모든 그래프 N 동일)
    - 출력:
        (N, D) 또는 (B, N, D)
    """
    def __init__(
        self,
        node_dim: int,
        embed_dim: int,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_hidden: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, embed_dim)
        self.layers = nn.ModuleList(
            [PomoEncoderLayer(embed_dim, n_heads, ff_hidden) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, node_dim) or (B, N, node_dim)
        """
        if x.dim() == 2:
            # 단일 그래프
            h = self.input_proj(x).unsqueeze(0)  # (1, N, D)
            for layer in self.layers:
                h = layer(h)                     # (1, N, D)
            return h.squeeze(0)                  # (N, D)

        elif x.dim() == 3:
            # 배치 그래프
            h = self.input_proj(x)               # (B, N, D)
            for layer in self.layers:
                h = layer(h)                     # (B, N, D)
            return h                             # (B, N, D)

        else:
            raise ValueError(f"Expected x.ndim in {{2, 3}}, got {x.dim()}")

    

class Net(nn.Module):
    def __init__(
            self,
            gfn=False,
            Z_out_dim=1,
            start_node=None,
            node_feature_dim=None,
            embedding_dim=32,
            encoder_type='gnn',
            matrix_output_space: str = "probs",
            ):
        super().__init__()
        # Shape notation:
        #   B: batch size, N: nodes per graph
        #   E: edges per graph (e.g., directed complete graph -> N*(N-1); undirected -> N*(N-1)/2)
        #   sum_E: total edges over the batch = B * E (when all graphs have same N)
        #   V: nodes per graph (=N), sum_V: total nodes over the batch = B * N
        #   D: embedding_dim
        if node_feature_dim is None:
            node_feature_dim = 1 if start_node is not None else 2
        
        self.encoder_type = encoder_type
    
        if self.encoder_type == 'gnn':
            self.emb_net = EmbNet(feats=node_feature_dim, units=embedding_dim)
            self.par_net_heu = ParNet(units=embedding_dim, output_space=matrix_output_space)
            self.trans_encoder = None
            self.edge_mlp = None
        else:  # transformer
            self.emb_net = None
            self.trans_encoder = PomoTransformerEncoder(
                node_dim=node_feature_dim,
                embed_dim=embedding_dim
            )
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.par_net_heu = ParNet(units=embedding_dim, output_space=matrix_output_space)

        self.gfn = gfn
        self.Z_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, Z_out_dim),
        ) if gfn else None

    def forward(self, pyg, return_logZ: bool = False, return_node_emb: bool = False):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr

        # ------------------------------------------------------------------
        # 1) GNN encoder: 기존 EmbNet 경로 (동작 동일)
        # ------------------------------------------------------------------
        if self.encoder_type == "gnn":
            emb_out = self.emb_net(x, edge_index, edge_attr, return_node_emb=return_node_emb)
            if return_node_emb:
                emb, node_emb = emb_out  # edge_emb(sum_E,D), node_emb(sum_V,D) when Batch; (E,D)/(N,D) for single graph
            else:
                emb, node_emb = emb_out, None

        # ------------------------------------------------------------------
        # 2) Transformer encoder: POMO-style self-attention over nodes
        # ------------------------------------------------------------------
        elif self.encoder_type == "transformer":
            # 노드 임베딩 계산
            if isinstance(pyg, Batch):
                ptr = pyg.ptr                       # (B+1,)
                B = pyg.num_graphs
                n_nodes = ptr[1:] - ptr[:-1]
                N = int(n_nodes[0].item())      # 모든 그래프 노드수 동일하다고 가정
                
                # (sum_V, feats) -> (B, N, feats)
                x_batched = x.view(B, N, -1)       # (B, N, feats)
                node_emb_batched = self.trans_encoder(x_batched)  # (B, N, D)

                # edge_index는 flatten된 노드 인덱스 기준이므로 다시 (sum_V, D)로 펴준다
                node_emb = node_emb_batched.view(-1, node_emb_batched.size(-1))  # (sum_V, D)                
            else:
                node_emb = self.trans_encoder(x)            # (N, D)

            # edge 임베딩: (h_i, h_j) concat 후 MLP
            src, dst = edge_index # (E,)
            h_src = node_emb[src]                          # (E, D)
            h_dst = node_emb[dst]                          # (E, D)
            edge_pair = torch.cat([h_src, h_dst], dim=-1)  # (E, 2D)
            emb = self.edge_mlp(edge_pair)                 # (E, D)

        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        # edge 단위 heuristic 스칼라 (E,)
        heu = self.par_net_heu(emb)

        # logZ (GFN용): emb(=edge_emb 또는 node_emb 기반) 평균 사용 – 기존 로직 유지
        if return_logZ:
            assert self.gfn and self.Z_net is not None
            logZ_edge = self.Z_net(emb).squeeze(-1) # (E,)
            if hasattr(pyg, "batch"):                             # Batch 입력이면
                edge_batch = pyg.batch[edge_index[0]]             # (E,)
                # 그래프별로 mean
                logZ = gnn.global_mean_pool(logZ_edge.unsqueeze(-1), edge_batch).squeeze(-1)  # (B,)
            else:                                                 # 단일 그래프면 기존처럼
                logZ = logZ_edge.mean(0)                          # scalar            if return_node_emb:
                return heu, logZ, node_emb
            return heu, logZ

        if return_node_emb:
            return heu, node_emb
        return heu

    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False

    @staticmethod
    def reshape(pyg, vector):
        """Turn phe/heu vector into matrix with zero padding"""
        device = pyg.x.device

        # Batch
        if isinstance(pyg, Batch):
            num_graphs = pyg.num_graphs
            ptr = pyg.ptr  # shape: (B+1,)

            # 모든 그래프의 노드수가 동일하므로 최대값 사용
            problem_size = int(torch.diff(ptr).max().item())
            matrix = torch.zeros((num_graphs, problem_size, problem_size), device=device)

            # 각 edge가 어느 그래프에 속하는지
            edge_batch = pyg.batch[pyg.edge_index[0]]
            # 그래프 내부 로컬 인덱스로 변환
            local_src = pyg.edge_index[0] - ptr[edge_batch]
            local_dst = pyg.edge_index[1] - ptr[edge_batch]
            matrix[edge_batch, local_src, local_dst] = vector
            return matrix

        # 단일 그래프
        n_nodes = pyg.x.shape[0]
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix
    
    @staticmethod
    def reshape_nodes(pyg, node_vec):
        # node_vec: (sum_V, D)
        device = pyg.x.device
        D = node_vec.size(-1)
        if isinstance(pyg, Batch):
            B = pyg.num_graphs
            ptr = pyg.ptr              # (B+1,)
            n_nodes = ptr[1:] - ptr[:-1]
            max_n = int(n_nodes.max())
            out = torch.zeros(B, max_n, D, device=device)

            batch_id = pyg.batch       # (sum_V,)
            local_idx = torch.arange(node_vec.size(0), device=device) - ptr[batch_id]
            out[batch_id, local_idx] = node_vec
            return out                 # (B, N, D)
        else:
            return node_vec.unsqueeze(0)   # (1, N, D)
        

class PomoLiteDecoder(nn.Module):
    def __init__(
        self,
        node_dim: int = 32,
        dyn_dim: int = 1,         # dynamic feature. ex) 2: [t_cur, traveled], 1: [t_cur]
        att_dim: int = 32,        # attention/logit 차원 (head_num * head_dim)
        tanh_clipping: float = 10.0,
        num_heads: int = 8,       # multi-head 개수
    ):
        super().__init__()
        self.tanh_clipping = tanh_clipping
        self.num_heads = num_heads
        self.dyn_dim = dyn_dim

        assert (
            att_dim % num_heads == 0
        ), f"att_dim({att_dim}) must be divisible by num_heads({num_heads})"
        self.head_dim = att_dim // num_heads

        self.Wq_1 = nn.Linear(node_dim, att_dim, bias=False)
        self.Wq_2 = nn.Linear(node_dim, att_dim, bias=False)
        self.Wq_last = nn.Linear(node_dim + dyn_dim, att_dim, bias=False)

        self.Wk = nn.Linear(node_dim, att_dim, bias=False)
        self.Wv = nn.Linear(node_dim, att_dim, bias=False)

        self.multi_head_combine = nn.Linear(att_dim, node_dim, bias=False)

        self.k = None
        self.v = None
        self.single_head_key = None
        self.q1 = None
        self.q2 = None

    def _reshape_by_heads(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, n, _ = qkv.size()
        q_reshaped = qkv.reshape(batch_size, n, self.num_heads, self.head_dim)
        return q_reshaped.transpose(1, 2)

    def _multi_head_attention(self, q, k, v, ninf_mask=None):
        # q: (B, H, G, head_dim), k/v: (B, H, N, head_dim), mask: (B, G, N) -> broadcast
        scores = torch.matmul(q, k.transpose(2, 3))  # (B, H, G, N)
        scores = scores / math.sqrt(self.head_dim)
        if ninf_mask is not None:
            scores = scores + ninf_mask[:, None, :, :]
        weights = F.softmax(scores, dim=3)
        out = torch.matmul(weights, v)  # (B, H, G, head_dim)
        out = out.transpose(1, 2).contiguous().view(q.size(0), q.size(2), -1)  # (B, G, att_dim)
        return out

    def set_kv(self, node_emb: torch.Tensor):
        # node_emb: (B, N, D)
        self.k = self._reshape_by_heads(self.Wk(node_emb))
        self.v = self._reshape_by_heads(self.Wv(node_emb))
        self.single_head_key = node_emb.transpose(1, 2).contiguous()  # (B, D, N)

    def set_q1(self, node_emb: torch.Tensor, first_idx: torch.Tensor):
        # first_idx: (B, G)
        B, G = first_idx.size()
        D = node_emb.size(-1)
        first = node_emb.gather(1, first_idx.unsqueeze(-1).expand(-1, -1, D))
        self.q1 = self._reshape_by_heads(self.Wq_1(first))

    def set_q2(self, node_emb: torch.Tensor, group_size: int):
        # graph-level context expanded over groups
        graph_ctx = node_emb.mean(dim=1, keepdim=True).expand(-1, group_size, -1)
        self.q2 = self._reshape_by_heads(self.Wq_2(graph_ctx))

    def reset_cache(self):
        self.k = None
        self.v = None
        self.single_head_key = None
        self.q1 = None
        self.q2 = None

    def forward(self, node_emb, cur_idx, first_idx, dyn_feat, mask=None):
        """
        node_emb : (B, N, D)
        cur_idx  : (B, G)
        first_idx: (B, G)
        dyn_feat : (B, G, dyn_dim)
        mask     : (B, G, N)  # 1 = valid, 0 = invalid
        return   : log_probs (B, G, N) with log softmax
        """
        B, N, D = node_emb.size()
        _, G = cur_idx.size()
        assert first_idx.shape == cur_idx.shape

        # ensure cached projections are up-to-date
        if self.k is None or self.v is None or self.single_head_key is None or self.k.size(2) != N or self.k.size(0) != B:
            self.set_kv(node_emb)
        if self.q1 is None or self.q1.size(0) != B or self.q1.size(2) != G:
            self.set_q1(node_emb, first_idx)
        if self.q2 is None or self.q2.size(0) != B or self.q2.size(2) != G:
            self.set_q2(node_emb, G)

        # gather node embeddings for current step
        h_cur = node_emb.gather(1, cur_idx.unsqueeze(-1).expand(-1, -1, D))     # (B, G, D)

        # build queries
        input_cat = torch.cat([h_cur, dyn_feat], dim=-1)  # (B, G, D+dyn_dim)
        q_last = self._reshape_by_heads(self.Wq_last(input_cat))
        q = q_last
        if self.q1 is not None:
            q = q + self.q1
        if self.q2 is not None:
            q = q + self.q2

        ninf_mask = None
        if mask is not None:
            ninf_mask = torch.where(mask > 0, 0.0, float("-inf"))

        mh_atten_out = self._multi_head_attention(q, self.k, self.v, ninf_mask)  # (B, G, att_dim)
        mh_atten_out = self.multi_head_combine(mh_atten_out)  # (B, G, D)

        score = torch.matmul(mh_atten_out, self.single_head_key)  # (B, G, N)
        score_scaled = score / math.sqrt(D)
        if self.tanh_clipping is not None:
            score_scaled = self.tanh_clipping * torch.tanh(score_scaled)
        if ninf_mask is not None:
            score_scaled = score_scaled + ninf_mask

        log_probs = F.log_softmax(score_scaled, dim=2)
        return log_probs

    
class Net_D(nn.Module):
    def __init__(self, gfn=False, Z_out_dim=1, matrix_output_space: str = "probs"):
        super().__init__()
        self.emb_net = EmbNet()
        self.par_net_heu = ParNet(output_space=matrix_output_space)

        self.gfn = gfn
        self.Z_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, Z_out_dim),
        ) if gfn else None

    def forward(self, pyg, return_logZ=False):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        emb = self.emb_net(x, edge_index, edge_attr)
        heu = self.par_net_heu(emb)

        if return_logZ:
            assert self.gfn and self.Z_net is not None
            logZ = self.Z_net(emb).mean(0)
            return heu, logZ

        return heu

    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False

    @staticmethod
    def reshape(pyg, vector):
        """Turn phe/heu vector into matrix with zero padding"""
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix


# ============================================================================================================================

class search():
    def __init__(
            self,
            distances,
            generate=20,
            alpha=1,
            beta=1,
            heuristic=None,
            heuristic_target=None,
            two_opt=False,
            device='cuda:0',
            local_search: str | None = 'nls',
            # tw
            tw_start=None,
            tw_end=None,
            service_time=None,
            enforce_time_windows: bool = True,
            depot: int=0,
            decoder=None,
            node_emb=None,
            tw_mask: bool = False,
            dyn_scale: float = 1.0,
            use_pomo: bool = False,
            heuristic_output_space: str = "probs",
            nls_T_nls: int = 5,
            nls_T_p: int = 20,
    ):

        distances = distances.to(device)
        self.device = device
        self.tw_mask = tw_mask
        self.use_pomo = use_pomo
        self.heuristic_output_space = heuristic_output_space

        # 단일 인스턴스: (N, N) / 배치 인스턴스: (B, N, N)
        if distances.dim() == 2:
            self.batch_mode = False
            self.batch_size = 1
            self.scale_ = distances.size(0)
        elif distances.dim() == 3:
            self.batch_mode = True
            self.batch_size = distances.size(0)
            self.scale_ = distances.size(1)
        else:
            raise ValueError(f"distances must be 2D or 3D, got shape {tuple(distances.shape)}")

        self.distances = distances
        self.generate = generate
        self.alpha = alpha
        self.beta = beta
        assert local_search in [None, "2opt", "nls"]
        self.local_search_type = "2opt" if two_opt else local_search

        base_heuristic = 1 / (distances + 1e-10) if heuristic is None else heuristic.to(device)
        base_heuristic_target = (
            1 / (distances + 1e-10) if heuristic_target is None else heuristic_target.to(device)
        )
        self.heuristic = base_heuristic
        self.heuristic_target = base_heuristic_target
        self.shortest_path = None
        self.lowest_cost = float("inf")

        # TW 정보
        self.tw_start = None if tw_start is None else tw_start.to(device)
        self.tw_end = None if tw_end is None else tw_end.to(device)
        self.service_time = None if service_time is None else service_time.to(device)
        self.enforce_time_windows = enforce_time_windows
        self.depot = depot
        self.eps = EPS
        self.debug_tw = False  # retained for backward compatibility, unused

        self.decoder = decoder
        self.node_emb = node_emb
        self.dyn_scale = dyn_scale
        self.nls_T_nls = int(nls_T_nls)
        self.nls_T_p = int(nls_T_p)

    def get_backward_log_probs(self, paths, alpha=1.0, heuristic_back=None):
        """
        Compute per-step log P_B(s_t | s_{t+1}) along given 'paths'.

        paths: Tensor [N, G]  (N = #nodes, G = #generated tours)
        returns: Tensor [N-1, G]  (per-step log-prob; sum(0) -> per-tour sum)
        """
        if heuristic_back is None:
            heuristic_back = self.heuristic  # use same edge scores by default

        N = paths.shape[0]             # number of nodes per tour
        G = paths.shape[1]             # number of tours
        device = self.device
        idx = torch.arange(G, device=device)

        # visited set at step t (before adding current node at t)
        visited = torch.zeros(G, self.scale_, device=device, dtype=torch.bool)
        start = paths[0]               # [G]
        visited[idx, start] = True

        log_probs = []
        for t in range(1, N):
            cur = paths[t]             # [G], current node j
            prev = paths[t-1]          # [G], actual parent i*

            # scores for candidate parents i in visited set S_{t-1}:
            # gather column 'cur[g]' from heuristic_back for all g in batch
            # shape: (N, G) -> transpose -> (G, N) aligned to visited mask
            scores = (heuristic_back[:, cur] ** alpha).transpose(0, 1)  # [G, N]

            # mask to visited parents only
            scores = scores * visited.float()

            # normalize over parents; fallback to uniform on visited if sum==0
            sums = scores.sum(dim=1, keepdim=True)
            fallback = (sums.squeeze(-1) <= 0)
            # uniform over visited
            uniform = visited.float() / (visited.float().sum(dim=1, keepdim=True) + 1e-12)
            probs = torch.where(fallback.unsqueeze(1), uniform, scores / (sums + 1e-12))

            # log-prob of choosing the actual parent 'prev'
            log_p = torch.log(probs[idx, prev] + 1e-12)  # [G]
            log_probs.append(log_p)

            # update visited set with current node
            visited[idx, cur] = True

        return torch.stack(log_probs, dim=0)  # [N-1, G]


    def get_route(self, alpha=1.0, require_prob=False, paths=None, start_node=None, desi=1):
        """
        Generate routes.

        - distances: (N, N)
            * paths:     (N, G)
            * log_probs: (N-1, G)
        - distances: (B, N, N)
            * paths:     (N, B, G)
            * log_probs: (N-1, B, G)
        """
        if self.distances.dim() == 2:
            return self._get_route_single(alpha, require_prob, paths, start_node, desi)
        else:
            return self._get_route_batched(alpha, require_prob, paths, start_node, desi)

    def _get_route_single(self, alpha=1.0, require_prob=False, paths=None, start_node=None, desi=1):
        """
        distances.shape == (N, N)
        paths:
            - forward  : None       -> 샘플링
            - backward : (N, G)     -> 주어진 경로의 log-prob만 계산
        """
        assert self.distances.dim() == 2, "single route generation requires 2D distances"
        N = self.distances.shape[0]
        G = self.generate
        device = self.device

        use_ar = (self.decoder is not None) and (self.node_emb is not None)

        # --------------------
        # start node 설정
        # --------------------
        if paths is None:
            if start_node is None:
                # (G,)
                start = torch.randint(low=0, high=N, size=(G,), device=device)
            else:
                if torch.is_tensor(start_node):
                    start = start_node.to(device)
                    if start.dim() == 0:
                        start = start.expand(G)
                else:
                    start = torch.full((G,), start_node, dtype=torch.long, device=device)
        else:
            # paths: (N, G) 가정
            start = paths[0]  # (G,)

        # 방문 마스크: (G, N)
        mask = torch.ones((G, N), device=device)

        # 기존 NAR용 heuristic prob_mat (AR일 때는 실제 sampling에는 사용 안 함)
        if self.heuristic_output_space == "logit":
            prob_mat = self.heuristic if paths is None else self.heuristic_target
        elif paths is None:
            prob_mat = (torch.ones_like(self.distances, device=device) ** self.alpha) * (
                self.heuristic ** self.beta
            )
        else:
            prob_mat = (torch.ones_like(self.distances, device=device) ** self.alpha) * (
                self.heuristic_target ** self.beta
            )

        # 시작 노드 방문 처리
        mask[torch.arange(G, device=device), start] = 0.0
        # depot(0) 재방문 금지
        mask[:, self.depot] = 0.0

        # --------------------
        # 초기 시간 / 누적 이동 거리
        # --------------------
        if self.tw_start is not None and self.service_time is not None:
            # 단일 인스턴스: tw_start, service_time은 (N,) 가정
            t0 = float(
                max(self.tw_start[self.depot].item(), 0.0)
                + self.service_time[self.depot].item()
            )
            t_cur = torch.full((G,), t0, device=device)  # (G,)
        else:
            t_cur = torch.zeros((G,), device=device)

        # 순수 이동 거리 누적 (시간이 아니라 거리 기준)
        traveled = torch.zeros((G,), device=device)  # (G,)

        decoder_node_emb = None
        if use_ar:
            decoder_node_emb = self.node_emb
            if decoder_node_emb.dim() == 2:
                decoder_node_emb = decoder_node_emb.unsqueeze(0)  # (1, N, D)
            self.decoder.reset_cache()
            self.decoder.set_kv(decoder_node_emb)
            self.decoder.set_q1(decoder_node_emb, start.unsqueeze(0))
            self.decoder.set_q2(decoder_node_emb, G)

        prev = start                      # (G,)
        paths_list = [start]              # 각 원소 (G,)
        log_probs_list = []

        for step in range(N - 1):

            # --------------------
            # TW feasibility mask
            # --------------------
            if self.enforce_time_windows and self.tw_start is not None:
                # distances[prev] : (G, N)
                travel_all = self.distances[prev]                 # (G, N)
                arrival = torch.maximum(
                    t_cur[:, None] + travel_all,                  # (G, N)
                    self.tw_start[None, :],                       # (1, N)
                )
                invalid_tw = arrival > (self.tw_end[None, :] + self.eps)

                finish = arrival + self.service_time[None, :]     # (G, N)
                back = self.distances[:, self.depot][None, :]     # (1, N)
                invalid_return = finish + back > (
                    self.tw_end[self.depot] + self.eps
                )

                tw_mask = (~(invalid_tw | invalid_return)).float()    # (G, N)
            else:
                tw_mask = 1.0

            # Sampling mask: only exclude visited nodes (ignore TW feasibility for sampling)
            if self.tw_mask=="on":
                combined_mask = mask * tw_mask                     # (G, N)
            else:
                combined_mask = mask                                     # (G, N)
            row_sum = combined_mask.sum(dim=-1, keepdim=True)        # (G, 1)
            fallback = (row_sum == 0).float()
            combined_mask = combined_mask + fallback * mask          # 최소 하나는 valid

            # --------------------
            # 확률 분포 계산 (NAR vs AR)
            # --------------------
            if not use_ar:
                # NAR: 기존 heuristic 사용
                dist_rows = prob_mat[prev]                           # (G, N)
                if self.heuristic_output_space == "logit":
                    denom = self.beta if self.beta != 0 else 1.0
                    logits = dist_rows / denom
                    logits = logits.masked_fill(combined_mask <= 0, float("-inf"))
                    log_probs_step = F.log_softmax(logits, dim=-1) if require_prob else None
                else:
                    dist = (dist_rows ** alpha) * combined_mask          # (G, N)
                    row_sum = dist.sum(dim=-1, keepdim=True)             # (G, 1)
                    zero_rows = row_sum <= 0
                    safe_row_sum = torch.where(zero_rows,
                                            torch.ones_like(row_sum),
                                            row_sum)
                    probs = dist / (safe_row_sum + 1e-12)                # (G, N)
                    if zero_rows.any():
                        uniform = mask / (mask.sum(dim=-1, keepdim=True) + 1e-12)
                        probs = torch.where(zero_rows, uniform, probs)
                    probs_flat = probs                                   # (G, N)
                    log_probs_step = None
            else:
                # AR 디코더: dyn_feat = [현재 시간, 누적 이동 거리]
                dyn_t = t_cur        # (G,)
                dyn_d = traveled     # (G,)
                # dyn_feat = torch.stack([dyn_t, dyn_d], dim=-1) * self.dyn_scale   # (G, 2)
                dyn_feat = torch.stack([dyn_t], dim=-1) * self.dyn_scale   # (G, 1)
                dyn_feat = dyn_feat.unsqueeze(0)                 # (1, G, 2)

                cur_idx = prev.unsqueeze(0)                      # (1, G)
                first_idx = start.unsqueeze(0)                   # (1, G)
                mask_ar = combined_mask.unsqueeze(0)             # (1, G, N)

                log_probs_step = self.decoder(
                    node_emb=decoder_node_emb,       # (1, N, D)
                    cur_idx=cur_idx,         # (1, G)
                    first_idx=first_idx,     # (1, G)
                    dyn_feat=dyn_feat,       # (1, G, dyn_dim)
                    mask=mask_ar,            # (1, G, N)
                )                             # -> (1, G, N), log-softmax

                probs = log_probs_step.exp()[0]                  # (G, N)
                probs_flat = probs                               # (G, N)

            # --------------------
            # action 선택
            # --------------------
            if paths is not None:
                # paths: (N, G)
                actions = paths[step + 1]        # (G,)
                actions_flat = actions
            elif self.use_pomo and step == 0 and paths is None:
                avail = max(N - 1, 1)
                forced = (torch.arange(G, device=device) % avail) + 1
                actions = forced
                actions_flat = actions
                if require_prob:
                    log_probs_list.append(torch.zeros_like(actions, dtype=torch.float, device=device))
            else:
                if not use_ar and self.heuristic_output_space == "logit":
                    dist_cat = Categorical(logits=logits)
                else:
                    dist_cat = Categorical(probs=probs_flat)
                epsilon = 0.05
                if desi == 1:
                    random_mask = torch.rand(G, device=device) < epsilon
                    sampled = dist_cat.sample()                  # (G,)
                    if not use_ar and self.heuristic_output_space == "logit":
                        greedy = logits.argmax(dim=-1)           # (G,)
                    else:
                        greedy = probs_flat.argmax(dim=-1)       # (G,)
                    actions_flat = torch.where(random_mask, sampled, greedy)
                else:
                    actions_flat = dist_cat.sample()
                actions = actions_flat                           # (G,)

            # --------------------
            # log-prob 기록
            # --------------------
            if require_prob:
                if not use_ar:
                    if self.heuristic_output_space == "logit":
                        logp = log_probs_step.gather(
                            1, actions_flat.view(-1, 1)
                        ).squeeze(-1)                            # (G,)
                    else:
                        probs_act = probs_flat.gather(
                            1, actions_flat.view(-1, 1)
                        ).clamp_min(1e-12)                       # (G, 1)
                        logp = probs_act.log().squeeze(-1)       # (G,)
                else:
                    # log_probs_step : (1, G, N)
                    logp = log_probs_step[0].gather(
                        1, actions.view(-1, 1)
                    ).squeeze(-1)                                # (G,)
                log_probs_list.append(logp)

            paths_list.append(actions)

            # --------------------
            # 시간 / 거리 상태 업데이트
            # --------------------
            step_travel = self.distances[prev, actions]          # (G,)
            traveled = traveled + step_travel                    # 누적 이동 거리

            if self.enforce_time_windows and self.tw_start is not None:
                arr = torch.maximum(
                    t_cur + step_travel,
                    self.tw_start[actions],
                )
                t_cur = arr + self.service_time[actions]

            prev = actions
            mask = mask.clone()
            mask[torch.arange(G, device=device), actions] = 0.0

        paths_tensor = torch.stack(paths_list, dim=0)            # (N, G)
        if require_prob:
            log_probs_tensor = torch.stack(log_probs_list, dim=0)  # (N-1, G)
            return paths_tensor, log_probs_tensor
        return paths_tensor


    def _get_route_batched(self, alpha=1.0, require_prob=False, paths=None, start_node=None, desi=1):
        """
        distances.shape == (B, N, N)
        paths:
            - forward  : None           -> 샘플링
            - backward : (N, B, G)      -> 주어진 경로의 log-prob만 계산
        """
        assert self.distances.dim() == 3, "batched route generation requires 3D distances"
        B, N, _ = self.distances.shape
        G = self.generate
        device = self.device

        use_ar = (self.decoder is not None) and (self.node_emb is not None)

        # --------------------
        # start node 설정
        # --------------------
        if paths is None:
            if start_node is None:
                start = torch.randint(low=0, high=N, size=(B, G), device=device)
            else:
                if torch.is_tensor(start_node):
                    start = start_node.to(device)
                    if start.dim() == 1:
                        start = start[:, None].expand(-1, G)
                else:
                    start = torch.full((B, G), start_node, dtype=torch.long, device=device)
        else:
            # paths: (N, B, G) 또는 (N, G) (B==1) 가정
            if paths.dim() == 3:
                start = paths[0]
            elif paths.dim() == 2:
                start = paths[0].unsqueeze(0).expand(B, -1)
            else:
                raise ValueError(f"Unexpected paths shape {tuple(paths.shape)}")

        # 방문 마스크
        mask = torch.ones((B, G, N), device=device)

        # 기존 NAR용 heuristic prob_mat (AR 사용 시에는 직접 쓰진 않음)
        if self.heuristic_output_space == "logit":
            prob_mat = self.heuristic if paths is None else self.heuristic_target
        elif paths is None:
            prob_mat = (torch.ones_like(self.distances, device=device) ** self.alpha) * (
                self.heuristic ** self.beta
            )
        # !TODO: AR일때 decoder logic작업 필요(현재 ar에서는 d 안씀)
        else:
            prob_mat = (torch.ones_like(self.distances, device=device) ** self.alpha) * (
                self.heuristic_target ** self.beta
            )

        # 시작 노드 방문 처리
        mask.scatter_(-1, start.unsqueeze(-1), 0.0)
        # depot(0) 재방문 금지
        mask[:, :, self.depot] = 0.0

        # --------------------
        # 초기 시간 / 누적 이동 거리
        # --------------------
        if self.tw_start is not None and self.service_time is not None:
            tw_start_0 = torch.clamp(self.tw_start[:, self.depot], min=0.0)
            service_0 = self.service_time[:, self.depot]
            t0 = (tw_start_0 + service_0).to(device)
            t_cur = t0[:, None].expand(B, G).clone()  # (B, G)
        else:
            t_cur = torch.zeros((B, G), device=device)

        # ★ 순수 이동 거리 누적 (시간이 아니라 거리 기준)
        traveled = torch.zeros((B, G), device=device)        # (B, G)

        decoder_node_emb = None
        if use_ar:
            decoder_node_emb = self.node_emb
            self.decoder.reset_cache()
            self.decoder.set_kv(decoder_node_emb)
            self.decoder.set_q1(decoder_node_emb, start)
            self.decoder.set_q2(decoder_node_emb, G)

        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, G)
        prev = start
        paths_list = [start]
        log_probs_list = []

        for step in range(N - 1):

            # --------------------
            # TW feasibility mask
            # --------------------
            if self.enforce_time_windows and self.tw_start is not None:
                travel_all = self.distances[batch_idx, prev]  # (B, G, N)
                arrival = torch.maximum(
                    t_cur[:, :, None] + travel_all,
                    self.tw_start[:, None, :],
                )
                invalid_tw = arrival > (self.tw_end[:, None, :] + self.eps)

                finish = arrival + self.service_time[:, None, :]
                back = self.distances[:, :, self.depot][:, None, :]  # (B,1,N)
                invalid_return = finish + back > (
                    self.tw_end[:, self.depot].view(B, 1, 1) + self.eps
                )

                tw_mask = (~(invalid_tw | invalid_return)).float()
            else:
                tw_mask = 1.0

            # Sampling mask: only exclude visited nodes (ignore TW feasibility for sampling)
            if self.tw_mask=="on":
                combined_mask = mask * tw_mask                     # (G, N)
            else:
                combined_mask = mask 
            row_sum = combined_mask.sum(dim=-1, keepdim=True)  # (B, G, 1)
            fallback = (row_sum == 0).float()
            combined_mask = combined_mask + fallback * mask    # 최소 하나는 valid하게

            # --------------------
            # 확률 분포 계산 (NAR vs AR)
            # --------------------
            if not use_ar:
                # 기존 NAR 방식
                dist_rows = prob_mat[batch_idx, prev]  # (B, G, N)
                if self.heuristic_output_space == "logit":
                    denom = self.beta if self.beta != 0 else 1.0
                    logits = dist_rows / denom
                    logits = logits.masked_fill(combined_mask <= 0, float("-inf"))
                    log_probs_step = F.log_softmax(logits, dim=-1) if require_prob else None
                else:
                    dist = (dist_rows ** alpha) * combined_mask  # (B, G, N)
                    row_sum = dist.sum(dim=-1, keepdim=True)
                    zero_rows = row_sum <= 0
                    safe_row_sum = torch.where(zero_rows, torch.ones_like(row_sum), row_sum)
                    probs = dist / (safe_row_sum + 1e-12)
                    if zero_rows.any():
                        uniform = mask / (mask.sum(dim=-1, keepdim=True) + 1e-12)
                        probs = torch.where(zero_rows, uniform, probs)
                    probs_flat = probs.view(B * G, N)
                    log_probs_step = None  # NAR에서는 따로 안 씀
            else:
                # ★ AR 디코더: dyn_feat = [현재 시간, 누적 이동 거리]
                dyn_t = t_cur        # (B, G)
                dyn_d = traveled     # (B, G)
                # dyn_feat = torch.stack([dyn_t, dyn_d], dim=-1) * self.dyn_scale  # (B, G, 2)
                dyn_feat = torch.stack([dyn_t], dim=-1) * self.dyn_scale  # (B, G, 1)

                log_probs_step = self.decoder(
                    node_emb=decoder_node_emb,      # (B, N, D)
                    cur_idx=prev,                # (B, G)
                    first_idx=start,             # (B, G)
                    dyn_feat=dyn_feat,           # (B, G, dyn_dim)
                    mask=combined_mask,          # (B, G, N)
                )                               # -> (B, G, N), log-softmax

                probs = log_probs_step.exp()
                # probs = log_probs_step
                probs_flat = probs.view(B * G, N)

            # --------------------
            # action 선택
            # --------------------
            if paths is not None:
                actions = paths[step + 1]
                if actions.dim() == 1:
                    actions = actions.view(1, -1).expand(B, -1)
                actions_flat = actions.view(B * G)
            elif self.use_pomo and step == 0 and paths is None:
                avail = max(N - 1, 1)
                forced = (torch.arange(G, device=device) % avail) + 1  # (G,)
                actions = forced.unsqueeze(0).expand(B, -1)
                actions_flat = actions.reshape(B * G)
                if require_prob:
                    log_probs_list.append(torch.zeros_like(actions, dtype=torch.float, device=device))
            else:
                if not use_ar and self.heuristic_output_space == "logit":
                    dist_cat = Categorical(logits=logits.view(B * G, N))
                else:
                    dist_cat = Categorical(probs=probs_flat)
                epsilon = 0.05
                if desi == 1:
                    random_mask = torch.rand(B * G, device=device) < epsilon
                    sampled = dist_cat.sample()
                    if not use_ar and self.heuristic_output_space == "logit":
                        greedy = logits.view(B * G, N).argmax(dim=-1)
                    else:
                        greedy = probs_flat.argmax(dim=-1)
                    actions_flat = torch.where(random_mask, sampled, greedy)
                else:
                    actions_flat = dist_cat.sample()
                actions = actions_flat.view(B, G)

            # --------------------
            # log-prob 기록
            # --------------------
            if require_prob:
                if not use_ar:
                    if self.heuristic_output_space == "logit":
                        logp = log_probs_step.gather(
                            -1, actions.unsqueeze(-1)
                        ).squeeze(-1)  # (B, G)
                    else:
                        # 기존 NAR: probs에서 log 취함
                        probs_act = probs_flat.gather(1, actions_flat.view(-1, 1)).clamp_min(1e-12)
                        logp_flat = probs_act.log().squeeze(-1)
                        logp = logp_flat.view(B, G)
                else:
                    # AR: 디코더에서 나온 log_probs_step를 그대로 사용
                    logp = log_probs_step.gather(
                        -1, actions.unsqueeze(-1)
                    ).squeeze(-1)  # (B, G)
                log_probs_list.append(logp)

            paths_list.append(actions)

            # --------------------
            # 시간 / 거리 상태 업데이트
            # --------------------
            step_travel = self.distances[batch_idx, prev, actions]  # (B, G)
            traveled = traveled + step_travel                       # ★ 순수 이동 거리 누적

            if self.enforce_time_windows and self.tw_start is not None:
                arr = torch.maximum(
                    t_cur + step_travel,
                    self.tw_start[batch_idx, actions],
                )
                t_cur = arr + self.service_time[batch_idx, actions]

            # 방문 처리
            prev = actions
            mask.scatter_(-1, actions.unsqueeze(-1), 0.0)

        paths_tensor = torch.stack(paths_list, dim=0)  # (N, B, G)
        if require_prob:
            log_probs_tensor = torch.stack(log_probs_list, dim=0)  # (N-1, B, G)
            return paths_tensor, log_probs_tensor
        return paths_tensor

        
    @torch.no_grad()
    def get_costs(self, paths):
        """
        - distances: (N, N),  paths: (N, G)      -> (G,)
        - distances: (B, N, N), paths: (N,B,G)   -> (B, G)
        """
        if self.distances.dim() == 2:
            return self._get_costs_single(paths)
        else:
            return self._get_costs_batched(paths)
        
    @torch.no_grad()
    def _get_costs_single(self, paths):
        assert paths.shape == (self.scale_, self.generate)
        u = paths.T                      # (G, N)
        v = torch.roll(u, shifts=1, dims=1)
        return torch.sum(self.distances[u, v], dim=1)

    @torch.no_grad()
    def _get_costs_batched(self, paths):
        assert paths.dim() == 3, "batched costs expect paths of shape (N, B, G)"
        N, B, G = paths.shape
        assert N == self.scale_, f"paths first dim {N} != scale_ {self.scale_}"
        dist = self.distances.to(paths.device)    # (B, N, N)

        # (N,B,G) -> (B,G,N)
        tours = paths.permute(1, 2, 0)
        u = tours
        v = torch.roll(tours, shifts=1, dims=-1)

        batch_idx = torch.arange(B, device=paths.device).view(B, 1, 1)
        batch_idx = batch_idx.expand(B, G, N)
        return dist[batch_idx, u, v].sum(dim=-1)  # (B, G)

    def generate_route(self, alpha=1.0, inference=False, start_node=None, desi=1):
        paths, log_probs = self.get_route(alpha=alpha, require_prob=True, start_node=start_node, desi=desi)
        paths, log_probs_D = self.get_route(alpha=alpha, require_prob=True, start_node=start_node, paths=paths, desi=desi)
        costs = self.get_costs(paths)
        return costs, log_probs, paths, log_probs_D

    @torch.no_grad()
    def val(self, n_iterations, inference=True, start_node=None, return_records=False, compute_bpd=True):
        assert n_iterations > 0
        collected_paths = []
        records = [] if return_records else None

        for _ in range(n_iterations):
            paths = self.get_route(alpha=1.0, require_prob=False, start_node=start_node, desi=1)
            _paths = paths.clone()
            costs = self.get_costs(paths)
            collected_paths.append(paths.T.detach().cpu())
            if return_records:
                records.append({
                    "paths": paths.detach().cpu(),
                    "costs": costs.detach().cpu(),
                })

            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost.item()

        diversity = self._calculate_diversity(collected_paths, compute_bpd=compute_bpd)
        if return_records:
            return self.lowest_cost, diversity, records
        return self.lowest_cost, diversity
    
    def improve_route(self, paths, start_node=None, alpha=1.0):
        paths = self.improve(paths)
        costs = self.get_costs(paths)
        paths, log_probs_D = self.get_route(alpha=alpha, require_prob=True, start_node=start_node, paths=paths)
        return costs, paths, log_probs_D

    def improve(self, paths, inference=False):
        if self.distances.dim() == 2:
            if self.local_search_type == "2opt":
                paths = self.improve_t(paths, inference)
            elif self.local_search_type == "nls":
                paths = self.improve_n(paths, inference, T_nls=self.nls_T_nls, T_p=self.nls_T_p)
            return paths

        # 배치 인스턴스: distances.shape == (B, N, N), paths.shape == (N, B, G)
        if self.local_search_type == "2opt":
            paths_bpg = paths.permute(1, 2, 0)  # (B,G,N)
            improved = self._two_opt_batch_gpu(paths_bpg, self.distances, inference=inference)
            return improved.permute(2, 0, 1)
        elif self.local_search_type == "nls":
            paths_bpg = paths.permute(1, 2, 0)
            improved = self._nls_gpu(paths_bpg, inference=inference, T_nls=self.nls_T_nls, T_p=self.nls_T_p)
            return improved.permute(2, 0, 1)
        return paths
    
    def _route_cost_from_dist(self, paths, dist_matrix=None):
        """
        paths: (B, P, L), dist_matrix: (B, N, N)
        return: (B, P)
        """
        dist = self.distances if dist_matrix is None else dist_matrix
        dist = dist.to(paths.device)
        B, P, L = paths.size()

        batch_idx = torch.arange(B, device=paths.device).view(B, 1, 1)
        u = paths
        v = torch.roll(paths, shifts=1, dims=-1)
        return dist[batch_idx, u, v].sum(dim=-1)

    def _two_opt_batch_gpu(self, paths, dist_matrix, inference=False, chunk_size: int = 32):
        """
        GPU friendly 2-opt for batched tours.
        paths: (B, P, L), dist_matrix: (B, N, N)
        """
        maxt = 10000 if inference else max(1, self.scale_ // 4)
        improved = paths.clone()
        B, P, L = improved.size()
        idx = torch.arange(L, device=improved.device)
        try:
            i_idx, j_idx = torch.meshgrid(idx, idx, indexing="ij")
        except TypeError:
            i_idx, j_idx = torch.meshgrid(idx, idx)

        valid_mask = (j_idx > i_idx) & (i_idx > 0)
        valid_mask = valid_mask.view(1, 1, L, L)  # (1,1,L,L)

        dist = dist_matrix.to(improved.device)
        batch_idx_base = torch.arange(B, device=improved.device).view(B, 1, 1, 1)

        for _ in range(maxt):
            prev_i_full = torch.roll(improved, 1, dims=-1)   # (B,P,L)
            next_j_full = torch.roll(improved, -1, dims=-1)  # (B,P,L)

            best_delta = torch.full((B, P), float("inf"), device=improved.device)
            best_i = torch.zeros((B, P), dtype=torch.long, device=improved.device)
            best_j = torch.zeros((B, P), dtype=torch.long, device=improved.device)

            for j_start in range(0, L, chunk_size):
                j_end = min(L, j_start + chunk_size)
                j_len = j_end - j_start

                node_i = improved.unsqueeze(-1).expand(-1, -1, -1, j_len)           # (B,P,L,j_len)
                node_j = improved.unsqueeze(-2)[..., j_start:j_end].expand(-1, -1, L, j_len)
                prev_i = prev_i_full.unsqueeze(-1).expand(-1, -1, -1, j_len)
                next_j = next_j_full.unsqueeze(-2)[..., j_start:j_end].expand(-1, -1, L, j_len)

                valid_chunk = valid_mask[..., j_start:j_end]                        # (1,1,L,j_len)

                batch_idx = batch_idx_base.expand(B, P, L, j_len)
                # 실제 delta 계산
                delta_chunk = (
                    dist[batch_idx, prev_i, node_j]
                    + dist[batch_idx, node_i, next_j]
                    - dist[batch_idx, prev_i, node_i]
                    - dist[batch_idx, node_j, next_j]
                )
                delta_chunk = delta_chunk.masked_fill(~valid_chunk, float("inf"))

                delta_flat = delta_chunk.view(B, P, -1)
                chunk_best_delta, chunk_idx = delta_flat.min(dim=-1)

                improve_mask = chunk_best_delta < best_delta
                best_delta = torch.where(improve_mask, chunk_best_delta, best_delta)

                i_local = chunk_idx // j_len
                j_local = chunk_idx % j_len + j_start
                best_i = torch.where(improve_mask, i_local, best_i)
                best_j = torch.where(improve_mask, j_local, best_j)

            improving = best_delta < -1e-6
            if not torch.any(improving):
                break

            pos_idx = idx.view(1, 1, L)
            i_exp = best_i.unsqueeze(-1)
            j_exp = best_j.unsqueeze(-1)
            rev_indices = torch.where(
                pos_idx < i_exp,
                pos_idx,
                torch.where(pos_idx > j_exp, pos_idx, i_exp + j_exp - pos_idx),
            )
            improved = improved.gather(-1, rev_indices)

        return improved

    def _nls_gpu(self, paths, inference=False, T_nls: int | None = None, T_p: int | None = None):
        """
        GPU 기반 NLS (노이즈 2-opt) – batched.
        paths: (B, P, L)
        """
        if T_nls is None:
            T_nls = self.nls_T_nls
        if T_p is None:
            T_p = self.nls_T_p
        best_paths = self._two_opt_batch_gpu(paths, self.distances, inference=inference)
        best_costs = self._route_cost_from_dist(best_paths)
        new_paths = best_paths

        # heuristic 거리행렬 (작을수록 선호)
        heur = self.heuristic.to(paths.device)
        heur_norm = heur / heur.amax(dim=-1, keepdim=True).clamp_min(EPS)
        heuristic_dist = 1.0 / (heur_norm + 1e-5)

        for _ in range(T_nls):
            perturbed_paths = self._two_opt_batch_gpu(new_paths, heuristic_dist, inference=False)
            new_paths = self._two_opt_batch_gpu(perturbed_paths, self.distances, inference=inference)
            new_costs = self._route_cost_from_dist(new_paths)

            improved_indices = new_costs < best_costs
            best_paths = torch.where(improved_indices.unsqueeze(-1), new_paths, best_paths)
            best_costs = torch.where(improved_indices, new_costs, best_costs)

        return best_paths


    @staticmethod
    def _tour_to_edge_signature(tour: np.ndarray):
        n = len(tour)
        edges = []
        for i in range(n):
            u = int(tour[i])
            v = int(tour[(i + 1) % n])
            if u == v:
                continue
            edges.append((u, v))
        edges.sort()
        return tuple(edges)

    @classmethod
    def _calculate_diversity(cls, collected_paths, compute_bpd=True):
        bpd_default = 0.0 if compute_bpd else float("nan")
        if not collected_paths:
            return {"unique_ratio": 0.0, "bpd": bpd_default}

        tours = torch.cat(collected_paths, dim=0).numpy()
        if tours.size == 0:
            return {"unique_ratio": 0.0, "bpd": bpd_default}

        edge_signatures = [cls._tour_to_edge_signature(tour) for tour in tours]
        total_tours = len(edge_signatures)
        if total_tours == 0:
            return {"unique_ratio": 0.0, "bpd": bpd_default}

        unique_ratio = len(set(edge_signatures)) / total_tours

        if not compute_bpd:
            return {"unique_ratio": float(unique_ratio), "bpd": bpd_default}

        if total_tours < 2:
            avg_bpd = 0.0
        else:
            edge_sets = [set(signature) for signature in edge_signatures]
            n_nodes = len(tours[0])
            total_bpd = 0.0
            pair_count = 0
            for i in range(total_tours - 1):
                set_i = edge_sets[i]
                for j in range(i + 1, total_tours):
                    inter_size = len(set_i & edge_sets[j])
                    bpd = 1.0 - (inter_size / n_nodes) if n_nodes > 0 else 0.0
                    total_bpd += bpd
                    pair_count += 1
            avg_bpd = total_bpd / pair_count if pair_count else 0.0

        return {"unique_ratio": float(unique_ratio), "bpd": float(avg_bpd)}

    def gen_numpy_path_costs(self, paths):
        assert paths.shape == (self.generate, self.scale_)
        u = paths
        v = np.roll(u, shift=1, axis=1)
        return np.sum(self.distances_numpy[u, v], axis=1)

    def improve_t(self, paths, inference=False):
        maxt = 10000 if inference else self.scale_ // 4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.T.cpu().numpy(), max_iterations=maxt)
        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)

        return best_paths

    def improve_n(self, paths, inference=False, T_nls: int | None = None, T_p: int | None = None):
        if T_nls is None:
            T_nls = self.nls_T_nls
        if T_p is None:
            T_p = self.nls_T_p
        maxt = 10000 if inference else self.scale_ // 4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.T.cpu().numpy(), max_iterations=maxt)
        best_costs = self.gen_numpy_path_costs(best_paths)
        new_paths = best_paths

        for _ in range(T_nls):
            perturbed_paths = batched_two_opt_python(self.heuristic_dist, new_paths, max_iterations=T_p)
            new_paths = batched_two_opt_python(self.distances_numpy, perturbed_paths, max_iterations=maxt)
            new_costs = self.gen_numpy_path_costs(new_paths)

            improved_indices = new_costs < best_costs
            best_paths[improved_indices] = new_paths[improved_indices]
            best_costs[improved_indices] = new_costs[improved_indices]

        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)
        return best_paths
    
    @cached_property
    def distances_numpy(self):
        return self.distances.detach().cpu().numpy().astype(np.float32)

    @cached_property
    def heuristic_numpy(self):
        return self.heuristic.detach().cpu().numpy().astype(np.float32)

    @cached_property
    def heuristic_dist(self):
        return 1 / (self.heuristic_numpy / self.heuristic_numpy.max(-1, keepdims=True) + 1e-5)
    
@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16), nogil=True)
def two_opt_once(distmat, tour, fixed_i = 0):
    '''in-place operation'''
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i==0 else range(fixed_i, fixed_i + 1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i - 1], tour[(j + 1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (
                distmat[node_prev, node_j] + distmat[node_i, node_next]
                - distmat[node_prev, node_i] - distmat[node_j, node_next]
            )
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p: q + 1] = np.flip(tour[p: q + 1])
        return delta
    else:
        return 0.0


@nb.njit(nb.uint16[:](nb.float32[:,:], nb.uint16[:], nb.int64), nogil=True)
def _two_opt_python(distmat, tour, max_iterations=1000):
    iterations = 0
    min_change = -1.0
    while min_change < -1e-6 and iterations < max_iterations:
        min_change = two_opt_once(distmat, tour, 0)
        iterations += 1
    return tour


def batched_two_opt_python(dist: np.ndarray, tours: np.ndarray, max_iterations=1000):
    dist = dist.astype(np.float32)
    tours = tours.astype(np.uint16)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for tour in tours:
            future = executor.submit(partial(_two_opt_python, distmat=dist, max_iterations=max_iterations), tour = tour)
            futures.append(future)
        return np.stack([f.result() for f in futures])
