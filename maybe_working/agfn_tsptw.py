import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Respect any externally selected GPU; otherwise default to the first visible GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pytz
import argparse
import pprint as pp
from datetime import datetime
import logging
import atexit
import csv
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import time
import random
import math
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from scipy.stats import ttest_rel
import torch.nn.functional as F
from dataclasses import dataclass
import torch
import pickle
import numpy as np
import sys
from subprocess import check_call
from urllib.parse import urlparse
from typing import List
from functools import cached_property
import concurrent.futures
from torch import nn, autograd
from torch.nn import functional as F
from torch.distributions import Categorical
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
import platform
from ctypes import Structure, CDLL, POINTER, c_int, c_double, c_char, sizeof, cast, byref
import scipy
from models.AGFNModel import Net, search, PomoLiteDecoder
from models.single_model import SINGLEModel as LegacySingleModel
import json

FIGURE_FONT_PATH = os.environ.get("AGFN_FIGURE_FONT", "/home/shyoon/Machine-intelligence/nimbusromandcy.otf")
EPS = 1e-10
START_NODE = 0
COORD_SCALE = 100
TIME_WINDOW_SCALE = None
GEN_SCALE = 100
TSP_FAKE_TW_END = 1e6


class SingleModelSearchDecoder(nn.Module):
    """
    Adapter to reuse SINGLEModel decoder inside the searcher.
    It caches encoded nodes and produces log-probabilities with masking.
    """

    def __init__(self, base_model: LegacySingleModel):
        super().__init__()
        self.base_model = base_model
        self.encoded_nodes: torch.Tensor | None = None

    def reset_cache(self):
        self.encoded_nodes = None

    def set_kv(self, encoded_nodes: torch.Tensor):
        # encoded_nodes: (B, N, D)
        self.encoded_nodes = encoded_nodes
        self.base_model.decoder.set_kv(encoded_nodes)

    def set_q1(self, *args, **kwargs):
        # SINGLE decoder does not use q1; keep for interface compatibility
        return

    def set_q2(self, *args, **kwargs):
        # SINGLE decoder does not use q2; keep for interface compatibility
        return

    def forward(self, node_emb, cur_idx, first_idx, dyn_feat, mask=None):
        """
        node_emb: (B, N, D) (unused; cached encoded_nodes are used)
        cur_idx : (B, G)
        first_idx: (B, G) (unused by SINGLEModel decoder)
        dyn_feat: (B, G, 1) current time feature
        mask    : (B, G, N) with 1 valid, 0 invalid
        """
        if self.encoded_nodes is None:
            raise RuntimeError("SingleModelSearchDecoder.set_kv must be called before forward")
        B, G = cur_idx.shape
        D = self.encoded_nodes.size(-1)
        gather_idx = cur_idx.unsqueeze(-1).expand(-1, -1, D)
        encoded_last = self.encoded_nodes.gather(1, gather_idx)  # (B, G, D)

        ninf_mask = None
        if mask is not None:
            ninf_mask = torch.where(mask > 0, 0.0, float("-inf"))

        probs = self.base_model.decoder(encoded_last, dyn_feat, ninf_mask=ninf_mask)
        log_probs = probs.clamp_min(1e-12).log()
        return log_probs

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0

class TimeEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total - count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time * 60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time * 60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)
        print("Step {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(count, total, elapsed_time_str,
                                                                             remain_time_str))
    
    def cal_time(t_time):
        d = int(t_time // (24 * 3600))
        h = int((t_time % (24 * 3600)) // 3600)
        m = int((t_time % 3600) // 60)
        s = int(t_time % 60)
        print(f"{d}days : {h}h : {m}m : {s}s")


class StreamToLogger:
    def __init__(self, logger, level, stream):
        self.logger = logger
        self.level = level
        self.stream = stream
        self._buffer = ""

    def write(self, message):
        if not message:
            return
        self.stream.write(message)
        self.stream.flush()
        sanitized = message.replace('\r', '\n')
        self._buffer += sanitized
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            line_to_log = line.rstrip()
            if line_to_log:
                self.logger.log(self.level, line_to_log)

    def flush(self):
        self.stream.flush()
        if self._buffer:
            line_to_log = self._buffer.rstrip()
            if line_to_log:
                self.logger.log(self.level, line_to_log)
            self._buffer = ""

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")

    def writable(self):
        return True


def setup_logging(log_directory, base_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_{base_filename}"
    log_file_path = os.path.join(log_directory, log_filename)

    logger = logging.getLogger("agfn_tsptw")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

    original_stdout = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout
    original_stderr = sys.__stderr__ if getattr(sys, "__stderr__", None) is not None else sys.stderr
    stdout_logger = StreamToLogger(logger, logging.INFO, original_stdout)
    stderr_logger = StreamToLogger(logger, logging.INFO, original_stderr)
    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    atexit.register(stdout_logger.flush)
    atexit.register(stderr_logger.flush)
    return log_file_path

def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used

def occumpy_mem(args):
    """
        Occupy GPU memory in advance.
    """
    torch.cuda.set_device(args.gpu_id)
    total, used = check_mem(args.gpu_id)
    total, used = int(total), int(used)
    block_mem = int((total - used) * args.occ_gpu)
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x

def seed_everything(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_env(problem):
    from envs import TSPEnv, TSPTWEnv
    training_problems = ['TSP', 'TSPTW']
    all_problems = {
        'TSP': TSPEnv,
        'TSPTW': TSPTWEnv,
    }
    if problem == "Train_ALL":
        return [all_problems[i] for i in training_problems]
    elif problem == "ALL":
        return list(all_problems.values())
    else:
        return [all_problems[problem]]

def get_opt_sol_path(dir, problem, size, hardness="no"):
    if problem in ["TSPTW"]:
        return os.path.join(dir, f"lkh_{problem.lower()}{size}_{hardness}.pkl")
    all_opt_sol = {
        'TSP': {50: 'lkh_tsp50_uniform.pkl', 100: 'lkh_tsp100_uniform.pkl', 200: 'lkh_tsp200_uniform.pkl'},
        'CVRP': {50: 'hgs_cvrp50_uniform.pkl', 100: 'hgs_cvrp100_uniform.pkl', 200: 'hgs_cvrp200_uniform.pkl'},
        'OVRP': {50: 'or_tools_200s_ovrp50_uniform.pkl', 100: 'lkh_ovrp100_uniform.pkl'},
        'VRPB': {50: 'or_tools_200s_vrpb50_uniform.pkl', 100: 'or_tools_400s_vrpb100_uniform.pkl'},
        'VRPL': {50: 'or_tools_200s_vrpl50_uniform.pkl', 100: 'lkh_vrpl100_uniform.pkl'},
        'VRPTW': {50: 'hgs_vrptw50_uniform.pkl', 100: 'hgs_vrptw100_uniform.pkl', 200: 'hgs_vrptw200_uniform.pkl'},
        'OVRPTW': {50: 'or_tools_200s_ovrptw50_uniform.pkl', 100: 'or_tools_400s_ovrptw100_uniform.pkl'},
        'OVRPB': {50: 'or_tools_200s_ovrpb50_uniform.pkl', 100: 'or_tools_400s_ovrpb100_uniform.pkl'},
        'OVRPL': {50: 'or_tools_200s_ovrpl50_uniform.pkl', 100: 'or_tools_400s_ovrpl100_uniform.pkl'},
        'VRPBL': {50: 'or_tools_200s_vrpbl50_uniform.pkl', 100: 'or_tools_400s_vrpbl100_uniform.pkl'},
        'VRPBTW': {50: 'or_tools_200s_vrpbtw50_uniform.pkl', 100: 'or_tools_400s_vrpbtw100_uniform.pkl'},
        'VRPLTW': {50: 'or_tools_200s_vrpltw50_uniform.pkl', 100: 'or_tools_400s_vrpltw100_uniform.pkl'},
        'OVRPBL': {50: 'or_tools_200s_ovrpbl50_uniform.pkl', 100: 'or_tools_400s_ovrpbl100_uniform.pkl'},
        'OVRPBTW': {50: 'or_tools_200s_ovrpbtw50_uniform.pkl', 100: 'or_tools_400s_ovrpbtw100_uniform.pkl'},
        'OVRPLTW': {50: 'or_tools_200s_ovrpltw50_uniform.pkl', 100: 'or_tools_400s_ovrpltw100_uniform.pkl'},
        'VRPBLTW': {50: 'or_tools_200s_vrpbltw50_uniform.pkl', 100: 'or_tools_400s_vrpbltw100_uniform.pkl'},
        'OVRPBLTW': {50: 'or_tools_200s_ovrpbltw50_uniform.pkl', 100: 'or_tools_400s_ovrpbltw100_uniform.pkl'},
    }
    return os.path.join(dir, all_opt_sol[problem][size])

def num_param(model):
    nb_param = 0
    for param in model.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(nb_param, nb_param / 1e6))

def check_null_hypothesis(a, b):
    print(len(a), a)
    print(len(b), b)
    alpha_threshold = 0.05
    t, p = ttest_rel(a, b)  # Calc p value
    print(t, p)
    p_val = p / 2  # one-sided
    print("p-value: {}".format(p_val))
    if p_val < alpha_threshold:
        print(">> Null hypothesis (two related or repeated samples have identical average values) is Rejected.")
    else:
        print(">> Null hypothesis (two related or repeated samples have identical average values) is Accepted.")

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename, disable_print=False):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    if not disable_print:
        print(">> Save dataset to {}".format(filename))

def load_dataset(filename, disable_print=False):
    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
    if not disable_print:
        print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
    return data

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True, disable_tqdm=True):
    os.makedirs(directory, exist_ok=True)
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval, disable=disable_tqdm))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus

def show(x, y, label, title, xdes, ydes, path, min_y=None, max_y=None, x_scale="linear", dpi=300):
    rc_updates = {
        "font.weight": "bold",
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.facecolor": "#f8f8f8",
    }
    if FIGURE_FONT_PATH and os.path.exists(FIGURE_FONT_PATH):
        fm.fontManager.addfont(FIGURE_FONT_PATH)
        font_prop = fm.FontProperties(fname=FIGURE_FONT_PATH, weight='bold')
        rc_updates["font.family"] = font_prop.get_name()
    plt.rcParams.update(rc_updates)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']

    assert len(x) == len(y)
    for i in range(len(x)):
        series_label = label[i] if i < len(label) else None
        ax.plot(x[i], y[i], marker=None,
                color=colors[i % len(colors)], label=series_label, markersize=2, linewidth=1.0)

    if min_y is not None and max_y is not None:
        ax.set_ylim((min_y, max_y))

    ax.set_xlabel(xdes)
    ax.set_ylabel(ydes)
    ax.set_title(title)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_weight('bold')
    ax.set_xscale(x_scale)
    ax.grid(True, ls='--', lw=0.6, alpha=0.7)
    if label:
        leg = ax.legend(loc='best', frameon=True)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(1.0)
    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

class Trainer:
    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.device = args.device
        self.log_path = args.log_path
        self.result_log = {
            "val_score": [],
            "val_gap": [],
            "val_unique_ratio": [],
            "val_bpd": [],
            "val_infeas_sol": [],
            "val_infeas_inst": [],
        }
        self.validation_steps = []
        self.problem_type = self.args.problem.upper()
        self.start_node = START_NODE
        legacy_tw_penalty = getattr(self.args, "tw_penalty_weight", 1.0)
        self.tw_penalty_weight_gflow = getattr(self.args, "tw_penalty_weight_gflow", None)
        if self.tw_penalty_weight_gflow is None:
            self.tw_penalty_weight_gflow = legacy_tw_penalty
        self.cost_penalty_weight = getattr(self.args, "cost_penalty_weight", 1.0)
        rate_weight = getattr(self.args, "cost_penalty_weight_rate", None)
        self.cost_penalty_weight_rate = self.cost_penalty_weight if rate_weight is None else rate_weight
        self.cost_penalty_type = str(getattr(self.args, "cost_penalty_type", "delta_value")).lower()
        self.K = max(1, int(getattr(self.args, "K", 1)))
        self.rho_x_init = float(getattr(self.args, "rho_x_init", 0.5))
        self.rho_z_init = float(getattr(self.args, "rho_z_init", 0.5))
        self.consensus_weight_update_interval = max(
            1, int(getattr(self.args, "consensus_weight_update_interval", 1))
        )
        self.consensus_correlation_threshold = float(
            getattr(self.args, "consensus_correlation_threshold", 0.1)
        )
        self.grad_accum = bool(getattr(self.args, "grad_accum", False))
        self.compute_val_bpd = bool(getattr(self.args, "val_bpd", False))
        self.select_best_mode = str(getattr(self.args, "select_best_mode", "feasible_min_cost")).lower()
        self.occ_time_bins = max(2, int(getattr(self.args, "occ_time_bins", 16)))
        self.occ_aux_weight = float(getattr(self.args, "occ_aux_weight", 0.1))

        # Main Components
        self.use_single_model = (
            self.args.encoder_type == "single_model" and self.args.decoding_type == "single_model"
        )
        if (self.args.encoder_type == "single_model") != (self.args.decoding_type == "single_model"):
            raise ValueError("encoder_type and decoding_type must both be 'single_model' or neither.")

        self.env = get_env(self.args.problem)[0](**self.env_params)  # a list of env classes
        self.problem_size = int(self.env_params["problem_size"])
        base_node_feature_dim = 5 if self.problem_type == "TSPTW" else 2
        self.base_node_feature_dim = base_node_feature_dim
        if self.problem_type == "TSPTW":
            self.rho_feature_idx = base_node_feature_dim
            self.u_feature_start = base_node_feature_dim + 1
            self.v_feature_start = self.u_feature_start + self.problem_size
            node_feature_dim = base_node_feature_dim + 1 + 2 * self.problem_size
        else:
            self.rho_feature_idx = None
            self.u_feature_start = None
            self.v_feature_start = None
            node_feature_dim = base_node_feature_dim
        self.node_feature_dim = node_feature_dim
        self.rep_net = None
        self.rep_decoder = None
        generator_params = []
        repair_params = []
        if self.use_single_model:
            single_params = {
                "problem": self.problem_type,
                "eval_type": "softmax",
                "embedding_dim": self.args.embedding_dim,
                "encoder_layer_num": self.args.sm_encoder_layer_num,
                "head_num": self.args.sm_head_num,
                "qkv_dim": self.args.sm_qkv_dim,
                "ff_hidden_dim": self.args.sm_ff_hidden_dim,
                "sqrt_embedding_dim": math.sqrt(self.args.embedding_dim),
                "logit_clipping": self.args.sm_logit_clipping,
                "tw_normalize": self.args.sm_tw_normalize,
                "device": self.device,
            }
            self.net = LegacySingleModel(**single_params).to(self.device)
            self.decoder = SingleModelSearchDecoder(self.net).to(self.device)
            generator_params = list(self.net.parameters())
            self.rep_net = LegacySingleModel(**single_params).to(self.device)
            self.rep_decoder = SingleModelSearchDecoder(self.rep_net).to(self.device)
            repair_params = list(self.rep_net.parameters())
        else:
            self.net = Net(
                gfn=True,
                Z_out_dim=1,
                node_feature_dim=node_feature_dim,
                embedding_dim=self.args.embedding_dim,
                encoder_type=self.args.encoder_type,
                matrix_output_space=self.args.gen_matrix_output_space,
                ).to(self.device)
            generator_params = list(self.net.parameters())
            self.decoder = None
            self.rep_decoder = None
            self.rep_net = Net(
                gfn=True,
                Z_out_dim=1,
                node_feature_dim=node_feature_dim,
                embedding_dim=self.args.embedding_dim,
                encoder_type=self.args.encoder_type,
                matrix_output_space=self.args.gen_matrix_output_space,
                ).to(self.device)
            repair_params = list(self.rep_net.parameters())
            if self.args.decoding_type == "ar":
                self.decoder = PomoLiteDecoder(
                    node_dim=self.args.embedding_dim,
                    att_dim=self.args.embedding_dim,
                    ).to(self.device)
                generator_params += list(self.decoder.parameters())
                self.rep_decoder = PomoLiteDecoder(
                    node_dim=self.args.embedding_dim,
                    att_dim=self.args.embedding_dim,
                    ).to(self.device)
                repair_params += list(self.rep_decoder.parameters())
        if self.optimizer_params["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(generator_params, lr=self.optimizer_params["lr"], weight_decay=self.optimizer_params["weight_decay"])
            self.rep_optimizer = torch.optim.Adam(
                repair_params, lr=self.optimizer_params["lr"], weight_decay=self.optimizer_params["weight_decay"]
            )
        elif self.optimizer_params["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(generator_params, lr=self.optimizer_params["lr"], weight_decay=self.optimizer_params["weight_decay"])
            self.rep_optimizer = torch.optim.AdamW(
                repair_params, lr=self.optimizer_params["lr"], weight_decay=self.optimizer_params["weight_decay"]
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            self.trainer_params["steps"],
            eta_min=self.optimizer_params["lr_min"],
        )
        self.rep_scheduler = None
        if self.rep_optimizer is not None:
            self.rep_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.rep_optimizer,
                self.trainer_params["steps"],
                eta_min=self.optimizer_params["lr_min"],
            )

        # model components

        num_param(self.net)
        if self.rep_net is not None:
            num_param(self.rep_net)
        # Restore
        self.start_step = 1
        if args.checkpoint is not None:
            checkpoint_fullname = args.checkpoint
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.net.load_state_dict(checkpoint['net_state_dict'], strict=True)
            checkpoint_step = int(checkpoint.get("step", checkpoint.get("epoch", 0)))
            self.start_step = 1 + checkpoint_step
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.rep_net is not None and "rep_net_state_dict" in checkpoint:
                self.rep_net.load_state_dict(checkpoint["rep_net_state_dict"], strict=True)
            if self.rep_optimizer is not None and "rep_optimizer_state_dict" in checkpoint:
                self.rep_optimizer.load_state_dict(checkpoint["rep_optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                self.scheduler.last_epoch = checkpoint_step - 1
            if self.rep_scheduler is not None:
                if "rep_scheduler_state_dict" in checkpoint:
                    self.rep_scheduler.load_state_dict(checkpoint["rep_scheduler_state_dict"])
                else:
                    self.rep_scheduler.last_epoch = checkpoint_step - 1
            print(">> Checkpoint (Step: {}) Loaded!".format(checkpoint_step))

        # utility
        self.time_estimator = TimeEstimator()

    def _append_metric_csv(self, filename, step, value):
        path = os.path.join(self.log_path, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        val = value if value is not None else float('nan')
        with open(path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["step", "value"])
            writer.writerow([step, val])

    def _load_metric_series(self, filename):
        path = os.path.join(self.log_path, filename)
        if not os.path.exists(path):
            return [], []
        x_vals, y_vals = [], []
        with open(path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                step_value = row.get("step", row.get("epoch"))
                if step_value is None:
                    continue
                x_vals.append(float(step_value))
                value = row["value"]
                if value in ("", None):
                    y_vals.append(float('nan'))
                else:
                    y_vals.append(float(value))
        return x_vals, y_vals

    def _update_metric_plot(self, csv_name, fig_name, title, label, xdes, ydes):
        x_vals, y_vals = self._load_metric_series(csv_name)
        if not x_vals:
            return
        show([x_vals], [y_vals], [label], title, xdes, ydes, os.path.join(self.log_path, fig_name))

    @staticmethod
    def _augment_xy_8_fold(coords_norm: torch.Tensor) -> torch.Tensor:
        # coords_norm: (B, N, 2)
        x = coords_norm[:, :, [0]]
        y = coords_norm[:, :, [1]]

        dats = torch.stack([
            torch.cat((x, y), dim=2),
            torch.cat((1-x, y), dim=2),
            torch.cat((x, 1-y), dim=2),
            torch.cat((1-x, 1-y), dim=2),
            torch.cat((y, x), dim=2),
            torch.cat((1-y, x), dim=2),
            torch.cat((y, 1-x), dim=2),
            torch.cat((1-y, 1-x), dim=2),
        ], dim=1)  # (B, 8, N, 2)

        B, A, N, _ = dats.shape
        return dats.reshape(B*A, N, 2)  # (B*8, N, 2)  # instance-major

    def _record_metric(self, csv_name, fig_name, step, value, title, label, xdes, ydes):
        try:
            numeric_value = float(value) if value is not None else float('nan')
        except (TypeError, ValueError):
            numeric_value = float('nan')
        self._append_metric_csv(csv_name, step, numeric_value)
        self._update_metric_plot(csv_name, fig_name, title, label, xdes, ydes)

    def _build_base_batch_cache(self, base_data, device=None, include_pyg=False):
        if not base_data:
            raise ValueError("base_data must not be empty.")
        target_device = self.device if device is None else device
        distances = torch.stack([entry[1] for entry in base_data], dim=0)
        if distances.device != target_device:
            distances = distances.to(target_device)
        metadata = {
            "coordinates": torch.stack([entry[2]["coordinates"] for entry in base_data], dim=0),
            "service_time": torch.stack([entry[2]["service_time"] for entry in base_data], dim=0),
            "tw_start": torch.stack([entry[2]["tw_start"] for entry in base_data], dim=0),
            "tw_end": torch.stack([entry[2]["tw_end"] for entry in base_data], dim=0),
        }
        for key, value in list(metadata.items()):
            if value.device != target_device:
                metadata[key] = value.to(target_device)
        cache = {
            "distances": distances,
            "metadata": metadata,
        }
        if include_pyg:
            pyg_list = []
            for entry in base_data:
                pyg_data = entry[0]
                if self.problem_type == "TSPTW":
                    pyg_x = getattr(pyg_data, "x", None)
                    if torch.is_tensor(pyg_x) and pyg_x.device != target_device:
                        pyg_data = pyg_data.to(target_device)
                pyg_list.append(pyg_data)
            cache["pyg"] = pyg_list
            if self.problem_type != "TSPTW":
                batch_pyg = Batch.from_data_list(pyg_list)
                batch_x = getattr(batch_pyg, "x", None)
                if torch.is_tensor(batch_x) and batch_x.device != target_device:
                    batch_pyg = batch_pyg.to(target_device)
                cache["batch_pyg"] = batch_pyg
        return cache

    @staticmethod
    def _repeat_metadata_batch(metadata_batch, repeats: int):
        if repeats <= 1:
            return {
                key: value
                for key, value in metadata_batch.items()
                if torch.is_tensor(value) and not str(key).startswith("_")
            }
        repeated = {}
        for key, value in metadata_batch.items():
            if not torch.is_tensor(value):
                continue
            if str(key).startswith("_"):
                continue
            repeated[key] = value.repeat_interleave(repeats, dim=0)
        return repeated

    def _collate_train_batch(self, dataset):
        """
        dataset: list of (pyg_data, distances, metadata)
        returns:
            batch_pyg       : PyG Batch
            batch_distances : (B, N, N) on device
            metadata_batch  : dict with batched tensors (B, ...)
        """
        if not dataset:
            raise ValueError("Empty dataset batch")

        pyg_list = []
        dist_list = []
        coord_list = []
        service_list = []
        tw_start_list = []
        tw_end_list = []
        group_id_list = []
        rho_list = []
        u_list = []
        v_list = []
        rho_seen = False
        u_seen = False
        v_seen = False

        for entry in dataset:
            pyg_data, distances, metadata, _, _ = self._unpack_entry(entry)
            pyg_list.append(pyg_data)
            dist_list.append(distances.to(self.device))
            coord_list.append(torch.as_tensor(metadata["coordinates"], device=self.device))
            service_list.append(torch.as_tensor(metadata["service_time"], device=self.device))
            tw_start_list.append(torch.as_tensor(metadata["tw_start"], device=self.device))
            tw_end_list.append(torch.as_tensor(metadata["tw_end"], device=self.device))
            gid = metadata.get("group_id", torch.tensor(-1))
            if not torch.is_tensor(gid):
                gid = torch.tensor(gid, device=self.device)
            else:
                gid = gid.to(self.device)
            group_id_list.append(gid)
            rho = metadata.get("rho", None)
            if rho is not None:
                rho_list.append(torch.as_tensor(rho, device=self.device, dtype=distances.dtype))
                rho_seen = True
            elif rho_seen:
                rho_list.append(torch.zeros((), device=self.device, dtype=distances.dtype))
            u_mat = metadata.get("u_matrix", None)
            if u_mat is not None:
                u_list.append(torch.as_tensor(u_mat, device=self.device, dtype=distances.dtype))
                u_seen = True
            elif u_seen:
                n_nodes = distances.size(0)
                u_list.append(torch.zeros((n_nodes, n_nodes), device=self.device, dtype=distances.dtype))
            v_mat = metadata.get("v_matrix", None)
            if v_mat is not None:
                v_list.append(torch.as_tensor(v_mat, device=self.device, dtype=distances.dtype))
                v_seen = True
            elif v_seen:
                n_nodes = distances.size(0)
                v_list.append(torch.zeros((n_nodes, n_nodes), device=self.device, dtype=distances.dtype))

        batch_pyg = Batch.from_data_list(pyg_list)
        batch_x = getattr(batch_pyg, "x", None)
        if torch.is_tensor(batch_x) and batch_x.device != self.device:
            batch_pyg = batch_pyg.to(self.device)
        batch_distances = torch.stack(dist_list, dim=0)
        metadata_batch = {
            "coordinates": torch.stack(coord_list, dim=0),
            "service_time": torch.stack(service_list, dim=0),
            "tw_start": torch.stack(tw_start_list, dim=0),
            "tw_end": torch.stack(tw_end_list, dim=0),
            "group_id": torch.stack(group_id_list, dim=0).long(),
        }
        if rho_seen:
            if len(rho_list) < len(dist_list):
                rho_list.extend([torch.zeros((), device=self.device, dtype=dist_list[0].dtype) for _ in range(len(dist_list) - len(rho_list))])
            metadata_batch["rho"] = torch.stack(rho_list, dim=0)
        if u_seen:
            if len(u_list) < len(dist_list):
                n_nodes = dist_list[0].size(0)
                u_list.extend(
                    [torch.zeros((n_nodes, n_nodes), device=self.device, dtype=dist_list[0].dtype) for _ in range(len(dist_list) - len(u_list))]
                )
            metadata_batch["u_matrix"] = torch.stack(u_list, dim=0)
        if v_seen:
            if len(v_list) < len(dist_list):
                n_nodes = dist_list[0].size(0)
                v_list.extend(
                    [torch.zeros((n_nodes, n_nodes), device=self.device, dtype=dist_list[0].dtype) for _ in range(len(dist_list) - len(v_list))]
                )
            metadata_batch["v_matrix"] = torch.stack(v_list, dim=0)
        return batch_pyg, batch_distances, metadata_batch

    @staticmethod
    def _unpack_entry(entry):
        if isinstance(entry, dict):
            return (
                entry["pyg_data"],
                entry["distances"],
                entry["metadata"],
                None,
                entry.get("path"),
            )
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            pyg_data, distances, metadata = entry[:3]
            path = entry[3] if len(entry) >= 4 else None
            return pyg_data, distances, metadata, None, path
        raise ValueError("Unsupported dataset entry format for training.")

    def _extract_single_features(self, batch_pyg):
        """
        Extract (B, N, 4) = [x, y, tw_start, tw_end] for SINGLEModel encoder.
        Assumes constant N per graph (as in current pipeline).
        """
        if isinstance(batch_pyg, Batch):
            B = batch_pyg.num_graphs
            ptr = batch_pyg.ptr
            N = int((ptr[1] - ptr[0]).item())
            x = batch_pyg.x.view(B, N, -1)
        else:
            x = batch_pyg.x.unsqueeze(0)
        return x[:, :, :4]

    def _pomo_start_nodes(self, distances, generate, use_pomo: bool):
        """
        Build start_node tensor (B, G) with deterministic distinct starts.
        For TSPTW depot is fixed at 0; start nodes stay at depot (0).
        """
        if not use_pomo:
            return None
        if distances.dim() == 2:
            return torch.zeros(generate, dtype=torch.long, device=self.device)
        B, _, _ = distances.shape
        base = torch.zeros(generate, dtype=torch.long, device=self.device)
        return base.unsqueeze(0).expand(B, -1)

    @staticmethod
    def _group_mean_by_key(values: torch.Tensor, group_keys: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros_like(values)
        orig_shape = values.shape
        values_flat = values.reshape(-1)
        keys_flat = group_keys.reshape(-1)
        if values_flat.numel() != keys_flat.numel():
            raise ValueError("values and group_keys must have the same number of elements.")

        unique_keys, inverse = torch.unique(keys_flat, sorted=False, return_inverse=True)
        sums = torch.zeros(unique_keys.size(0), device=values_flat.device, dtype=values_flat.dtype)
        counts = torch.zeros(unique_keys.size(0), device=values_flat.device, dtype=values_flat.dtype)
        sums.scatter_add_(0, inverse, values_flat)
        counts.scatter_add_(0, inverse, torch.ones_like(values_flat))
        means_per_group = sums / counts.clamp_min(1.0)
        return means_per_group.gather(0, inverse).view(orig_shape)

    @staticmethod
    def _best_feasible_min_cost_idx(tw_values: torch.Tensor, cost_values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if tw_values.shape != cost_values.shape:
            raise ValueError("tw_values and cost_values must have the same shape.")
        inf_cost = torch.full_like(cost_values, float("inf"))
        feasible = tw_values <= 0

        feasible_costs = torch.where(feasible, cost_values, inf_cost)
        best_feasible_idx = feasible_costs.argmin(dim=dim)
        has_feasible = feasible.any(dim=dim)

        infeasible_tw = torch.where(feasible, torch.full_like(tw_values, float("inf")), tw_values)
        min_infeasible_tw = infeasible_tw.min(dim=dim, keepdim=True).values
        infeasible_ties = (~feasible) & (infeasible_tw <= min_infeasible_tw)
        infeasible_costs = torch.where(infeasible_ties, cost_values, inf_cost)
        best_infeasible_idx = infeasible_costs.argmin(dim=dim)

        return torch.where(has_feasible, best_feasible_idx, best_infeasible_idx)

    @staticmethod
    def _entry_group_id(entry) -> int | None:
        if not isinstance(entry, dict):
            return None
        metadata = entry.get("metadata", {})
        gid = metadata.get("group_id", entry.get("group_id", None))
        if gid is None:
            return None
        if torch.is_tensor(gid):
            return int(gid.item())
        return int(gid)

    def _build_grouped_batches(self, buffer, batch_size):
        if batch_size <= 0:
            return []
        groups = {}
        for entry in buffer:
            gid = self._entry_group_id(entry)
            if gid is None:
                gid = id(entry)
            groups.setdefault(gid, []).append(entry)
        for entries in groups.values():
            random.shuffle(entries)
        group_ids = list(groups.keys())

        batches = []
        while True:
            group_ids = [gid for gid in group_ids if groups[gid]]
            if not group_ids:
                break
            random.shuffle(group_ids)
            batch = []
            for gid in group_ids:
                if not groups[gid]:
                    continue
                batch.append(groups[gid].pop())
                if len(batch) == batch_size:
                    break
            if len(batch) < batch_size:
                break
            batches.append(batch)
        return batches

    def _prepare_training_batch(self, dataset):
        if not dataset:
            return [], None

        processed = []
        path_tensor = []
        missing_path = []

        for idx, entry in enumerate(dataset):
            pyg_data, distances, metadata, _, path = self._unpack_entry(entry)
            if path is None:
                missing_path.append(idx)
                continue

            path_t = torch.as_tensor(path, device=self.device, dtype=torch.long)
            if path_t.dim() == 1:
                path_t = path_t.unsqueeze(1)
            if path_t.dim() != 2:
                raise ValueError("path must be a 1D or 2D tensor.")

            pyg_base = pyg_data.to(self.device) if self.problem_type == "TSPTW" else pyg_data
            rho = metadata.get("rho", None)
            u_matrix = metadata.get("u_matrix", None)
            v_matrix = metadata.get("v_matrix", None)
            if self.problem_type == "TSPTW":
                pyg_mod = self._apply_consensus_state(
                    pyg_base,
                    rho=rho,
                    u_matrix=u_matrix,
                    v_matrix=v_matrix,
                )
            else:
                pyg_mod = pyg_base

            meta_j = dict(metadata)
            gid = meta_j.get("group_id", None)
            if gid is not None and not torch.is_tensor(gid):
                gid = torch.tensor(gid)
            if gid is not None:
                meta_j["group_id"] = gid

            processed.append((pyg_mod, distances, meta_j))
            path_tensor.append(path_t)

        if missing_path:
            raise ValueError(f"Training buffer entries must include path. Missing indices: {missing_path}")
        if not processed:
            return [], None

        path_tensor = torch.stack(path_tensor, dim=1)
        return processed, path_tensor

    def _train_gfn_batch(
            self,
            model,
            decoder,
            optimizer,
            dataset,
            generate,
            alpha,
            beta,
            reward_mode,
            tw_mask,
            use_pomo,
            u_state=None,
            v_state=None,
        ):
        model.train()
        if decoder is not None:
            decoder.train()
        loss_mode = str(self.model_params.get("loss", "gflow")).lower()

        if not dataset:
            return None, {
                "loss": 0.0,
                "R_mean": 0.0,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "route_infeas": 0.0,
                "cost_mean": 0.0,
                "adv_min": 0.0,
                "adv_max": 0.0,
                "forward_flow": 0.0,
                "flow_gap_mean": 0.0,
                "entropy_term": 0.0,
                "penalty_term": 0.0,
                "infeas_norm": 0.0,
                "tw_late": 0.0,
            }

        processed, paths = self._prepare_training_batch(dataset)
        batch_pyg, batch_distances, metadata_batch = self._collate_train_batch(processed)
        B = batch_distances.size(0)
        if B == 0:
            return None, {}
        if paths is None:
            raise ValueError("Training requires buffered path entries.")
        group_ids = metadata_batch.get("group_id", None)
        group_ids = group_ids.to(self.device) if group_ids is not None else None

        tw_kwargs = self._build_tw_kwargs(metadata_batch)
        dyn_scale = self._resolve_dyn_scale(metadata_batch)

        if paths is not None:
            if paths.dim() == 2:
                paths = paths.unsqueeze(2)
            G = paths.size(2)
        else:
            G = generate

        # GNN forward
        if self.use_single_model:
            feats = self._extract_single_features(batch_pyg)
            node_emb = model.encoder(None, feats)
            searcher_gen_eval = search(
                batch_distances,
                G,
                heuristic=None,
                device=self.device,
                decoder=decoder,
                node_emb=node_emb,
                tw_mask=tw_mask,
                dyn_scale=dyn_scale,
                use_pomo=use_pomo,
                heuristic_output_space=self.args.gen_matrix_output_space,
                **tw_kwargs,
            )
            flow = torch.zeros(B, device=self.device)
            consensus_score_matrix = None
        elif self.args.decoding_type == "nar":
            her, flows = model(batch_pyg, True)
            flow = flows.squeeze()
            mat = model.reshape(batch_pyg, her) + EPS
            consensus_score_matrix = mat
            searcher_gen_eval = search(
                batch_distances,
                G,
                heuristic=mat,
                heuristic_target=mat,
                device=self.device,
                tw_mask=tw_mask,
                use_pomo=use_pomo,
                heuristic_output_space=self.args.gen_matrix_output_space,
                **tw_kwargs,
            )
        else:
            her, flows, node_emb_flat = model(batch_pyg, return_logZ=True, return_node_emb=True)
            node_emb = model.reshape_nodes(batch_pyg, node_emb_flat)
            consensus_score_matrix = None
            searcher_gen_eval = search(
                batch_distances,
                G,
                heuristic=None,
                device=self.device,
                decoder=decoder,
                node_emb=node_emb,
                tw_mask=tw_mask,
                dyn_scale=dyn_scale,
                use_pomo=use_pomo,
                heuristic_output_space=self.args.gen_matrix_output_space,
                **tw_kwargs,
            )
            flow = flows.squeeze()

        start_override = self._pomo_start_nodes(batch_distances, G, use_pomo)
        start_node = start_override if use_pomo else START_NODE
        if paths is None:
            paths, forward = searcher_gen_eval.get_route(
                alpha=alpha,
                require_prob=True,
                start_node=start_node,
                desi=0,
            )
        else:
            _, forward = searcher_gen_eval.get_route(
                alpha=alpha,
                require_prob=True,
                paths=paths,
                start_node=start_node,
                desi=0,
            )

        costs = searcher_gen_eval.get_costs(paths)
        costs_flat = costs.reshape(-1)
        if paths.dim() == 2:
            paths = paths.unsqueeze(2)
        tw_lateness = self._compute_tw_lateness_penalty_batch(paths, metadata_batch)
        tw_penalty_mean = tw_lateness.mean().item() if tw_lateness.numel() > 0 else 0.0

        if reward_mode == "tw":
            objective_primary_term = self.tw_penalty_weight_gflow * tw_lateness
            objective_primary_label = "objective_tw_term"
        else:
            if self.cost_penalty_type == "delta_rate":
                objective_primary_term = self.cost_penalty_weight_rate * costs
                objective_primary_label = "objective_cost_rate_term"
            else:
                objective_primary_term = self.cost_penalty_weight * costs
                objective_primary_label = "objective_cost_term"

        rho_batch = metadata_batch.get("rho", None)
        if rho_batch is None:
            default_rho = self.rho_z_init if reward_mode == "tw" else self.rho_x_init
            rho_batch = torch.full((B,), default_rho, device=costs.device, dtype=costs.dtype)
        else:
            rho_batch = rho_batch.to(device=costs.device, dtype=costs.dtype).view(-1)
        n_nodes = paths.size(0)
        if u_state is None:
            u_state_tensor = torch.zeros((B, self.occ_time_bins, n_nodes, n_nodes), device=costs.device, dtype=costs.dtype)
        else:
            u_state_tensor = torch.as_tensor(u_state, device=costs.device, dtype=costs.dtype)
        if v_state is None:
            v_state_tensor = torch.zeros((B, self.occ_time_bins, n_nodes, n_nodes), device=costs.device, dtype=costs.dtype)
        else:
            v_state_tensor = torch.as_tensor(v_state, device=costs.device, dtype=costs.dtype)
        project_to_window = reward_mode == "tw"

        consensus_penalty = self._consensus_penalty(
            paths=paths,
            rho=rho_batch,
            u_state=u_state_tensor,
            v_state=v_state_tensor,
            metadata_batch=metadata_batch,
            project_to_window=project_to_window,
        )
        occ_aux_loss = self._occupancy_aux_loss(
            consensus_score_matrix,
            paths,
            metadata_batch,
            rho_batch,
            u_state_tensor,
            v_state_tensor,
            project_to_window=project_to_window,
        )
        objective_total = objective_primary_term + consensus_penalty
        reward_raw = -objective_total

        raw_infeas, normalized_infeas, _ = self._infeasibility_penalty_batch(
            paths, metadata_batch, target_device=costs.device
        )
        infeas_norm_mean = normalized_infeas.mean().item() if normalized_infeas.numel() > 0 else 0.0

        if self.args.baseline_type == "per_instance":
            if group_ids is not None:
                reward_entry = reward_raw.mean(dim=1)
                baseline = self._group_mean_by_key(reward_entry, group_ids)
                reward_centered = reward_raw - baseline.unsqueeze(1)
            else:
                reward_centered = reward_raw - reward_raw.mean(dim=1, keepdim=True)
        elif self.args.baseline_type == "per_batch":
            reward_centered = reward_raw - reward_raw.mean()
        else:
            reward_centered = reward_raw

        combined_signal = -reward_centered.reshape(-1)
        combined_std = combined_signal.std(unbiased=False)
        adv_flat = combined_signal if math.isfinite(combined_std.item()) else combined_signal
        R_flat = reward_raw.reshape(-1)

        forward_logp = forward.sum(0)
        logZ = flow
        if logZ.dim() == 0:
            logZ = logZ.unsqueeze(0)
        if logZ.dim() == 1:
            logZ = logZ.unsqueeze(1)
        logZ = logZ.expand_as(forward_logp)
        forward_flow = forward_logp + logZ
        forward_flow_flat = forward_flow.reshape(-1)

        entropy_term = math.log(1)
        penalty_term_flat = (adv_flat.detach()) * beta
        backward_flow_flat = entropy_term - penalty_term_flat
        flow_gap_flat = forward_flow_flat - backward_flow_flat

        if loss_mode == "reinforce":
            log_prob_flat = forward.sum(0).reshape(-1)
            reward_flat = (-adv_flat).detach()
            centered_reward = reward_flat
            loss = -(centered_reward * log_prob_flat).mean()
        else:
            loss = torch.pow(forward_flow_flat - backward_flow_flat, 2).mean()
        loss = loss + (self.occ_aux_weight * occ_aux_loss)

        G_eff = paths.size(2)
        if raw_infeas.numel() == B * G_eff:
            violation_matrix = raw_infeas.view(B, G_eff)
        else:
            violation_matrix = self._compute_tw_violation_counts_batch(paths, metadata_batch)
        node_infeas_per_inst = violation_matrix.mean(dim=1)
        route_infeas_per_inst = (violation_matrix > 0).float().mean(dim=1)

        if group_ids is not None:
            unique_keys, inverse = torch.unique(group_ids, sorted=False, return_inverse=True)
            group_count = unique_keys.numel()
            denom = max(int(group_count), 1)
            sums_node = torch.zeros(group_count, device=violation_matrix.device, dtype=violation_matrix.dtype)
            sums_route = torch.zeros(group_count, device=violation_matrix.device, dtype=violation_matrix.dtype)
            counts = torch.zeros(group_count, device=violation_matrix.device, dtype=violation_matrix.dtype)
            sums_node.scatter_add_(0, inverse, node_infeas_per_inst)
            sums_route.scatter_add_(0, inverse, route_infeas_per_inst)
            counts.scatter_add_(0, inverse, torch.ones_like(route_infeas_per_inst))
            node_infeas_total = float((sums_node / counts.clamp_min(1.0)).sum().item())
            route_infeas_total = float((sums_route / counts.clamp_min(1.0)).sum().item())
        else:
            denom = max(B, 1)
            node_infeas_total = float(node_infeas_per_inst.sum().item())
            route_infeas_total = float(route_infeas_per_inst.sum().item())
        adv_min = float(adv_flat.min().item()) if adv_flat.numel() > 0 else 0.0
        adv_max = float(adv_flat.max().item()) if adv_flat.numel() > 0 else 0.0
        penalty_term_mean = float((-penalty_term_flat).mean().item()) if penalty_term_flat.numel() > 0 else 0.0
        flow_gap_mean = float(flow_gap_flat.pow(2).mean().sqrt().item()) if flow_gap_flat.numel() > 0 else 0.0
        reward_flat = reward_raw.reshape(-1).detach()
        reward_total_mean = reward_raw.mean().item() if reward_raw.numel() > 0 else 0.0
        objective_primary_mean = objective_primary_term.mean().item() if objective_primary_term.numel() > 0 else 0.0
        consensus_penalty_mean = consensus_penalty.mean().item() if consensus_penalty.numel() > 0 else 0.0
        objective_total_mean = objective_total.mean().item() if objective_total.numel() > 0 else 0.0
        metrics = {
            "loss": float(loss.item()),
            "R_mean": float(R_flat.mean().item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "route_infeas": float(route_infeas_total / denom),
            "cost_mean": float(costs_flat.mean().item()),
            "adv_min": adv_min,
            "adv_max": adv_max,
            "forward_flow": float(forward_flow_flat.mean().item()),
            "flow_gap_mean": flow_gap_mean,
            "entropy_term": float(entropy_term),
            "penalty_term": penalty_term_mean,
            "infeas_norm": float(infeas_norm_mean),
            "tw_late": float(tw_penalty_mean),
            "objective_mean": float(objective_total_mean),
            "consensus_penalty": float(consensus_penalty_mean),
            "occ_aux_loss": float(occ_aux_loss.item()),
            "reward_total": float(reward_total_mean),
            "reward_mean": float(reward_flat.mean().item()) if reward_flat.numel() > 0 else 0.0,
            "reward_std": float(reward_flat.std(unbiased=False).item()) if reward_flat.numel() > 0 else 0.0,
            "loss_mode": loss_mode,
        }
        metrics[objective_primary_label] = float(objective_primary_mean)
        return loss, metrics

    def generator_train(
            self,
            dataset,
            generate,
            alpha,
            beta,
            u_state=None,
            v_state=None,
        ):
        return self._train_gfn_batch(
            self.net,
            self.decoder,
            self.optimizer,
            dataset,
            generate,
            alpha,
            beta,
            reward_mode="cost",
            tw_mask=self.args.tw_mask_train,
            use_pomo=self.args.train_use_pomo,
            u_state=u_state,
            v_state=v_state,
        )

    def repair_train(
            self,
            dataset,
            generate,
            alpha,
            beta,
            u_state=None,
            v_state=None,
        ):
        return self._train_gfn_batch(
            self.rep_net,
            self.rep_decoder,
            self.rep_optimizer,
            dataset,
            generate,
            alpha,
            beta,
            reward_mode="tw",
            tw_mask=self.args.tw_mask_train,
            use_pomo=self.args.train_use_pomo,
            u_state=u_state,
            v_state=v_state,
        )

    @staticmethod
    def _average_metrics(metrics_list: list[dict]) -> dict:
        if not metrics_list:
            return {}
        acc = {}
        list_series = {}
        list_last = {}
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    acc.setdefault(key, []).append(float(value))
                elif isinstance(value, (list, tuple, np.ndarray)):
                    if key in ("cost_mean", "tw_late"):
                        vals = [float(v) if v is not None else float("nan") for v in value]
                        list_series.setdefault(key, []).append(vals)
                    else:
                        list_last[key] = list(value)
        averaged = {key: float(np.mean(vals)) for key, vals in acc.items()}
        for key, series in list_series.items():
            if not series:
                continue
            max_len = max(len(row) for row in series)
            if max_len == 0:
                averaged[key] = []
                continue
            arr = np.full((len(series), max_len), np.nan, dtype=np.float32)
            for i, row in enumerate(series):
                arr[i, :len(row)] = row
            with np.errstate(all="ignore"):
                averaged[key] = [float(v) for v in np.nanmean(arr, axis=0)]
        for key, value in list_last.items():
            if key not in averaged:
                averaged[key] = value
        for key in ("loss_mode",):
            if key in metrics_list[-1]:
                averaged[key] = metrics_list[-1][key]
        return averaged

    def _build_buffer_entries(
        self,
        base_data,
        paths,
        step_idx: int,
        step_count: int,
        rho=None,
        u_matrix=None,
        v_matrix=None,
    ):
        entries = []
        if not base_data or paths is None:
            return entries
        _, batch_size, _ = paths.shape
        for i in range(batch_size):
            pyg_data, distances, metadata = base_data[i]
            meta_device = distances.device if torch.is_tensor(distances) else self.device
            base_gid = metadata.get("group_id", i)
            if torch.is_tensor(base_gid):
                base_gid = int(base_gid.item())
            group_id = int(base_gid) * int(step_count) + int(step_idx)
            meta = dict(metadata)
            meta["group_id"] = torch.tensor(group_id, dtype=torch.long, device=meta_device)
            if rho is not None:
                rho_val = rho[i] if torch.is_tensor(rho) else rho
                meta["rho"] = torch.as_tensor(rho_val, device=meta_device).detach()
            if u_matrix is not None:
                u_val = u_matrix[i] if torch.is_tensor(u_matrix) else u_matrix
                if torch.is_tensor(u_val) and u_val.dim() == 3:
                    u_val = self._occupancy_state_summary(u_val.unsqueeze(0)).squeeze(0)
                meta["u_matrix"] = torch.as_tensor(u_val, device=meta_device).detach()
            if v_matrix is not None:
                v_val = v_matrix[i] if torch.is_tensor(v_matrix) else v_matrix
                if torch.is_tensor(v_val) and v_val.dim() == 3:
                    v_val = self._occupancy_state_summary(v_val.unsqueeze(0)).squeeze(0)
                meta["v_matrix"] = torch.as_tensor(v_val, device=meta_device).detach()
            entries.append({
                "pyg_data": pyg_data,
                "distances": distances,
                "metadata": meta,
                "path": paths[:, i, :].detach(),
            })
        return entries

    @torch.no_grad()
    def _summarize_rollout_stats(self, paths, distances_batch, metadata_batch):
        if paths is None:
            return float("nan"), float("nan")
        paths_t = paths if torch.is_tensor(paths) else torch.as_tensor(paths)
        if paths_t.dim() == 2:
            paths_t = paths_t.unsqueeze(2)
        paths_dev = paths_t.to(self.device)
        dist_dev = distances_batch if torch.is_tensor(distances_batch) else torch.as_tensor(distances_batch, device=self.device)
        if dist_dev.device != paths_dev.device:
            dist_dev = dist_dev.to(paths_dev.device)
        cost = self._compute_route_costs(dist_dev, paths_dev)
        tw_late = self._compute_tw_lateness_penalty_batch(paths_dev, metadata_batch)
        cost_mean = cost.mean().item() if cost.numel() > 0 else 0.0
        tw_mean = tw_late.mean().item() if tw_late.numel() > 0 else 0.0
        return cost_mean, tw_mean

    @torch.no_grad()
    def _sample_model_paths(
        self,
        model,
        decoder,
        base_data,
        generate: int,
        tw_mask: str,
        use_pomo: bool,
        alpha: float,
        desi: int,
        offload: bool = False,
        rho: torch.Tensor | None = None,
        u_matrix: torch.Tensor | None = None,
        v_matrix: torch.Tensor | None = None,
        return_log_probs: bool = False,
        paths_override: torch.Tensor | None = None,
        batch_cache: dict | None = None,
    ):
        if not base_data:
            return None
        batch_size = len(base_data)
        n_nodes = base_data[0][1].size(0)

        data_list = []
        cache_distances = None
        cache_metadata = None
        cache_pyg = None
        cache_batch_pyg = None
        if batch_cache is not None:
            cache_distances = batch_cache.get("distances", None)
            cache_metadata = batch_cache.get("metadata", None)
            cache_pyg = batch_cache.get("pyg", None)
            cache_batch_pyg = batch_cache.get("batch_pyg", None)
        if cache_distances is None or cache_metadata is None:
            fallback_cache = self._build_base_batch_cache(base_data, device=self.device, include_pyg=False)
            cache_distances = fallback_cache["distances"]
            cache_metadata = fallback_cache["metadata"]

        for i in range(batch_size):
            pyg_data = base_data[i][0]
            if cache_pyg is not None and i < len(cache_pyg):
                pyg_base = cache_pyg[i]
            else:
                pyg_x = getattr(pyg_data, "x", None)
                if torch.is_tensor(pyg_x) and pyg_x.device != self.device:
                    pyg_base = pyg_data.to(self.device)
                else:
                    pyg_base = pyg_data
            if self.problem_type == "TSPTW":
                rho_i = rho[i] if (rho is not None and torch.is_tensor(rho)) else rho
                u_i = u_matrix[i] if (u_matrix is not None and torch.is_tensor(u_matrix)) else u_matrix
                v_i = v_matrix[i] if (v_matrix is not None and torch.is_tensor(v_matrix)) else v_matrix
                if torch.is_tensor(u_i) and u_i.dim() == 3:
                    u_i = self._occupancy_state_summary(u_i.unsqueeze(0)).squeeze(0)
                if torch.is_tensor(v_i) and v_i.dim() == 3:
                    v_i = self._occupancy_state_summary(v_i.unsqueeze(0)).squeeze(0)
                pyg_mod = self._apply_consensus_state(
                    pyg_base,
                    rho=rho_i,
                    u_matrix=u_i,
                    v_matrix=v_i,
                )
            else:
                pyg_mod = pyg_base
            data_list.append(pyg_mod)

        batch_distances = cache_distances
        metadata_batch = cache_metadata

        search_generate = generate
        if self.problem_type != "TSPTW" and cache_batch_pyg is not None:
            batch_pyg = cache_batch_pyg
        else:
            batch_pyg = Batch.from_data_list(data_list)
            batch_x = getattr(batch_pyg, "x", None)
            if torch.is_tensor(batch_x) and batch_x.device != self.device:
                batch_pyg = batch_pyg.to(self.device)
        tw_kwargs = self._build_tw_kwargs(metadata_batch)
        dyn_scale = self._resolve_dyn_scale(metadata_batch)

        if self.use_single_model:
            feats = self._extract_single_features(batch_pyg)
            node_emb = model.encoder(None, feats)
            searcher_eval = search(
                batch_distances,
                search_generate,
                heuristic=None,
                device=self.device,
                decoder=decoder,
                node_emb=node_emb,
                tw_mask=tw_mask,
                dyn_scale=dyn_scale,
                use_pomo=use_pomo,
                heuristic_output_space=self.args.gen_matrix_output_space,
                **tw_kwargs,
            )
        elif self.args.decoding_type == "nar":
            her = model(batch_pyg)
            mat = model.reshape(batch_pyg, her) + EPS
            searcher_eval = search(
                batch_distances,
                search_generate,
                heuristic=mat,
                heuristic_target=mat,
                device=self.device,
                tw_mask=tw_mask,
                use_pomo=use_pomo,
                heuristic_output_space=self.args.gen_matrix_output_space,
                **tw_kwargs,
            )
        else:
            _, node_emb_flat = model(batch_pyg, return_node_emb=True)
            node_emb = model.reshape_nodes(batch_pyg, node_emb_flat)
            searcher_eval = search(
                batch_distances,
                search_generate,
                heuristic=None,
                device=self.device,
                decoder=decoder,
                node_emb=node_emb,
                tw_mask=tw_mask,
                dyn_scale=dyn_scale,
                use_pomo=use_pomo,
                heuristic_output_space=self.args.gen_matrix_output_space,
                **tw_kwargs,
            )

        start_override = self._pomo_start_nodes(batch_distances, search_generate, use_pomo)
        start_node = start_override if use_pomo else START_NODE
        if paths_override is not None:
            target_paths = paths_override
            if target_paths.dim() == 2:
                target_paths = target_paths.unsqueeze(2)
            if target_paths.dim() != 3:
                raise ValueError("paths_override must be (N, B, G), (N, B), or (N, BG).")
            if target_paths.size(0) != n_nodes:
                raise ValueError("paths_override node dimension mismatch.")
            target_paths = target_paths.to(self.device)
            result = searcher_eval.get_route(
                alpha=alpha,
                require_prob=return_log_probs,
                start_node=start_node,
                paths=target_paths,
                desi=desi,
            )
        else:
            result = searcher_eval.get_route(
                alpha=alpha,
                require_prob=return_log_probs,
                start_node=start_node,
                desi=desi,
            )

        if return_log_probs:
            paths_out, log_probs = result
        else:
            paths_out = result[0] if isinstance(result, (tuple, list)) else result
            log_probs = None
        if paths_out.dim() == 2:
            paths_out = paths_out.unsqueeze(1)
        if paths_out.dim() != 3:
            raise ValueError("Unexpected paths_out shape.")
        paths_out = paths_out.detach()
        if log_probs is not None:
            log_probs = log_probs.detach()
            if log_probs.dim() == 3 and log_probs.size(2) == 1:
                log_probs = log_probs.squeeze(2)
            if log_probs.dim() == 2:
                if log_probs.size(1) != batch_size * generate:
                    raise ValueError("log_probs size does not match batch_size * generate.")
                log_probs = log_probs.view(n_nodes - 1, batch_size, generate)
            elif log_probs.dim() == 3:
                if log_probs.size(1) != batch_size or log_probs.size(2) != generate:
                    raise ValueError("Unexpected log_probs shape for sampled paths.")
        if offload:
            paths_out = paths_out.cpu()
            if log_probs is not None:
                log_probs = log_probs.cpu()
        return (paths_out, log_probs) if return_log_probs else paths_out

    @torch.no_grad()
    def _select_best_z(
        self,
        candidates,
        metadata_batch,
        distances_batch=None,
        rho=None,
        u_state=None,
        v_state=None,
        return_idx: bool = False,
    ):
        if candidates is None:
            return candidates
        cand = candidates
        if cand.dim() == 2:
            if return_idx:
                _, batch_size = cand.shape
                best_idx = torch.zeros((batch_size,), device=cand.device, dtype=torch.long)
                return cand, best_idx
            return cand
        if cand.dim() != 3:
            raise ValueError("candidates must be (N, B, G)")
        original_device = cand.device
        if cand.device != self.device:
            cand = cand.to(self.device)
        n_nodes, batch_size, generate = cand.shape
        dist = distances_batch
        if torch.is_tensor(dist) and dist.device != self.device:
            dist = dist.to(self.device)
        tw_late = self._compute_tw_lateness_penalty_batch(cand, metadata_batch)
        if self.select_best_mode == "best_r":
            if rho is None or u_state is None or v_state is None:
                raise ValueError("best_R mode requires rho, u_state, and v_state.")
            rho_tensor = torch.as_tensor(rho, device=cand.device, dtype=cand.dtype).view(-1)
            u_tensor = torch.as_tensor(u_state, device=cand.device, dtype=cand.dtype)
            v_tensor = torch.as_tensor(v_state, device=cand.device, dtype=cand.dtype)
            consensus = self._consensus_penalty(
                paths=cand,
                rho=rho_tensor,
                u_state=u_tensor,
                v_state=v_tensor,
                metadata_batch=metadata_batch,
                project_to_window=True,
            )
            objective = tw_late + consensus
            best_idx = objective.argmin(dim=1)
        else:
            costs = self._compute_route_costs(dist, cand) if dist is not None else None
            costs_for_select = costs if costs is not None else torch.zeros_like(tw_late)
            best_idx = self._best_feasible_min_cost_idx(tw_late, costs_for_select, dim=1)

        gather_idx = best_idx.view(1, batch_size, 1).expand(n_nodes, -1, 1)
        best = cand.gather(2, gather_idx).squeeze(2)
        if best.device != original_device:
            best = best.to(original_device)
        if return_idx:
            if best_idx.device != original_device:
                best_idx = best_idx.to(original_device)
            return best, best_idx
        return best

    @torch.no_grad()
    def _select_min_cost_tours(self, paths, distances_batch):
        if paths is None:
            return paths
        if paths.dim() == 2:
            return paths
        if paths.dim() != 3:
            raise ValueError("paths must be (N, B, G)")
        n_nodes, batch_size, _ = paths.shape
        dist = distances_batch
        if torch.is_tensor(dist) and dist.device != paths.device:
            dist = dist.to(paths.device)
        costs = self._compute_route_costs(dist, paths)
        best_idx = costs.argmin(dim=1)
        gather_idx = best_idx.view(1, batch_size, 1).expand(n_nodes, -1, 1)
        best = paths.gather(2, gather_idx).squeeze(2)
        return best

    @staticmethod
    def _occupancy_state_summary(state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 3:
            return state
        if state.dim() != 4:
            raise ValueError("Occupancy state must be (B, H, N, N) or (B, N, N).")
        return state.sum(dim=1)

    def _resolve_occ_horizon_batch(self, metadata_batch: dict, device=None) -> torch.Tensor:
        if metadata_batch is None:
            raise ValueError("metadata_batch is required for occupancy consensus.")
        target_device = self.device if device is None else device
        tw_end = torch.as_tensor(metadata_batch["tw_end"], device=target_device, dtype=torch.float32) / GEN_SCALE
        if tw_end.dim() == 1:
            tw_end = tw_end.unsqueeze(0)
        return tw_end.max(dim=1).values.clamp_min(1.0)

    @torch.no_grad()
    def _route_step_indices_and_bins(
        self,
        paths: torch.Tensor,
        metadata_batch: dict,
        project_to_window: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if paths.dim() == 2:
            paths = paths.unsqueeze(2)
        tours = paths.long().to(self.device)
        if tours.dim() != 3:
            raise ValueError("paths must be (N, B) or (N, B, G).")

        n_nodes, batch_size, generate = tours.shape
        coords = torch.as_tensor(metadata_batch["coordinates"], device=self.device, dtype=torch.float32) / GEN_SCALE
        service_time = torch.as_tensor(metadata_batch["service_time"], device=self.device, dtype=torch.float32) / GEN_SCALE
        tw_start = torch.as_tensor(metadata_batch["tw_start"], device=self.device, dtype=torch.float32) / GEN_SCALE
        tw_end = torch.as_tensor(metadata_batch["tw_end"], device=self.device, dtype=torch.float32) / GEN_SCALE
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if service_time.dim() == 1:
            service_time = service_time.unsqueeze(0)
        if tw_start.dim() == 1:
            tw_start = tw_start.unsqueeze(0)
        if tw_end.dim() == 1:
            tw_end = tw_end.unsqueeze(0)
        if coords.size(0) != batch_size:
            raise ValueError("metadata batch size does not match paths batch size.")

        travel_matrix = metadata_batch.get("_coord_dist", None) if isinstance(metadata_batch, dict) else None
        if travel_matrix is None or not torch.is_tensor(travel_matrix) or travel_matrix.device != self.device:
            travel_matrix = torch.cdist(coords, coords, p=2)
            if isinstance(metadata_batch, dict):
                metadata_batch["_coord_dist"] = travel_matrix

        horizon = self._resolve_occ_horizon_batch(metadata_batch, device=self.device).view(batch_size, 1)
        batch_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1).expand(batch_size, generate)
        start_node = tours[0]
        current_node = start_node
        current_time = torch.clamp(tw_start[batch_idx, start_node], min=0.0) + service_time[batch_idx, start_node]

        def _service_to_bin(service_begin: torch.Tensor, dest_node: torch.Tensor) -> torch.Tensor:
            normalized = (service_begin / horizon).clamp(min=0.0, max=1.0 - EPS)
            bin_idx = torch.floor(normalized * self.occ_time_bins).long().clamp(0, self.occ_time_bins - 1)
            if project_to_window:
                lo = torch.floor(
                    (tw_start[batch_idx, dest_node] / horizon).clamp(min=0.0, max=1.0 - EPS) * self.occ_time_bins
                ).long().clamp(0, self.occ_time_bins - 1)
                hi = torch.ceil(
                    (tw_end[batch_idx, dest_node] / horizon).clamp(min=0.0, max=1.0) * self.occ_time_bins
                ).long() - 1
                hi = hi.clamp(0, self.occ_time_bins - 1)
                hi = torch.maximum(hi, lo)
                bin_idx = torch.minimum(torch.maximum(bin_idx, lo), hi)
            return bin_idx

        src_steps = []
        dst_steps = []
        bin_steps = []
        for step_idx in range(1, n_nodes):
            next_node = tours[step_idx]
            travel = travel_matrix[batch_idx, current_node, next_node]
            arrival = current_time + travel
            service_begin = torch.maximum(arrival, tw_start[batch_idx, next_node])
            src_steps.append(current_node)
            dst_steps.append(next_node)
            bin_steps.append(_service_to_bin(service_begin, next_node))
            current_time = service_begin + service_time[batch_idx, next_node]
            current_node = next_node

        return_node = start_node
        travel_back = travel_matrix[batch_idx, current_node, return_node]
        arrival_back = current_time + travel_back
        service_back = torch.maximum(arrival_back, tw_start[batch_idx, return_node])
        src_steps.append(current_node)
        dst_steps.append(return_node)
        bin_steps.append(_service_to_bin(service_back, return_node))

        return (
            torch.stack(src_steps, dim=0),
            torch.stack(dst_steps, dim=0),
            torch.stack(bin_steps, dim=0),
        )

    @torch.no_grad()
    def _materialize_occupancy_tensor(
        self,
        paths: torch.Tensor,
        metadata_batch: dict,
        project_to_window: bool = False,
    ) -> torch.Tensor:
        if paths.dim() == 2:
            paths = paths.unsqueeze(2)
        src, dst, bin_idx = self._route_step_indices_and_bins(paths, metadata_batch, project_to_window=project_to_window)
        n_nodes, batch_size, generate = paths.shape
        flat_size = self.occ_time_bins * n_nodes * n_nodes
        occ = torch.zeros((batch_size, generate, flat_size), device=self.device, dtype=torch.float32)
        flat_idx = (bin_idx * (n_nodes * n_nodes) + src * n_nodes + dst).permute(1, 2, 0)
        values = torch.ones_like(flat_idx, dtype=occ.dtype)
        occ.scatter_add_(2, flat_idx, values)
        occ = occ.view(batch_size, generate, self.occ_time_bins, n_nodes, n_nodes)
        if generate == 1:
            return occ.squeeze(1)
        return occ

    def _project_occupancy_state(self, occ_state: torch.Tensor, metadata_batch: dict) -> torch.Tensor:
        state = occ_state.to(self.device)
        if state.dim() != 4:
            raise ValueError("Occupancy projection expects state with shape (B, H, N, N).")
        batch_size = state.size(0)
        tw_start = torch.as_tensor(metadata_batch["tw_start"], device=self.device, dtype=state.dtype) / GEN_SCALE
        tw_end = torch.as_tensor(metadata_batch["tw_end"], device=self.device, dtype=state.dtype) / GEN_SCALE
        if tw_start.dim() == 1:
            tw_start = tw_start.unsqueeze(0)
        if tw_end.dim() == 1:
            tw_end = tw_end.unsqueeze(0)
        horizon = self._resolve_occ_horizon_batch(metadata_batch, device=self.device).to(dtype=state.dtype)
        centers = ((torch.arange(self.occ_time_bins, device=self.device, dtype=state.dtype) + 0.5) / self.occ_time_bins)
        centers = centers.view(1, self.occ_time_bins, 1) * horizon.view(batch_size, 1, 1)
        valid = (centers >= tw_start.unsqueeze(1)) & (centers <= tw_end.unsqueeze(1))
        return state * valid.unsqueeze(2).to(dtype=state.dtype)

    @torch.no_grad()
    def _consensus_penalty(
        self,
        paths: torch.Tensor,
        rho: torch.Tensor,
        u_state: torch.Tensor,
        v_state: torch.Tensor,
        metadata_batch: dict,
        project_to_window: bool = False,
    ) -> torch.Tensor:
        if paths.dim() == 2:
            paths = paths.unsqueeze(2)
        src, dst, bin_idx = self._route_step_indices_and_bins(paths, metadata_batch, project_to_window=project_to_window)
        n_nodes, batch_size, generate = paths.shape
        transition_count = src.size(0)

        rho_tensor = torch.as_tensor(rho, device=self.device, dtype=torch.float32).view(-1)
        u_tensor = torch.as_tensor(u_state, device=self.device, dtype=torch.float32)
        v_tensor = torch.as_tensor(v_state, device=self.device, dtype=torch.float32)
        if u_tensor.dim() == 3:
            u_tensor = u_tensor.unsqueeze(0)
        if v_tensor.dim() == 3:
            v_tensor = v_tensor.unsqueeze(0)
        if u_tensor.dim() != 4 or v_tensor.dim() != 4:
            raise ValueError("Consensus state must be (B, H, N, N) or (H, N, N).")
        if u_tensor.size(0) != batch_size or v_tensor.size(0) != batch_size:
            raise ValueError("Consensus state batch size does not match paths batch size.")

        rho_view = rho_tensor.view(-1, 1, 1, 1).clamp_min(EPS)
        state_tensor = v_tensor + (u_tensor / rho_view)
        base_norm = state_tensor.pow(2).sum(dim=(1, 2, 3))
        flat_state = state_tensor.view(batch_size, -1)
        flat_idx = (bin_idx * (n_nodes * n_nodes) + src * n_nodes + dst).permute(1, 2, 0).reshape(batch_size, -1)
        gathered = flat_state.gather(1, flat_idx).view(batch_size, generate, transition_count)
        return 0.5 * rho_tensor.view(-1, 1) * (base_norm.view(-1, 1) + float(transition_count) - 2.0 * gathered.sum(dim=2))

    def _occupancy_aux_loss(
        self,
        score_matrix: torch.Tensor | None,
        paths: torch.Tensor,
        metadata_batch: dict,
        rho: torch.Tensor,
        u_state: torch.Tensor,
        v_state: torch.Tensor,
        project_to_window: bool = False,
    ) -> torch.Tensor:
        if score_matrix is None:
            return torch.zeros((), device=self.device)
        if paths.dim() == 2:
            paths = paths.unsqueeze(2)
        score_tensor = self._align_matrix_output_space(
            score_matrix,
            from_space=self.args.gen_matrix_output_space,
            to_space="probs",
        ).to(self.device)
        if score_tensor.dim() != 3:
            raise ValueError("score_matrix must be (B, N, N).")

        src, dst, bin_idx = self._route_step_indices_and_bins(paths, metadata_batch, project_to_window=project_to_window)
        n_nodes, batch_size, generate = paths.shape
        transition_count = src.size(0)
        rho_tensor = torch.as_tensor(rho, device=self.device, dtype=score_tensor.dtype).view(-1)
        u_tensor = torch.as_tensor(u_state, device=self.device, dtype=score_tensor.dtype)
        v_tensor = torch.as_tensor(v_state, device=self.device, dtype=score_tensor.dtype)
        if u_tensor.dim() == 3:
            u_tensor = u_tensor.unsqueeze(0)
        if v_tensor.dim() == 3:
            v_tensor = v_tensor.unsqueeze(0)
        rho_view = rho_tensor.view(-1, 1, 1, 1).clamp_min(EPS)
        target_tensor = v_tensor + (u_tensor / rho_view)
        if project_to_window:
            target_tensor = self._project_occupancy_state(target_tensor, metadata_batch)
        target_tensor = target_tensor.detach()

        flat_score_idx = (src * n_nodes + dst).permute(1, 2, 0).reshape(batch_size, -1)
        route_scores = score_tensor.view(batch_size, -1).gather(1, flat_score_idx).view(batch_size, generate, transition_count)
        flat_target_idx = (bin_idx * (n_nodes * n_nodes) + src * n_nodes + dst).permute(1, 2, 0).reshape(batch_size, -1)
        route_targets = target_tensor.view(batch_size, -1).gather(1, flat_target_idx).view(batch_size, generate, transition_count)
        weighted_sq = 0.5 * rho_tensor.view(-1, 1, 1) * (route_scores - route_targets).pow(2)
        return weighted_sq.mean()

    @torch.no_grad()
    def _update_global_consensus(
        self,
        rho_x: torch.Tensor,
        rho_z: torch.Tensor,
        u_x: torch.Tensor,
        u_z: torch.Tensor,
        y_x: torch.Tensor,
        y_z: torch.Tensor,
    ) -> torch.Tensor:
        state_rank = y_x.dim()
        if state_rank == 3:
            view_shape = (-1, 1, 1)
        elif state_rank == 4:
            view_shape = (-1, 1, 1, 1)
        else:
            raise ValueError("Consensus tensors must be 3D or 4D.")
        rho_x_view = rho_x.view(*view_shape)
        rho_z_view = rho_z.view(*view_shape)
        denom = (rho_x + rho_z).view(*view_shape).clamp_min(EPS)
        numer = (rho_x_view * y_x - u_x) + (rho_z_view * y_z - u_z)
        return numer / denom

    @staticmethod
    def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        den_safe = torch.where(den.abs() < EPS, torch.full_like(den, EPS), den)
        return num / den_safe

    @torch.no_grad()
    def _update_consensus_weights(
        self,
        delta_y: torch.Tensor,
        delta_hat_u: torch.Tensor,
        delta_u: torch.Tensor,
        delta_v: torch.Tensor,
        rho_curr: torch.Tensor,
    ) -> torch.Tensor:
        dy = delta_y.reshape(delta_y.size(0), -1)
        dhu = delta_hat_u.reshape(delta_hat_u.size(0), -1)
        du = delta_u.reshape(delta_u.size(0), -1)
        dv = delta_v.reshape(delta_v.size(0), -1)

        dy_dhu = (dy * dhu).sum(dim=1)
        dy_dy = (dy * dy).sum(dim=1)
        dhu_dhu = (dhu * dhu).sum(dim=1)

        alpha_sd = self._safe_div(dhu_dhu, dy_dhu)
        alpha_mg = self._safe_div(dy_dhu, dy_dy)
        alpha_hat = torch.where(2.0 * alpha_mg > alpha_sd, alpha_mg, alpha_sd - 0.5 * alpha_mg)

        dv_du = (dv * du).sum(dim=1)
        dv_dv = (dv * dv).sum(dim=1)
        du_du = (du * du).sum(dim=1)

        beta_sd = self._safe_div(du_du, dv_du)
        beta_mg = self._safe_div(dv_du, dv_dv)
        beta_hat = torch.where(2.0 * beta_mg > beta_sd, beta_mg, beta_sd - 0.5 * beta_mg)

        alpha_den = torch.linalg.norm(dy, dim=1) * torch.linalg.norm(dhu, dim=1)
        beta_den = torch.linalg.norm(dv, dim=1) * torch.linalg.norm(du, dim=1)
        alpha_cor = self._safe_div(dy_dhu, alpha_den.clamp_min(EPS))
        beta_cor = self._safe_div(dv_du, beta_den.clamp_min(EPS))

        eps_cor = self.consensus_correlation_threshold
        both_good = (alpha_cor > eps_cor) & (beta_cor > eps_cor)
        alpha_only = (alpha_cor > eps_cor) & (beta_cor <= eps_cor)
        beta_only = (alpha_cor <= eps_cor) & (beta_cor > eps_cor)

        rho_candidate = torch.where(
            both_good,
            torch.sqrt((alpha_hat.clamp_min(EPS) * beta_hat.clamp_min(EPS)).clamp_min(EPS)),
            rho_curr,
        )
        rho_candidate = torch.where(alpha_only, alpha_hat, rho_candidate)
        rho_candidate = torch.where(beta_only, beta_hat, rho_candidate)
        rho_candidate = torch.where(torch.isfinite(rho_candidate), rho_candidate, rho_curr)
        return rho_candidate.clamp_min(EPS)

    @torch.no_grad()
    def _select_best_x(
        self,
        paths,
        distances_batch,
        metadata_batch,
        rho=None,
        u_state=None,
        v_state=None,
        return_idx: bool = False,
    ):
        if paths is None:
            return paths
        if paths.dim() == 2:
            if return_idx:
                _, batch_size = paths.shape
                best_idx = torch.zeros((batch_size,), device=paths.device, dtype=torch.long)
                return paths, best_idx
            return paths
        if paths.dim() != 3:
            raise ValueError("paths must be (N, B, G)")
        original_device = paths.device
        if paths.device != self.device:
            paths = paths.to(self.device)
        n_nodes, batch_size, generate = paths.shape
        dist = distances_batch
        if torch.is_tensor(dist) and dist.device != paths.device:
            dist = dist.to(paths.device)
        costs = self._compute_route_costs(dist, paths)

        if self.select_best_mode == "best_r":
            if rho is None or u_state is None or v_state is None:
                raise ValueError("best_R mode requires rho, u_state, and v_state.")
            rho_tensor = torch.as_tensor(rho, device=paths.device, dtype=paths.dtype).view(-1)
            u_tensor = torch.as_tensor(u_state, device=paths.device, dtype=paths.dtype)
            v_tensor = torch.as_tensor(v_state, device=paths.device, dtype=paths.dtype)
            consensus = self._consensus_penalty(
                paths=paths,
                rho=rho_tensor,
                u_state=u_tensor,
                v_state=v_tensor,
                metadata_batch=metadata_batch,
                project_to_window=False,
            )
            objective = costs + consensus
            best_idx = objective.argmin(dim=1)
        else:
            tw_late = self._compute_tw_lateness_penalty_batch(paths, metadata_batch)
            best_idx = self._best_feasible_min_cost_idx(tw_late, costs, dim=1)

        gather_idx = best_idx.view(1, batch_size, 1).expand(n_nodes, -1, 1)
        best = paths.gather(2, gather_idx).squeeze(2)
        if best.device != original_device:
            best = best.to(original_device)
        if return_idx:
            if best_idx.device != original_device:
                best_idx = best_idx.to(original_device)
            return best, best_idx
        return best

    def gen_pyg_data(self, node_feature, distances, k_sparse, time_window=None, distances_return=None):
        n_nodes = len(node_feature)
        device = node_feature.device

        if n_nodes <= 1:
            dummy_index = torch.zeros((2, 1), dtype=torch.long, device=device)
            edge_attr = torch.zeros((1, 1), device=device)
            pyg_data = Data(x=node_feature, edge_index=dummy_index, edge_attr=edge_attr)
            return pyg_data, distances

        max_neighbors = max(n_nodes - 1, 1)
        k_effective = max(1, min(k_sparse, max_neighbors))

        if self.problem_type == "TSPTW" and time_window is not None:
            tw_start = time_window[:, 0]
            tw_end = time_window[:, 1]
            overlap = torch.minimum(tw_end[:, None], tw_end[None, :]) - torch.maximum(tw_start[:, None], tw_start[None, :])
            diag_mask = torch.eye(n_nodes, device=device, dtype=overlap.dtype) * 1e9
            overlap = overlap - diag_mask
            topk_indices = torch.topk(overlap, k=k_effective, dim=1, largest=True).indices
            topk_values = distances[torch.arange(n_nodes, device=device).unsqueeze(1), topk_indices]
        else:
            masked_distances = distances.clone()
            masked_distances[torch.arange(n_nodes, device=device), torch.arange(n_nodes, device=device)] = float('inf')
            topk_values, topk_indices = torch.topk(masked_distances,
                                                   k=k_effective,
                                                   dim=1, largest=False)

        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(n_nodes, device=device),
                                    repeats=k_effective),
            torch.flatten(topk_indices)
            ])
        edge_attr = topk_values.reshape(-1, 1)
        pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
        return pyg_data, (distances if distances_return is None else distances_return)
    

    def _compose_node_features(
        self,
        coords,
        time_window,
        service_time,
        depot_index,
        rho=None,
        u_matrix=None,
        v_matrix=None,
    ):
        if self.problem_type == "TSPTW":
            if time_window is None or service_time is None:
                raise ValueError("Time window and service time features are required for TSPTW instances.")
            if service_time.dim() == 1:
                service_time = service_time.unsqueeze(-1)
            features = [coords, time_window]
            depot_indicator = torch.zeros((coords.shape[0], 1), device=coords.device, dtype=coords.dtype)
            if depot_index is not None:
                depot_indicator[depot_index, 0] = 1.0
            features.append(depot_indicator)
            if rho is None:
                rho_feat = torch.zeros((coords.shape[0], 1), device=coords.device, dtype=coords.dtype)
            else:
                rho_tensor = torch.as_tensor(rho, device=coords.device, dtype=coords.dtype)
                if rho_tensor.numel() == 1:
                    rho_feat = rho_tensor.view(1, 1).expand(coords.shape[0], 1)
                else:
                    rho_tensor = rho_tensor.view(-1, 1)
                    if rho_tensor.size(0) != coords.shape[0]:
                        raise ValueError("rho must be scalar or per-node vector.")
                    rho_feat = rho_tensor
            features.append(rho_feat)
            if u_matrix is None:
                u_feat = torch.zeros((coords.shape[0], self.problem_size), device=coords.device, dtype=coords.dtype)
            else:
                u_tensor = torch.as_tensor(u_matrix, device=coords.device, dtype=coords.dtype)
                if u_tensor.dim() == 3 and u_tensor.size(0) == 1:
                    u_tensor = u_tensor.squeeze(0)
                if u_tensor.dim() != 2:
                    raise ValueError("u_matrix must be (N, N) for a single instance.")
                if u_tensor.shape[0] != coords.shape[0] or u_tensor.shape[1] != self.problem_size:
                    raise ValueError("u_matrix shape must be (N, problem_size).")
                u_feat = u_tensor
            features.append(u_feat)
            if v_matrix is None:
                v_feat = torch.zeros((coords.shape[0], self.problem_size), device=coords.device, dtype=coords.dtype)
            else:
                v_tensor = torch.as_tensor(v_matrix, device=coords.device, dtype=coords.dtype)
                if v_tensor.dim() == 3 and v_tensor.size(0) == 1:
                    v_tensor = v_tensor.squeeze(0)
                if v_tensor.dim() != 2:
                    raise ValueError("v_matrix must be (N, N) for a single instance.")
                if v_tensor.shape[0] != coords.shape[0] or v_tensor.shape[1] != self.problem_size:
                    raise ValueError("v_matrix shape must be (N, problem_size).")
                v_feat = v_tensor
            features.append(v_feat)
            return torch.cat(features, dim=1)
        return coords

    def _apply_consensus_state(self, pyg_data, rho=None, u_matrix=None, v_matrix=None):
        x = pyg_data.x
        if x.size(1) < self.node_feature_dim:
            pad = torch.zeros((x.size(0), self.node_feature_dim - x.size(1)), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        x_new = x.clone()
        if rho is not None and self.rho_feature_idx is not None:
            rho_tensor = torch.as_tensor(rho, device=x_new.device, dtype=x_new.dtype)
            if rho_tensor.numel() == 1:
                x_new[:, self.rho_feature_idx] = rho_tensor.view(1)
            else:
                rho_flat = rho_tensor.view(-1)
                if rho_flat.numel() == x_new.size(0):
                    x_new[:, self.rho_feature_idx] = rho_flat
                else:
                    x_new[:, self.rho_feature_idx] = rho_flat[0]
        if u_matrix is not None and self.u_feature_start is not None:
            u_tensor = torch.as_tensor(u_matrix, device=x_new.device, dtype=x_new.dtype)
            if u_tensor.dim() == 3 and u_tensor.size(0) == 1:
                u_tensor = u_tensor.squeeze(0)
            if u_tensor.dim() != 2:
                raise ValueError("u_matrix must be (N, N) for a single instance.")
            if u_tensor.shape[0] != x_new.size(0) or u_tensor.shape[1] != self.problem_size:
                raise ValueError("u_matrix shape must be (N, problem_size).")
            x_new[:, self.u_feature_start:self.u_feature_start + self.problem_size] = u_tensor
        if v_matrix is not None and self.v_feature_start is not None:
            v_tensor = torch.as_tensor(v_matrix, device=x_new.device, dtype=x_new.dtype)
            if v_tensor.dim() == 3 and v_tensor.size(0) == 1:
                v_tensor = v_tensor.squeeze(0)
            if v_tensor.dim() != 2:
                raise ValueError("v_matrix must be (N, N) for a single instance.")
            if v_tensor.shape[0] != x_new.size(0) or v_tensor.shape[1] != self.problem_size:
                raise ValueError("v_matrix shape must be (N, problem_size).")
            x_new[:, self.v_feature_start:self.v_feature_start + self.problem_size] = v_tensor
        return Data(x=x_new, edge_index=pyg_data.edge_index, edge_attr=pyg_data.edge_attr)

    @torch.no_grad()
    def _sample_uniform_tours(self, batch_size: int, n_nodes: int) -> torch.Tensor:
        if batch_size <= 0:
            return torch.zeros((n_nodes, 0), device=self.device, dtype=torch.long)
        if n_nodes <= 1:
            return torch.zeros((n_nodes, batch_size), device=self.device, dtype=torch.long)
        rand = torch.rand((batch_size, n_nodes - 1), device=self.device)
        perm = rand.argsort(dim=1) + 1
        start = torch.zeros((batch_size, 1), device=self.device, dtype=torch.long)
        tours = torch.cat([start, perm], dim=1)  # (B, N)
        return tours.transpose(0, 1)  # (N, B)

    @torch.no_grad()
    def _sample_uniform_tours_batch(self, batch_size: int, n_nodes: int, generate: int) -> torch.Tensor:
        if generate <= 0:
            return torch.zeros((n_nodes, batch_size, 0), device=self.device, dtype=torch.long)
        total = batch_size * generate
        tours = self._sample_uniform_tours(total, n_nodes)
        return tours.view(n_nodes, batch_size, generate)

    @staticmethod
    def _compute_route_costs(distances: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
        if distances.dim() == 2:
            u = paths
            v = torch.roll(paths, shifts=1, dims=0)
            return distances[u, v].sum(dim=0)
        if distances.dim() == 3:
            B, _, _ = distances.shape
            u = paths.permute(1, 2, 0)  # (B, G, N)
            v = torch.roll(u, shifts=1, dims=-1)
            batch_idx = torch.arange(B, device=distances.device).view(B, 1, 1)
            return distances[batch_idx, u, v].sum(dim=-1)
        raise ValueError("distances must be 2D or 3D.")

    def _generate_dummy_time_window(self, node_tensor):
        if not torch.is_tensor(node_tensor):
            node_tensor = torch.as_tensor(node_tensor, dtype=torch.float32)
        batch_size, n_nodes, _ = node_tensor.shape
        base = node_tensor.new_zeros((batch_size, n_nodes))
        service_time = base.clone()
        tw_start = base.clone()
        tw_end = base.clone().fill_(TSP_FAKE_TW_END)
        return service_time, tw_start, tw_end

    def _get_problem_tensors(self, batch_size):
        if self.problem_type == "TSPTW":
            return self.env.get_random_problems(batch_size, self.env_params["problem_size"])
        node = self.env.get_random_problems(batch_size, self.env_params["problem_size"])
        service_time, tw_start, tw_end = self._generate_dummy_time_window(node)
        return node, service_time, tw_start, tw_end

    def _maybe_augment_instances(self, node, service_time, tw_start, tw_end, use_aug: bool):
        """
        Augment coordinates with 8-fold symmetry if requested.
        Assumes coords are raw; we normalize by COORD_SCALE before applying transforms.
        """
        batch_size = node.size(0)
        base_ids = torch.arange(batch_size, device=node.device)
        if not use_aug:
            return node, service_time, tw_start, tw_end, base_ids

        coords_norm = node / COORD_SCALE
        aug_coords_norm = self._augment_xy_8_fold(coords_norm)
        aug_factor = aug_coords_norm.size(0) // coords_norm.size(0)
        B, N = batch_size, coords_norm.size(1)

        # (8, B, N, 2) -> (B, 8, N, 2) -> (8B, N, 2)  (instance-major)
        aug_coords_norm = (
            aug_coords_norm.view(aug_factor, B, N, 2)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(B * aug_factor, N, 2)
        )
        
        aug_node = aug_coords_norm * COORD_SCALE
        aug_service = service_time.repeat_interleave(aug_factor, dim=0)
        aug_tw_start = tw_start.repeat_interleave(aug_factor, dim=0)
        aug_tw_end = tw_end.repeat_interleave(aug_factor, dim=0)
        group_ids = base_ids.repeat_interleave(aug_factor)
        variant_ids = torch.arange(aug_factor, device=node.device).repeat(batch_size)
        return aug_node, aug_service, aug_tw_start, aug_tw_end, group_ids

    def _standardize_dataset(self, dataset):
        if torch.is_tensor(dataset):
            service_time, tw_start, tw_end = self._generate_dummy_time_window(dataset)
            return dataset, service_time, tw_start, tw_end
        if isinstance(dataset, (list, tuple)) and len(dataset) == 4:
            return dataset
        raise ValueError("Unsupported dataset format for validation data.")

    def _resolve_tw_scale(self, tw_end, device=None):
        if TIME_WINDOW_SCALE is not None:
            return float(TIME_WINDOW_SCALE)
        if tw_end is None:
            return 1.0
        if torch.is_tensor(tw_end):
            scale = tw_end[..., 0].float()
            if device is not None:
                scale = scale.to(device)
            return torch.clamp(scale, min=EPS)
        scale_val = np.asarray(tw_end, dtype=np.float32).reshape(-1)[0]
        return max(float(scale_val), EPS)

    def _resolve_tw_scale_batch(self, tw_end: torch.Tensor, device=None) -> torch.Tensor:
        if not torch.is_tensor(tw_end):
            tw_end = torch.as_tensor(tw_end, dtype=torch.float32)
        tw_end = tw_end.float()
        if device is not None:
            tw_end = tw_end.to(device)
        if TIME_WINDOW_SCALE is not None:
            return torch.full((tw_end.size(0),), float(TIME_WINDOW_SCALE), device=tw_end.device, dtype=tw_end.dtype)
        if tw_end.dim() == 1:
            base = tw_end
        else:
            base = tw_end[:, 0]
        return torch.clamp(base, min=EPS)

    def _resolve_dyn_scale(self, metadata):
        if TIME_WINDOW_SCALE is not None:
            return GEN_SCALE / TIME_WINDOW_SCALE
        if metadata is None:
            return 1.0
        tw_end = metadata.get("tw_end")
        if tw_end is None:
            return 1.0
        tw_scale = self._resolve_tw_scale(tw_end, device=self.device)
        if torch.is_tensor(tw_scale):
            dyn_scale = GEN_SCALE / tw_scale
            if dyn_scale.dim() == 1:
                return dyn_scale.view(-1, 1, 1)
            return dyn_scale
        return GEN_SCALE / tw_scale

    def _align_matrix_output_space(self, matrix: torch.Tensor, from_space: str, to_space: str) -> torch.Tensor:
        if from_space == to_space:
            return matrix
        if from_space == "logit" and to_space == "probs":
            return torch.sigmoid(matrix)
        if from_space == "probs" and to_space == "logit":
            clipped = matrix.clamp(min=EPS, max=1.0 - EPS)
            return torch.log(clipped / (1.0 - clipped))
        return matrix

    def _build_tw_kwargs(self, metadata):
        if metadata is None:
            return {}
        return {
            "tw_start": metadata["tw_start"].to(self.device).float() / GEN_SCALE,
            "tw_end": metadata["tw_end"].to(self.device).float() / GEN_SCALE,
            "service_time": metadata["service_time"].to(self.device).float() / GEN_SCALE,
        }

    
    @torch.no_grad()
    def train_data(self, rollout_batch_size, k_sparse):
        node, service_time, tw_e, tw_l = self._get_problem_tensors(rollout_batch_size)
        node, service_time, tw_e, tw_l, group_ids = self._maybe_augment_instances(
            node, service_time, tw_e, tw_l, use_aug=self.args.train_aug
        )
        train_data = []
        total_instances = node.shape[0]
        cpu_device = torch.device("cpu")
        node_cpu = node.to(cpu_device)
        service_cpu = service_time.to(cpu_device)
        tw_e_cpu = tw_e.to(cpu_device)
        tw_l_cpu = tw_l.to(cpu_device)

        node_xy_model_all = node_cpu / COORD_SCALE
        node_xy_gen_all = node_cpu / GEN_SCALE
        distances_model_all = torch.cdist(node_xy_model_all, node_xy_model_all, p=2)
        distances_gen_all = torch.cdist(node_xy_gen_all, node_xy_gen_all, p=2)
        diag_idx = torch.arange(node_xy_model_all.size(1), device=cpu_device)
        distances_model_all[:, diag_idx, diag_idx] = 1e9
        distances_gen_all[:, diag_idx, diag_idx] = 1e9
        tw_scale_all = self._resolve_tw_scale_batch(tw_l_cpu, device=cpu_device)
        time_window_all = torch.stack([tw_e_cpu, tw_l_cpu], dim=-1) / tw_scale_all[:, None, None]
        service_time_norm_all = service_cpu / tw_scale_all[:, None]

        for i in range(total_instances):
            node_xy_model = node_xy_model_all[i]
            distances_model = distances_model_all[i]
            distances_gen = distances_gen_all[i]
            time_window = time_window_all[i]
            service_time_norm = service_time_norm_all[i]
            node_feature = self._compose_node_features(node_xy_model, time_window, service_time_norm, self.start_node)
            pyg_data, sparse_dist = self.gen_pyg_data(node_feature, distances_model, k_sparse, time_window=time_window, distances_return=distances_gen)
            metadata = {
                "coordinates": node_cpu[i].detach().clone(),
                "service_time": service_cpu[i].detach().clone(),
                "tw_start": tw_e_cpu[i].detach().clone(),
                "tw_end": tw_l_cpu[i].detach().clone(),
                "group_id": group_ids[i].detach().cpu().clone(),
            }
            # print("coordinates_norm:", node_xy)
            # print("time_window_norm:", time_window)
            # break
            train_data.append((pyg_data, sparse_dist, metadata))
        return train_data
            
    def load_val_data(self, dataset, k_sparse, start_node=None):
        node, service_time, time_e, time_l = self._standardize_dataset(dataset)
        node, service_time, time_e, time_l, group_ids = self._maybe_augment_instances(
            node, service_time, time_e, time_l, use_aug=self.args.val_aug
        )
        val_data = []
        idx = 0
        diag_idx = torch.arange(node.size(1), device=self.device)
        while idx < len(node):
            # Gather all variants for the same original instance when augmentation is enabled
            current_gid = group_ids[idx].item() if group_ids is not None else idx
            variant_indices = [j for j in range(idx, len(node)) if group_ids[j].item() == current_gid] if self.args.val_aug else [idx]
            variants = []
            for j in variant_indices:
                node_xy_raw = node[j].float()
                service_time_raw = service_time[j].float()
                tw_start_raw = time_e[j].float()
                tw_end_raw = time_l[j].float()

                node_xy_model = node_xy_raw.to(self.device) / COORD_SCALE
                node_xy_gen = node_xy_raw.to(self.device) / GEN_SCALE
                distances_model = torch.cdist(node_xy_model, node_xy_model, p=2)
                distances_model[diag_idx, diag_idx] = 1e9
                distances_gen = torch.cdist(node_xy_gen, node_xy_gen, p=2)
                distances_gen[diag_idx, diag_idx] = 1e9
                tw_start = tw_start_raw.to(self.device)
                tw_end = tw_end_raw.to(self.device)
                tw_scale = self._resolve_tw_scale(tw_end, device=self.device)
                time_window = torch.cat([tw_start.unsqueeze(-1), tw_end.unsqueeze(-1)], dim=1) / tw_scale
                service_time_norm = service_time_raw.to(self.device) / tw_scale
                node_feature = self._compose_node_features(node_xy_model, time_window, service_time_norm, self.start_node if start_node is None else start_node)
                data, distances = self.gen_pyg_data(node_feature, distances_model, k_sparse=k_sparse, time_window=time_window, distances_return=distances_gen)
                metadata = {
                    "coordinates": node_xy_raw,
                    "service_time": service_time_raw,
                    "tw_start": tw_start_raw,
                    "tw_end": tw_end_raw,
                    "group_id": torch.tensor(current_gid),
                }
                variants.append((data, distances, metadata))
            if self.args.val_aug:
                val_data.append({"variants": variants, "group_id": current_gid})
            else:
                val_data.append(variants[0])
            idx = variant_indices[-1] + 1
        return val_data
            
    @staticmethod
    def _rotate_tour_to_start(tour: np.ndarray, start_node: int):
        indices = np.where(tour == start_node)[0]
        if indices.size == 0:
            return tour.copy(), int(tour[0])
        idx = int(indices[0])
        rotated = np.concatenate([tour[idx:], tour[:idx]])
        return rotated, start_node

    @staticmethod
    def _tour_feasibility_core(tour: np.ndarray,
                               coords: np.ndarray,
                               service_time: np.ndarray,
                               tw_start: np.ndarray,
                               tw_end: np.ndarray,
                               tolerance: float = 1e-5) -> tuple[float, int]:
        depot = 0 if np.any(tour == 0) else int(tour[0])
        ordered_tour, depot = Trainer._rotate_tour_to_start(tour, depot)
        ordered_tour = ordered_tour.astype(int, copy=False)
        coords = coords.astype(float, copy=False)
        service_time = service_time.astype(float, copy=False)
        tw_start = tw_start.astype(float, copy=False)
        tw_end = tw_end.astype(float, copy=False)

        violation_count = 0
        current_time = max(tw_start[depot], 0.0) + service_time[depot]
        prev = depot
        for node in ordered_tour[1:]:
            travel = np.linalg.norm(coords[prev] - coords[node])
            arrival = current_time + travel
            if arrival > tw_end[node] + tolerance:
                violation_count += 1
            current_time = max(arrival, tw_start[node]) + service_time[node]
            prev = node

        travel_to_depot = np.linalg.norm(coords[prev] - coords[depot])
        arrival_depot = current_time + travel_to_depot
        if arrival_depot > tw_end[depot] + tolerance:
            violation_count += 1

        is_feasible = 1.0 if violation_count == 0 else 0.0
        return is_feasible, violation_count

    @staticmethod
    def _is_tour_feasible(tour: np.ndarray,
                          coords: np.ndarray,
                          service_time: np.ndarray,
                          tw_start: np.ndarray,
                          tw_end: np.ndarray,
                          tolerance: float = 1e-5) -> float:
        feasible, _ = Trainer._tour_feasibility_core(
            tour= tour,
            coords= coords,
            service_time= service_time,
            tw_start= tw_start,
            tw_end= tw_end,
            tolerance= tolerance,
        )
        return feasible

    @staticmethod
    def _collect_route_violation_data(paths: torch.Tensor, metadata: dict, tolerance: float = 1e-5):
        if metadata is None:
            return [], []

        tours = paths if isinstance(paths, torch.Tensor) else torch.as_tensor(paths)
        if tours.dim() == 3:
            if tours.size(1) != 1:
                raise ValueError("_collect_route_violation_data expects single-instance tours shaped (N, G).")
            tours = tours[:, 0, :]
        tours = tours.long()
        if tours.dim() != 2:
            tours = tours.reshape(tours.size(0), -1)
        n_nodes, num_tours = tours.shape
        if num_tours == 0:
            return [], []

        device = tours.device
        coords = torch.as_tensor(metadata["coordinates"], device=device, dtype=torch.float32) / GEN_SCALE
        service_time = torch.as_tensor(metadata["service_time"], device=device, dtype=torch.float32) / GEN_SCALE
        tw_start = torch.as_tensor(metadata["tw_start"], device=device, dtype=torch.float32) / GEN_SCALE
        tw_end = torch.as_tensor(metadata["tw_end"], device=device, dtype=torch.float32) / GEN_SCALE
        travel_matrix = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0), p=2).squeeze(0)

        depot = 0
        current_node = torch.full((num_tours,), depot, dtype=torch.long, device=device)
        depot_time = torch.clamp(tw_start[depot], min=0.0) + service_time[depot]
        current_time = depot_time.repeat(num_tours)
        violation_counts = torch.zeros((num_tours,), dtype=torch.float32, device=device)

        for idx in range(1, n_nodes):
            next_node = tours[idx]
            travel = travel_matrix[current_node, next_node]
            arrival = current_time + travel
            violation_counts += (arrival > (tw_end[next_node] + tolerance)).float()
            service_begin = torch.maximum(arrival, tw_start[next_node])
            current_time = service_begin + service_time[next_node]
            current_node = next_node

        travel_back = travel_matrix[current_node, depot]
        arrival_back = current_time + travel_back
        violation_counts += (arrival_back > (tw_end[depot] + tolerance)).float()

        infeasible_flags = (violation_counts > 0).float()
        return violation_counts.detach().cpu().tolist(), infeasible_flags.detach().cpu().tolist()

    def _compute_tw_violation_counts_batch(
        self,
        paths: torch.Tensor,
        metadata_batch: dict,
        tolerance: float = 1e-5,
    ) -> torch.Tensor:
        if metadata_batch is None:
            if paths.dim() == 2:
                return torch.zeros((1, paths.size(1)), device=self.device)
            if paths.dim() == 3:
                return torch.zeros((paths.size(1), paths.size(2)), device=self.device)
            raise ValueError("paths must be 2D or 3D.")

        tours = paths if isinstance(paths, torch.Tensor) else torch.as_tensor(paths, device=self.device)
        tours = tours.long().to(self.device)
        if tours.dim() == 2:
            tours = tours.unsqueeze(1)
        if tours.dim() != 3:
            raise ValueError("paths must be (N, G) or (N, B, G).")
        N, B, G = tours.shape
        if B == 0 or G == 0:
            return torch.zeros((B, G), device=self.device)

        coords = torch.as_tensor(metadata_batch["coordinates"], device=self.device, dtype=torch.float32) / GEN_SCALE
        service_time = torch.as_tensor(metadata_batch["service_time"], device=self.device, dtype=torch.float32) / GEN_SCALE
        tw_start = torch.as_tensor(metadata_batch["tw_start"], device=self.device, dtype=torch.float32) / GEN_SCALE
        tw_end = torch.as_tensor(metadata_batch["tw_end"], device=self.device, dtype=torch.float32) / GEN_SCALE

        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if service_time.dim() == 1:
            service_time = service_time.unsqueeze(0)
        if tw_start.dim() == 1:
            tw_start = tw_start.unsqueeze(0)
        if tw_end.dim() == 1:
            tw_end = tw_end.unsqueeze(0)
        if coords.size(0) != B:
            raise ValueError("metadata batch size does not match paths batch size.")
        travel_matrix = metadata_batch.get("_coord_dist", None) if isinstance(metadata_batch, dict) else None
        if travel_matrix is None or not torch.is_tensor(travel_matrix) or travel_matrix.device != self.device:
            travel_matrix = torch.cdist(coords, coords, p=2)
            if isinstance(metadata_batch, dict):
                metadata_batch["_coord_dist"] = travel_matrix

        depot = 0
        batch_idx = torch.arange(B, device=self.device).view(B, 1).expand(B, G)
        current_node = torch.full((B, G), depot, dtype=torch.long, device=self.device)
        depot_time = torch.clamp(tw_start[:, depot], min=0.0) + service_time[:, depot]
        current_time = depot_time.view(B, 1).expand(B, G).clone()
        violations = torch.zeros((B, G), device=self.device, dtype=torch.float32)

        for idx in range(1, N):
            next_node = tours[idx]
            travel = travel_matrix[batch_idx, current_node, next_node]
            arrival = current_time + travel
            tw_end_next = tw_end[batch_idx, next_node]
            violations += (arrival > (tw_end_next + tolerance)).float()
            tw_start_next = tw_start[batch_idx, next_node]
            service_begin = torch.maximum(arrival, tw_start_next)
            service_next = service_time[batch_idx, next_node]
            current_time = service_begin + service_next
            current_node = next_node

        travel_back = travel_matrix[batch_idx, current_node, depot]
        arrival_back = current_time + travel_back
        tw_end_depot = tw_end[:, depot].view(B, 1).expand(B, G)
        violations += (arrival_back > (tw_end_depot + tolerance)).float()
        return violations

    def _infeasibility_penalty(self, paths: torch.Tensor, metadata: dict, target_device, tolerance: float = 1e-5):
        tours = paths if isinstance(paths, torch.Tensor) else torch.as_tensor(paths)
        if tours.dim() == 3:
            if tours.size(1) != 1:
                raise ValueError("_infeasibility_penalty expects single-instance paths.")
            tours = tours[:, 0, :]
        if tours.dim() == 1:
            tours = tours.unsqueeze(1)
        metadata_single = {
            "coordinates": torch.as_tensor(metadata["coordinates"]).unsqueeze(0),
            "service_time": torch.as_tensor(metadata["service_time"]).unsqueeze(0),
            "tw_start": torch.as_tensor(metadata["tw_start"]).unsqueeze(0),
            "tw_end": torch.as_tensor(metadata["tw_end"]).unsqueeze(0),
        }
        violation_counts = self._compute_tw_violation_counts_batch(tours, metadata_single, tolerance=tolerance).squeeze(0)
        violation_tensor = violation_counts.to(device=target_device, dtype=torch.float32).reshape(-1)
        normalized = violation_tensor
        return violation_tensor, normalized, float(violation_tensor.mean().item() if violation_tensor.numel() > 0 else 0.0)
    
    def _infeasibility_penalty_batch(
        self,
        paths: torch.Tensor,
        metadata_batch: dict,
        target_device,
        tolerance: float = 1e-5,
    ):
        """
        Batched wrapper around _infeasibility_penalty.
        paths: (N, B, G)
        metadata_batch: dict with tensors of shape (B, N)
        """
        if paths.dim() != 3:
            return self._infeasibility_penalty(paths, metadata_batch, target_device, tolerance)

        violations = self._compute_tw_violation_counts_batch(paths, metadata_batch, tolerance=tolerance)
        raw_tensor = violations.reshape(-1).to(device=target_device, dtype=torch.float32)
        norm_tensor = raw_tensor
        avg_violation = float(violations.mean().item() if violations.numel() > 0 else 0.0)
        return raw_tensor, norm_tensor, avg_violation

    def _compute_tw_lateness_penalty(self, paths: torch.Tensor, metadata: dict):
        if metadata is None:
            tours_tensor = paths if isinstance(paths, torch.Tensor) else torch.as_tensor(paths)
            num_tours = tours_tensor.shape[1] if tours_tensor.ndim >= 2 else tours_tensor.shape[0]
            return torch.zeros(num_tours, device=self.device)

        def _to_device_tensor(key, scale):
            return metadata[key].to(self.device).float() / scale

        coords = metadata["coordinates"].to(self.device).float() / GEN_SCALE
        service_time = _to_device_tensor("service_time", GEN_SCALE)
        tw_start = _to_device_tensor("tw_start", GEN_SCALE)
        tw_end = _to_device_tensor("tw_end", GEN_SCALE)
        travel_matrix = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0), p=2).squeeze(0)

        tours = paths if isinstance(paths, torch.Tensor) else torch.as_tensor(paths, device=self.device)
        tours = tours.long().to(self.device)
        n_nodes, num_tours = tours.shape
        depot = 0
        current_node = torch.full((num_tours,), depot, dtype=torch.long, device=self.device)
        depot_time = torch.clamp(tw_start[depot], min=0.0) + service_time[depot]
        current_time = depot_time.repeat(num_tours)
        lateness = torch.zeros(num_tours, device=self.device)

        for idx in range(1, n_nodes):
            next_node = tours[idx]
            travel = travel_matrix[current_node, next_node]
            arr = current_time + travel
            service_begin = torch.maximum(arr, tw_start[next_node])
            lateness += torch.clamp(service_begin - tw_end[next_node], min=0.0)
            current_time = service_begin + service_time[next_node]
            current_node = next_node

        travel_back = travel_matrix[current_node, depot]
        arr_back = current_time + travel_back
        service_back = torch.maximum(arr_back, tw_start[depot])
        lateness += torch.clamp(service_back - tw_end[depot], min=0.0)
        return lateness
    

    def _compute_tw_lateness_penalty_batch(self, paths: torch.Tensor, metadata_batch: dict):
        """
        Batched wrapper for _compute_tw_lateness_penalty.
        paths: (N, B, G)
        """
        if paths.dim() != 3:
            return self._compute_tw_lateness_penalty(paths, metadata_batch)
        if metadata_batch is None:
            return torch.zeros((paths.size(1), paths.size(2)), device=self.device)

        tours = paths if isinstance(paths, torch.Tensor) else torch.as_tensor(paths, device=self.device)
        tours = tours.long().to(self.device)
        N, B, G = tours.shape
        if B == 0 or G == 0:
            return torch.zeros((B, G), device=self.device)

        coords = metadata_batch["coordinates"].to(self.device).float() / GEN_SCALE
        service_time = metadata_batch["service_time"].to(self.device).float() / GEN_SCALE
        tw_start = metadata_batch["tw_start"].to(self.device).float() / GEN_SCALE
        tw_end = metadata_batch["tw_end"].to(self.device).float() / GEN_SCALE
        travel_matrix = metadata_batch.get("_coord_dist", None) if isinstance(metadata_batch, dict) else None
        if travel_matrix is None or not torch.is_tensor(travel_matrix) or travel_matrix.device != self.device:
            travel_matrix = torch.cdist(coords, coords, p=2)
            if isinstance(metadata_batch, dict):
                metadata_batch["_coord_dist"] = travel_matrix

        depot = 0
        batch_idx = torch.arange(B, device=self.device).view(B, 1).expand(B, G)
        current_node = torch.full((B, G), depot, dtype=torch.long, device=self.device)
        depot_time = torch.clamp(tw_start[:, depot], min=0.0) + service_time[:, depot]
        current_time = depot_time.view(B, 1).expand(B, G).clone()
        lateness = torch.zeros((B, G), device=self.device)

        for idx in range(1, N):
            next_node = tours[idx]
            travel = travel_matrix[batch_idx, current_node, next_node]
            arr = current_time + travel
            tw_start_next = tw_start[batch_idx, next_node]
            tw_end_next = tw_end[batch_idx, next_node]
            service_begin = torch.maximum(arr, tw_start_next)
            lateness += torch.clamp(service_begin - tw_end_next, min=0.0)
            service_next = service_time[batch_idx, next_node]
            current_time = service_begin + service_next
            current_node = next_node

        travel_back = travel_matrix[batch_idx, current_node, depot]
        arr_back = current_time + travel_back
        tw_start_depot = tw_start[:, depot].view(B, 1).expand(B, G)
        tw_end_depot = tw_end[:, depot].view(B, 1).expand(B, G)
        service_back = torch.maximum(arr_back, tw_start_depot)
        lateness += torch.clamp(service_back - tw_end_depot, min=0.0)
        return lateness

    @staticmethod
    def _summarize_route_infeasibility(paths: torch.Tensor, metadata: dict, tolerance: float = 1e-5) -> tuple[float, float]:
        violation_counts, infeasible_flags = Trainer._collect_route_violation_data(paths, metadata, tolerance)
        if not violation_counts:
            return 0.0, 0.0

        violation_counts = np.asarray(violation_counts, dtype=np.float32)
        infeasible_flags = np.asarray(infeasible_flags, dtype=np.float32)
        avg_node_infeas = float(violation_counts.mean())
        route_infeas_ratio = float(infeasible_flags.mean())
        return avg_node_infeas, route_infeas_ratio

    @staticmethod
    def _normalize_cost(cost_value):
        if cost_value is None:
            return None
        cost_float = float(cost_value)
        if math.isnan(cost_float):
            return None
        return cost_float / GEN_SCALE

    @staticmethod
    def _format_metric_value(key: str, value) -> str:
        formatters = {
            "loss": "{:.4f}",
            "R_mean": "{:.3f}",
            "lr": "{:.2e}",
            "route_infeas": "{:.2f}",
            "pair_count": "{:.0f}",
            "route_count": "{:.0f}",
            "feas_avg": "{:.2f}",
            "infeas_avg": "{:.2f}",
            "cost_mean": "{:.3f}",
            "adv_min": "{:.3f}",
            "adv_max": "{:.3f}",
            "forward_flow": "{:.3f}",
            "flow_gap_mean": "{:.3f}",
            "entropy_term": "{:.3f}",
            "penalty_term": "{:.3f}",
            "infeas_norm": "{:.3f}",
            "tw_late": "{:.3f}",
            "objective_mean": "{:.3f}",
            "consensus_penalty": "{:.3f}",
            "objective_tw_term": "{:.3f}",
            "objective_cost_term": "{:.3f}",
            "objective_cost_rate_term": "{:.3f}",
            "reward_mean": "{:.3f}",
            "reward_std": "{:.3f}",
            "occ_aux_loss": "{:.3f}",
        }
        full_list_keys = {"cost_mean", "tw_late", "rho_x_k", "rho_z_k", "obj_x_k", "obj_z_k"}
        if isinstance(value, torch.Tensor):
            value = value.detach().flatten().cpu().tolist()
        if isinstance(value, (list, tuple)):
            if not value:
                return f"{key}=[]"
            formatted_items = []
            for item in value:
                if isinstance(item, (int, float, np.floating)):
                    num = float(item)
                    if math.isnan(num):
                        formatted_items.append("nan")
                    elif math.isinf(num):
                        formatted_items.append("inf" if num > 0 else "-inf")
                    else:
                        formatted_items.append("{:.3f}".format(num))
                else:
                    formatted_items.append(str(item))
            if key in full_list_keys:
                return f"{key}=[{', '.join(formatted_items)}]"
            preview = formatted_items[:3]
            if len(formatted_items) > 3:
                preview.append("...")
            return f"{key}=[{', '.join(preview)}]"
        if isinstance(value, (int, float)):
            fmt = formatters.get(key, "{:.4f}")
            return f"{key}={fmt.format(value)}"
        return f"{key}={value}"

    def _log_train_step(self, phase: str, step_idx: int, total_steps: int, metrics: dict):
        if not metrics:
            return
        ordered_keys = [
            "loss",
            "R_mean",
            "lr",
            "route_infeas",
            "pair_count",
            "route_count",
            "feas_avg",
            "infeas_avg",
            "infeas_norm",
            "tw_late",
            "cost_mean",
            "objective_mean",
            "consensus_penalty",
            "rho_x_k",
            "rho_z_k",
            "obj_x_k",
            "obj_z_k",
            "adv_min",
            "adv_max",
            "forward_flow",
            "flow_gap_mean",
            "entropy_term",
            "penalty_term",
            "reward_total",
            "objective_cost_term",
            "objective_cost_rate_term",
            "objective_tw_term",
            "reward_mean",
            "reward_std",
            "occ_aux_loss",
            "loss_mode",
        ]
        metric_parts = [self._format_metric_value(key, metrics[key]) for key in ordered_keys if key in metrics]
        if not metric_parts:
            metric_parts = [f"{key}={value}" for key, value in metrics.items()]
        metric_str = ", ".join(metric_parts)
        print(f"{phase} [{step_idx}/{total_steps}] | {metric_str}")

    def _compute_feasibility_metrics(self, paths: torch.Tensor, metadata: dict):
        _, infeasible_flags = self._collect_route_violation_data(paths, metadata)
        solution_flags = [1.0 - float(flag) for flag in infeasible_flags]

        instance_feasible = 1.0 if any(flag > 0.5 for flag in solution_flags) else 0.0
        return {
            "solution_flags": solution_flags,
            "instance_feasible": instance_feasible,
        }

    @staticmethod
    def _diversity_from_tours(tours: list[list[int]], compute_bpd: bool = True):
        bpd_default = 0.0 if compute_bpd else float("nan")
        if not tours:
            return None
        edge_signatures = []
        for tour in tours:
            if not tour:
                continue
            edges = []
            n = len(tour)
            for i in range(n):
                u = int(tour[i])
                v = int(tour[(i + 1) % n])
                if u == v:
                    continue
                edges.append((u, v))
            edges.sort()
            edge_signatures.append(tuple(edges))
        if not edge_signatures:
            return {"unique_ratio": 0.0, "bpd": bpd_default}
        total = len(edge_signatures)
        unique_ratio = len(set(edge_signatures)) / total if total else 0.0
        if not compute_bpd:
            return {"unique_ratio": float(unique_ratio), "bpd": bpd_default}
        if total < 2:
            avg_bpd = 0.0
        else:
            edge_sets = [set(sig) for sig in edge_signatures]
            n_nodes = len(tours[0]) if tours and len(tours[0]) > 0 else 1
            total_bpd = 0.0
            pair_count = 0
            for i in range(total - 1):
                set_i = edge_sets[i]
                for j in range(i + 1, total):
                    inter_size = len(set_i & edge_sets[j])
                    bpd = 1.0 - (inter_size / n_nodes) if n_nodes > 0 else 0.0
                    total_bpd += bpd
                    pair_count += 1
            avg_bpd = total_bpd / pair_count if pair_count else 0.0
        return {
            "unique_ratio": float(unique_ratio),
            "bpd": float(avg_bpd),
        }

    def _analyze_iteration_records(self, records, metadata, gt_cost):
        per_iteration = []
        if not records:
            return per_iteration

        for entry in records:
            paths = entry.get("paths")
            costs = entry.get("costs")
            if paths is None or costs is None:
                per_iteration.append({
                    "best_cost": None,
                    "gt_cost": None,
                    "solution_flags": [],
                    "instance_feasible": 0.0,
                    "feasible_tours": [],
                    "feas_inst_sol_infeas": None,
                })
                continue

            costs_tensor = costs.detach().cpu().reshape(-1)
            num_tours = costs_tensor.shape[0]

            violation_counts, infeasible_flags = self._collect_route_violation_data(paths, metadata)
            infeasible_tensor = torch.as_tensor(infeasible_flags, dtype=torch.float32)
            if infeasible_tensor.numel() > num_tours:
                infeasible_tensor = infeasible_tensor[:num_tours]
            elif infeasible_tensor.numel() < num_tours:
                pad = torch.zeros(num_tours - infeasible_tensor.numel(), dtype=torch.float32)
                infeasible_tensor = torch.cat([infeasible_tensor, pad], dim=0)
            solution_flags = (1.0 - infeasible_tensor).tolist()

            feasible_mask = [flag > 0.5 for flag in solution_flags]
            feasible_indices = [idx for idx, flag in enumerate(feasible_mask) if flag]
            feasible_count = len(feasible_indices)

            tours_tensor = paths.detach().cpu()
            if tours_tensor.dim() != 2:
                tours_tensor = tours_tensor.reshape(tours_tensor.size(0), -1)
            tours_np = tours_tensor.transpose(0, 1).numpy()
            feasible_tours = [tours_np[idx].astype(int).tolist() for idx in feasible_indices if idx < tours_np.shape[0]]

            if feasible_indices:
                feasible_costs = costs_tensor[feasible_indices]
                best_cost = float(feasible_costs.min().item())
                feas_inst_sol_infeas = 100.0 * (num_tours - feasible_count) / num_tours if num_tours > 0 else None
            else:
                best_cost = None
                feas_inst_sol_infeas = None

            per_iteration.append({
                "best_cost": best_cost,
                "gt_cost": float(gt_cost) if best_cost is not None else None,
                "solution_flags": solution_flags,
                "instance_feasible": 1.0 if feasible_count > 0 else 0.0,
                "feasible_tours": feasible_tours,
                "feas_inst_sol_infeas": feas_inst_sol_infeas,
            })
        return per_iteration

    @staticmethod
    def _mean_or_nan(values):
        valid = [float(v) for v in values if v is not None and not math.isnan(v)]
        if not valid:
            return float("nan")
        return float(sum(valid) / len(valid))

    @staticmethod
    def _compute_average_gap(predictions, gt_costs):
        gaps = []
        for pred, gt in zip(predictions, gt_costs):
            if pred is None or gt is None:
                continue
            if math.isnan(pred) or math.isnan(gt) or abs(gt) < 1e-9:
                continue
            gaps.append((pred - gt) / gt * 100)
        if not gaps:
            return float("nan")
        return float(sum(gaps) / len(gaps))

    @staticmethod
    def _aggregate_infeasibility(solution_flags_list, instance_flags_list):
        flat_flags = [flag for flags in solution_flags_list for flag in flags]
        if flat_flags:
            feasible_ratio = sum(flat_flags) / len(flat_flags)
            infeas_sol = 100.0 * (1.0 - feasible_ratio)
        else:
            infeas_sol = float("nan")
        if instance_flags_list:
            feasible_inst_ratio = sum(instance_flags_list) / len(instance_flags_list)
            infeas_inst = 100.0 * (1.0 - feasible_inst_ratio)
        else:
            infeas_inst = float("nan")
        return infeas_sol, infeas_inst

    @staticmethod
    def _format_scalar(value, precision=4):
        if value is None or math.isnan(value):
            return "nan"
        fmt = "{:." + str(precision) + "f}"
        return fmt.format(value)

    @staticmethod
    def _feasible_instance_infeas_solution_pct(solution_flags_list):
        if not solution_flags_list:
            return float("nan")
        ratios = []
        for flags in solution_flags_list:
            total = len(flags)
            if total == 0:
                continue
            feasible = sum(flags)
            infeasible = total - feasible
            ratios.append(100.0 * infeasible / total)
        if not ratios:
            return float("nan")
        return float(sum(ratios) / len(ratios))

    @torch.no_grad()
    def _overall_inference_batch(self, base_data, generate, k_steps, desi, tw_mask, use_pomo, batch_cache=None):
        if not base_data:
            return None, None
        batch_size = len(base_data)
        n_nodes = base_data[0][1].size(0)
        generate = max(1, int(generate))
        k_steps = max(1, int(k_steps))

        if batch_cache is None:
            batch_cache = self._build_base_batch_cache(base_data, device=self.device, include_pyg=True)
        dist_batch = batch_cache["distances"]
        metadata_batch = batch_cache["metadata"]

        rho_x = torch.full((batch_size,), self.rho_x_init, device=self.device, dtype=dist_batch.dtype)
        rho_z = torch.full((batch_size,), self.rho_z_init, device=self.device, dtype=dist_batch.dtype)
        u_x = torch.zeros((batch_size, self.occ_time_bins, n_nodes, n_nodes), device=self.device, dtype=dist_batch.dtype)
        u_z = torch.zeros((batch_size, self.occ_time_bins, n_nodes, n_nodes), device=self.device, dtype=dist_batch.dtype)
        v = torch.zeros((batch_size, self.occ_time_bins, n_nodes, n_nodes), device=self.device, dtype=dist_batch.dtype)

        T_f = self.consensus_weight_update_interval
        anchor = None
        z_cand = None

        for k in range(k_steps):
            u_x_summary = self._occupancy_state_summary(u_x)
            u_z_summary = self._occupancy_state_summary(u_z)
            v_summary = self._occupancy_state_summary(v)

            x_cand = self._sample_model_paths(
                self.net,
                self.decoder,
                base_data,
                generate,
                tw_mask=tw_mask,
                use_pomo=use_pomo,
                alpha=1.0,
                desi=desi,
                offload=True,
                rho=rho_x,
                u_matrix=u_x_summary,
                v_matrix=v_summary,
                batch_cache=batch_cache,
            )
            x_selected = self._select_best_x(
                x_cand,
                dist_batch,
                metadata_batch,
                rho=rho_x,
                u_state=u_x,
                v_state=v,
            )

            z_cand = self._sample_model_paths(
                self.rep_net,
                self.rep_decoder,
                base_data,
                generate,
                tw_mask=tw_mask,
                use_pomo=use_pomo,
                alpha=1.0,
                desi=desi,
                offload=True,
                rho=rho_z,
                u_matrix=u_z_summary,
                v_matrix=v_summary,
                batch_cache=batch_cache,
            )
            z_selected = self._select_best_z(
                z_cand,
                metadata_batch,
                distances_batch=dist_batch,
                rho=rho_z,
                u_state=u_z,
                v_state=v,
            )

            y_x = self._materialize_occupancy_tensor(x_selected, metadata_batch, project_to_window=False)
            y_z = self._materialize_occupancy_tensor(z_selected, metadata_batch, project_to_window=True)

            rho_x_view = rho_x.view(-1, 1, 1, 1)
            rho_z_view = rho_z.view(-1, 1, 1, 1)
            hat_u_x = u_x + rho_x_view * (v - y_x)
            hat_u_z = u_z + rho_z_view * (v - y_z)

            v_next = self._update_global_consensus(rho_x, rho_z, u_x, u_z, y_x, y_z)
            u_x_next = u_x + rho_x_view * (v_next - y_x)
            u_z_next = u_z + rho_z_view * (v_next - y_z)

            current_state = {
                "y_x": y_x,
                "y_z": y_z,
                "hat_u_x": hat_u_x,
                "hat_u_z": hat_u_z,
                "u_x": u_x_next,
                "u_z": u_z_next,
                "v": v_next,
            }
            if anchor is None:
                anchor = {k_: v_.clone() for k_, v_ in current_state.items()}
            elif (k + 1) % T_f == 0:
                delta_v = anchor["v"] - current_state["v"]
                rho_x = self._update_consensus_weights(
                    current_state["y_x"] - anchor["y_x"],
                    current_state["hat_u_x"] - anchor["hat_u_x"],
                    current_state["u_x"] - anchor["u_x"],
                    delta_v,
                    rho_x,
                )
                rho_z = self._update_consensus_weights(
                    current_state["y_z"] - anchor["y_z"],
                    current_state["hat_u_z"] - anchor["hat_u_z"],
                    current_state["u_z"] - anchor["u_z"],
                    delta_v,
                    rho_z,
                )
                anchor = {k_: v_.clone() for k_, v_ in current_state.items()}

            u_x = u_x_next
            u_z = u_z_next
            v = v_next

        z_tours_device = z_cand.to(self.device)
        costs = self._compute_route_costs(dist_batch, z_tours_device)
        return z_cand, costs

    def validate(self, model_inp, distances, metadata, generate):
        self.net.eval()
        if self.decoder is not None:
            self.decoder.eval()
        if self.rep_net is not None:
            self.rep_net.eval()
        if self.rep_decoder is not None:
            self.rep_decoder.eval()

        base_data = [(model_inp, distances, metadata)]
        paths_iter1, costs_iter1 = self._overall_inference_batch(
            base_data,
            generate,
            self.K,
            desi=1,
            tw_mask=self.args.tw_mask_val,
            use_pomo=self.args.val_use_pomo,
        )
        if paths_iter1 is None or costs_iter1 is None:
            raise RuntimeError("Overall inference returned no paths.")

        paths_iter1 = paths_iter1[:, 0, :]
        costs_iter1 = costs_iter1[0]
        bes = costs_iter1.mean().item()
        bes_cost = costs_iter1.min().item()
        bes1 = costs_iter1.min().item()

        iter1_records = [{
            "paths": paths_iter1.detach().cpu(),
            "costs": costs_iter1.detach().cpu(),
        }]

        diversity_iter1 = self._diversity_from_tours(
            paths_iter1.transpose(0, 1).tolist(), compute_bpd=self.compute_val_bpd
        )
        if diversity_iter1 is None:
            diversity_iter1 = {"unique_ratio": 0.0, "bpd": 0.0 if self.compute_val_bpd else float("nan")}

        val_iter = int(getattr(self.args, "validation_iter", 0))
        if val_iter <= 1:
            besT = bes1
            diversity_iterT = diversity_iter1
            iterT_records = []
        else:
            all_tours = [tour for tour in paths_iter1.transpose(0, 1).tolist()]
            best_cost = bes1
            iterT_records = []
            for _ in range(val_iter - 1):
                paths_iter, costs_iter = self._overall_inference_batch(
                    base_data,
                    generate,
                    self.K,
                    desi=1,
                    tw_mask=self.args.tw_mask_val,
                    use_pomo=self.args.val_use_pomo,
                )
                paths_iter = paths_iter[:, 0, :]
                costs_iter = costs_iter[0]
                iterT_records.append({
                    "paths": paths_iter.detach().cpu(),
                    "costs": costs_iter.detach().cpu(),
                })
                all_tours.extend(paths_iter.transpose(0, 1).tolist())
                best_cost = min(best_cost, costs_iter.min().item())
            diversity_iterT = self._diversity_from_tours(all_tours, compute_bpd=self.compute_val_bpd)
            if diversity_iterT is None:
                diversity_iterT = {"unique_ratio": 0.0, "bpd": 0.0 if self.compute_val_bpd else float("nan")}
            besT = best_cost

        diversity_metrics = {
            "unique_ratio": diversity_iterT["unique_ratio"],
            "bpd": diversity_iterT["bpd"],
            "unique_ratio_iter1": diversity_iter1["unique_ratio"],
            "bpd_iter1": diversity_iter1["bpd"],
        }
        feasibility_metrics = self._compute_feasibility_metrics(paths_iter1, metadata)
        iter_records = {"iter1": iter1_records}
        if val_iter > 1:
            iter_records[f"iter{val_iter}"] = iterT_records
        return np.array([bes, bes_cost, bes1, besT]), diversity_metrics, feasibility_metrics, costs_iter1, paths_iter1, iter_records

    @torch.no_grad()
    def validation(self, dataset, opt_sol, step, val_episodes, collect_details=False):
        generate = self.model_params["val_generate"]
        sample_records = [] if collect_details else None
        total_solution_time = 0.0
        total_solutions = 0
        instance_times = []
        val_batch_size = max(1, int(getattr(self.args, "val_batch_size", 1)))

        opt_subset = opt_sol[:val_episodes] if len(opt_sol) >= val_episodes else opt_sol
        opt_records = []
        for entry in opt_subset:
            cost = None
            path = None
            if isinstance(entry, dict):
                cost = entry.get("cost")
                path = entry.get("path")
            elif isinstance(entry, (list, tuple)):
                if len(entry) >= 1:
                    cost = entry[0]
                if len(entry) >= 2:
                    path = entry[1]
            else:
                cost = entry
            opt_records.append({
                "cost": float(cost) if cost is not None else None,
                "cost_normalized": Trainer._normalize_cost(cost),
                "path": path.tolist() if isinstance(path, np.ndarray) else (list(path) if path is not None else None),
            })
        opt_sol_costs = [
            rec["cost_normalized"] if rec["cost_normalized"] is not None else float("nan")
            for rec in opt_records
        ]
        val_iter = int(getattr(self.args, "validation_iter", 0))
        iter_keys = ["iter1"] if val_iter <= 1 else ["iter1", f"iter{val_iter}"]
        iteration_info = {key: [] for key in iter_keys}
        iter_runs = 1 if val_iter <= 1 else val_iter
        iter_t_key = f"iter{val_iter}"

        def _init_iter_slot():
            return {
                "best_costs": [],
                "gt_costs": [],
                "solution_flags": [],
                "instance_flags": [],
                "feasible_tours": [],
                "feas_inst_infeas": [],
            }

        eval_data = dataset[:val_episodes]
        chunk_total = math.ceil(len(eval_data) / val_batch_size) if val_batch_size > 0 else 0
        for chunk_start in tqdm(range(0, len(eval_data), val_batch_size), total=chunk_total, desc="Validation"):
            chunk_entries = eval_data[chunk_start:chunk_start + val_batch_size]
            if not chunk_entries:
                continue

            chunk_variants = []
            chunk_variant_owner = []
            chunk_variant_counts = []
            for local_idx, entry in enumerate(chunk_entries):
                variants = entry["variants"] if isinstance(entry, dict) and "variants" in entry else [entry]
                chunk_variant_counts.append(len(variants))
                for v_idx, variant in enumerate(variants):
                    chunk_variants.append(variant)
                    chunk_variant_owner.append((local_idx, v_idx))

            if not chunk_variants:
                continue

            per_variant_records = []
            for variant_count in chunk_variant_counts:
                variant_records = []
                for _ in range(variant_count):
                    records = {"iter1": []}
                    if val_iter > 1:
                        records[iter_t_key] = []
                    variant_records.append(records)
                per_variant_records.append(variant_records)

            chunk_instance_times = [0.0] * len(chunk_entries)
            chunk_instance_solutions = [0] * len(chunk_entries)
            chunk_cache = self._build_base_batch_cache(chunk_variants, device=self.device, include_pyg=True)

            for iter_run_idx in range(iter_runs):
                t_start = time.perf_counter()
                paths_batch, costs_batch = self._overall_inference_batch(
                    chunk_variants,
                    generate,
                    self.K,
                    desi=1,
                    tw_mask=self.args.tw_mask_val,
                    use_pomo=self.args.val_use_pomo,
                    batch_cache=chunk_cache,
                )
                t_end = time.perf_counter()
                if paths_batch is None or costs_batch is None:
                    raise RuntimeError("Overall inference returned no paths.")

                elapsed = t_end - t_start
                produced = len(chunk_variants) * generate
                total_solution_time += elapsed
                total_solutions += produced

                if len(chunk_variants) > 0:
                    inv_total = 1.0 / len(chunk_variants)
                    for local_idx, variant_count in enumerate(chunk_variant_counts):
                        chunk_instance_times[local_idx] += elapsed * variant_count * inv_total
                        chunk_instance_solutions[local_idx] += variant_count * generate

                iter_key = "iter1" if iter_run_idx == 0 else iter_t_key
                for flat_idx, (local_idx, v_idx) in enumerate(chunk_variant_owner):
                    record = {
                        "paths": paths_batch[:, flat_idx, :].detach().cpu(),
                        "costs": costs_batch[flat_idx].detach().cpu(),
                    }
                    per_variant_records[local_idx][v_idx][iter_key].append(record)

            for local_idx, entry in enumerate(chunk_entries):
                global_idx = chunk_start + local_idx
                variants = entry["variants"] if isinstance(entry, dict) and "variants" in entry else [entry]
                gt_cost = opt_sol_costs[global_idx] if global_idx < len(opt_sol_costs) else float("nan")
                per_iter_agg = {key: [] for key in iter_keys}
                first_paths = None

                for v_idx, (_, _, metadata) in enumerate(variants):
                    iteration_records = per_variant_records[local_idx][v_idx]
                    if v_idx == 0:
                        iter1_records = iteration_records.get("iter1", [])
                        if iter1_records:
                            first_paths = iter1_records[0].get("paths")

                    if collect_details and sample_records is not None and v_idx == 0 and first_paths is not None:
                        coords_tensor = torch.as_tensor(metadata["coordinates"]).detach().cpu()
                        tw_start_tensor = torch.as_tensor(metadata["tw_start"]).detach().cpu()
                        tw_end_tensor = torch.as_tensor(metadata["tw_end"]).detach().cpu()

                        sample_entry = {"sample": f"{global_idx:03d}"}
                        gt_info = opt_records[global_idx] if global_idx < len(opt_records) else {"path": None}
                        gt_path = gt_info.get("path")
                        if gt_path is not None:
                            gt_indices = [int(node) for node in gt_path]
                            if not gt_indices or gt_indices[0] != 0:
                                gt_indices = [0] + gt_indices
                        else:
                            gt_indices = [0]

                        gt_trajectory = [coords_tensor[node].tolist() for node in gt_indices]
                        gt_time_windows = [
                            [float(tw_start_tensor[node].item()), float(tw_end_tensor[node].item())]
                            for node in gt_indices
                        ]
                        sample_entry["gt"] = {
                            "node_indices": gt_indices,
                            "trajectory": gt_trajectory,
                            "time_windows": gt_time_windows,
                        }

                        paths_tensor = first_paths.detach().cpu() if isinstance(first_paths, torch.Tensor) else torch.as_tensor(first_paths)
                        total_generations = paths_tensor.shape[1] if paths_tensor.ndim == 2 else 0
                        for gen_idx in range(total_generations):
                            node_order = [int(n) for n in paths_tensor[:, gen_idx].tolist()]
                            gen_entry = {
                                "node_indices": node_order,
                                "trajectory": [coords_tensor[node].tolist() for node in node_order],
                            }
                            sample_entry[f"gen{gen_idx + 1}"] = gen_entry
                        sample_records.append(sample_entry)

                    for iter_key in iter_keys:
                        records = iteration_records.get(iter_key, [])
                        per_iter_results = self._analyze_iteration_records(records, metadata, gt_cost)
                        agg_list = per_iter_agg[iter_key]
                        while len(agg_list) < len(per_iter_results):
                            agg_list.append({
                                "best_cost": None,
                                "gt_cost": None,
                                "solution_flags": [],
                                "instance_flags": [],
                                "feasible_tours": [],
                                "feas_inst_infeas": [],
                            })
                        for iter_idx, iter_metrics in enumerate(per_iter_results):
                            agg = agg_list[iter_idx]
                            agg["solution_flags"].extend(iter_metrics["solution_flags"])
                            agg["instance_flags"].append(iter_metrics["instance_feasible"])
                            agg["feasible_tours"].extend(iter_metrics["feasible_tours"])
                            if iter_metrics["feas_inst_sol_infeas"] is not None:
                                agg["feas_inst_infeas"].append(iter_metrics["feas_inst_sol_infeas"])
                            if iter_metrics["best_cost"] is not None:
                                current = agg["best_cost"]
                                new_best = iter_metrics["best_cost"]
                                agg["best_cost"] = new_best if current is None else min(current, new_best)
                                agg["gt_cost"] = iter_metrics["gt_cost"]

                if chunk_instance_solutions[local_idx] > 0:
                    instance_times.append(chunk_instance_times[local_idx])

                # Reduce aggregated variants into slots
                for iter_key in iter_keys:
                    agg_list = per_iter_agg[iter_key]
                    slots = iteration_info[iter_key]
                    while len(slots) < len(agg_list):
                        slots.append(_init_iter_slot())
                    for iter_idx, agg in enumerate(agg_list):
                        slot = slots[iter_idx]
                        slot["solution_flags"].extend(agg["solution_flags"])
                        any_feasible = any(flag > 0.5 for flag in agg["solution_flags"])
                        slot["instance_flags"].append(1.0 if any_feasible else 0.0)
                        slot["feasible_tours"].extend(agg["feasible_tours"])
                        if agg["feas_inst_infeas"]:
                            slot["feas_inst_infeas"].append(sum(agg["feas_inst_infeas"]) / len(agg["feas_inst_infeas"]))
                        if agg["best_cost"] is not None and any_feasible:
                            slot["best_costs"].append(agg["best_cost"])
                            slot["gt_costs"].append(agg.get("gt_cost", gt_cost))

        def _avg_or_nan(values):
            cleaned = [v for v in values if v is not None and not math.isnan(v)]
            if not cleaned:
                return float("nan")
            return float(sum(cleaned) / len(cleaned))

        def _summarize_slot(slot):
            avg_obj = self._mean_or_nan(slot["best_costs"])
            avg_gap = self._compute_average_gap(slot["best_costs"], slot["gt_costs"])
            diversity = self._diversity_from_tours(
                slot["feasible_tours"],
                compute_bpd=self.compute_val_bpd,
            )
            if diversity is None:
                unique_ratio = avg_bpd = float("nan")
            else:
                unique_ratio = diversity["unique_ratio"]
                avg_bpd = diversity["bpd"]

            if slot["solution_flags"]:
                feasible_ratio = sum(slot["solution_flags"]) / len(slot["solution_flags"])
                infeas_sol = 100.0 * (1.0 - feasible_ratio)
            else:
                infeas_sol = float("nan")

            if slot["instance_flags"]:
                feasible_inst_ratio = sum(slot["instance_flags"]) / len(slot["instance_flags"])
                infeas_inst = 100.0 * (1.0 - feasible_inst_ratio)
            else:
                infeas_inst = float("nan")

            feas_inst_sol_infeas = self._mean_or_nan(slot["feas_inst_infeas"])
            return {
                "avg_obj": avg_obj,
                "avg_gap": avg_gap,
                "diversity": {
                    "unique_ratio": unique_ratio,
                    "bpd": avg_bpd,
                },
                "infeasibility": {
                    "sol": infeas_sol,
                    "inst": infeas_inst,
                    "feas_inst_sol_infeas": feas_inst_sol_infeas,
                },
            }

        iteration_stats = {}
        for iter_key in iter_keys:
            slots = iteration_info.get(iter_key, [])
            per_iter_metrics = [_summarize_slot(slot) for slot in slots] if slots else []
            aggregated = {
                "avg_obj": _avg_or_nan([m["avg_obj"] for m in per_iter_metrics]),
                "avg_gap": _avg_or_nan([m["avg_gap"] for m in per_iter_metrics]),
                "diversity": {
                    "unique_ratio": _avg_or_nan([m["diversity"]["unique_ratio"] for m in per_iter_metrics]),
                    "bpd": _avg_or_nan([m["diversity"]["bpd"] for m in per_iter_metrics]),
                },
                "infeasibility": {
                    "sol": _avg_or_nan([m["infeasibility"]["sol"] for m in per_iter_metrics]),
                    "inst": _avg_or_nan([m["infeasibility"]["inst"] for m in per_iter_metrics]),
                    "feas_inst_sol_infeas": _avg_or_nan([m["infeasibility"]["feas_inst_sol_infeas"] for m in per_iter_metrics]),
                },
                "per_iteration": per_iter_metrics,
            }
            iteration_stats[iter_key] = aggregated

        for iter_key in iter_keys:
            stats = iteration_stats.get(iter_key, {})
            diversity_vals = stats.get("diversity", {})
            infeas_vals = stats.get("infeasibility", {})
            print(f"    [{iter_key}] infeasibility: sol% = [{self._format_scalar(infeas_vals.get('sol', float('nan')), 2)}], "
                  f"inst% = [{self._format_scalar(infeas_vals.get('inst', float('nan')), 2)}]")
            print(f"        feasible only: avg obj = [{self._format_scalar(stats.get('avg_obj', float('nan')), 6)}], "
                  f"avg gap (%) = [{self._format_scalar(stats.get('avg_gap', float('nan')), 4)}]")
            print(f"        feasible diversity: unique_ratio = [{self._format_scalar(diversity_vals.get('unique_ratio', float('nan')), 4)}], "
                  f"bpd = [{self._format_scalar(diversity_vals.get('bpd', float('nan')), 4)}]")
            feas_inst_val = infeas_vals.get("feas_inst_sol_infeas", float("nan"))
            print(f"        feas-inst sol_infeas% = [{self._format_scalar(feas_inst_val, 2)}]")

        avg_solution_time = (total_solution_time / total_solutions) if total_solutions > 0 else float("nan")
        avg_instance_time = (
            float(sum(instance_times) / len(instance_times)) if instance_times else float("nan")
        )
        print(f"    [walltime] per-solution = [{self._format_scalar(avg_solution_time, 6)}] s, "
              f"per-instance = [{self._format_scalar(avg_instance_time, 6)}] s")

        final_iter_label = "iter1" if val_iter <= 1 else f"iter{val_iter}"
        final_all_stats = iteration_stats.get(final_iter_label, {})
        score_value = final_all_stats.get("avg_obj", float("nan"))
        gap_value = final_all_stats.get("avg_gap", float("nan"))
        diversity_final = final_all_stats.get("diversity", {"unique_ratio": float("nan"), "bpd": float("nan")})
        infeas_final = final_all_stats.get("infeasibility", {"sol": float("nan"), "inst": float("nan"), "feas_inst_sol_infeas": float("nan")})

        metrics_summary = {
            "average_obj": float(score_value) if not math.isnan(score_value) else 0.0,
            "average_gap": float(gap_value) if not math.isnan(gap_value) else 0.0,
            "walltime": {
                "per_solution": avg_solution_time,
                "per_instance": avg_instance_time,
            },
            "diversity": {
                "iter1": iteration_stats.get("iter1", {}).get("diversity", {"unique_ratio": float("nan"), "bpd": float("nan")}),
                final_iter_label: diversity_final,
            },
            "infeasibility": infeas_final,
            "iterations": iteration_stats,
            "final_iter_label": final_iter_label,
        }

        score_return = float(score_value) if not math.isnan(score_value) else 0.0
        gap_return = float(gap_value) if not math.isnan(gap_value) else 0.0
        unique_return = float(diversity_final["unique_ratio"]) if not math.isnan(diversity_final["unique_ratio"]) else 0.0
        if self.compute_val_bpd:
            bpd_return = float(diversity_final["bpd"]) if not math.isnan(diversity_final["bpd"]) else 0.0
        else:
            bpd_return = float("nan")
        infeas_sol_return = float(infeas_final["sol"]) if not math.isnan(infeas_final["sol"]) else 0.0
        infeas_inst_return = float(infeas_final["inst"]) if not math.isnan(infeas_final["inst"]) else 0.0

        return (
            score_return,
            gap_return,
            unique_return,
            bpd_return,
            infeas_sol_return,
            infeas_inst_return,
            sample_records,
            metrics_summary,
        )
    
    def _collect_run_config(self):
        snapshot = getattr(self.args, "config_snapshot", None)
        if snapshot is None:
            snapshot = {k: v for k, v in vars(self.args).items()}
        config = {}
        for key, value in snapshot.items():
            if isinstance(value, (int, float, bool)) or value is None:
                config[key] = value
            elif isinstance(value, str):
                config[key] = value
            else:
                config[key] = str(value)
        return config

    @staticmethod
    def _format_yaml(content, indent=0):
        lines = []
        for key, value in content.items():
            prefix = " " * indent
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(Trainer._format_yaml(value, indent + 2))
            else:
                if isinstance(value, float):
                    value_str = f"{value:.6f}"
                else:
                    value_str = value
                lines.append(f"{prefix}{key}: {value_str}")
        return lines

    def _save_validation_outputs(self, sample_records, metrics_summary):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.dirname(self.args.log_file)
        os.makedirs(log_dir, exist_ok=True)

        json_payload = {"samples": sample_records}
        json_path = os.path.join(log_dir, f"{timestamp}_validation_samples.json")
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(json_payload, json_file, indent=2)

        config_section = self._collect_run_config()
        yaml_content = {"config": config_section}
        yaml_content.update(metrics_summary)

        yaml_lines = self._format_yaml(yaml_content)
        yaml_path = os.path.join(log_dir, f"{timestamp}_validation_metrics.yaml")
        with open(yaml_path, "w", encoding="utf-8") as yaml_file:
            yaml_file.write("\n".join(yaml_lines) + "\n")

        print(f"    Saved validation samples to {json_path}")
        print(f"    Saved validation metrics to {yaml_path}")

    def _update_metric_artifacts(self, step, metrics_summary):
        iterations = metrics_summary.get("iterations", {})
        val_iter = int(getattr(self.args, "validation_iter", 0))
        default_label = "iter1" if val_iter <= 1 else f"iter{val_iter}"
        final_label = metrics_summary.get("final_iter_label", default_label)
        final_stats = iterations.get(final_label, {})
        feasible_div = final_stats.get("diversity", {})
        overall_infeas = final_stats.get("infeasibility", {})

        feasible_specs = [
            ("feasible_gap.csv", "feasible_gap.png", final_stats.get("avg_gap"),
             "Opt. Gap (Feasible Only)", "Opt. Gap (Feasible Only)", "Step", "Opt. Gap (%)"),
            ("feasible_unique_ratio.csv", "feasible_unique_ratio.png", feasible_div.get("unique_ratio"),
             "Unique Ratio (Feasible Only)", "Unique Ratio (Feasible Only)", "Step", "Unique Ratio"),
        ]
        if self.compute_val_bpd:
            feasible_specs.append(
                ("feasible_bpd.csv", "feasible_bpd.png", feasible_div.get("bpd"),
                 "BPD (Feasible Only)", "BPD (Feasible Only)", "Step", "BPD"),
            )

        for csv_name, fig_name, value, title, label, xdes, ydes in feasible_specs:
            self._record_metric(csv_name, fig_name, step, value, title, label, xdes, ydes)

        overall_specs = [
            ("overall_infeas_sol.csv", "overall_infeas_sol.png", overall_infeas.get("sol"),
             "Solution Infeasibility", "Solution Infeasibility", "Step", "Sol. Infeasibility (%)"),
            ("overall_infeas_inst.csv", "overall_infeas_inst.png", overall_infeas.get("inst"),
             "Instance Infeasibility", "Instance Infeasibility", "Step", "Inst. Infeasibility (%)"),
            ("feas_inst_sol_infeas.csv", "feas_inst_sol_infeas.png", overall_infeas.get("feas_inst_sol_infeas"),
             "Feasible-Inst Sol Infeasibility", "Feasible-Inst Sol Infeasibility", "Step", "Sol. Infeasibility (%)"),
        ]

        for csv_name, fig_name, value, title, label, xdes, ydes in overall_specs:
            self._record_metric(csv_name, fig_name, step, value, title, label, xdes, ydes)
    

    def run(self):
        self.time_estimator.reset(self.start_step)
        best_result = float('inf')
        
        # load validation data
        val_episodes, val_problem, problem_size = 1000 , self.env_params['problem'], self.env_params['problem_size']
        val_path = "{}{}_{}.pkl".format(val_problem.lower(), problem_size, self.env_params["hardness"])
        dir = os.path.join("./1_data", val_problem)
        val_data = self.load_val_data(self.env.load_dataset(os.path.join(dir, val_path), offset=0, num_samples=val_episodes), self.model_params["k_sparse"])

        # load optimal solutions
        opt_sol = load_dataset(get_opt_sol_path(dir, val_problem, problem_size, self.env_params["hardness"]), disable_print=True)[ : val_episodes]
        
        total_steps = int(self.trainer_params["steps"])
        for step in range(self.start_step, total_steps + 1):
            print('=================================================================')

            beta_min, beta_max, beta_flat_param = self.optimizer_params["beta_schedule_params"]
            if 0.0 <= beta_flat_param <= 1.0:
                beta_flat_steps = int(round(beta_flat_param * total_steps))
            else:
                beta_flat_steps = int(beta_flat_param)
            beta_flat_steps = max(beta_flat_steps, 0)
            remaining_steps = max(total_steps - beta_flat_steps, 2)
            if step <= beta_flat_steps:
                beta_ratio = 0.0
            else:
                progress = min(max((step - beta_flat_steps) / remaining_steps, 0.0), 1.0)
                beta_ratio = math.log1p(progress * (math.e - 1.0))
            beta = beta_min + (beta_max - beta_min) * beta_ratio

            alpha_min, alpha_max, alpha_flat_epochs = self.optimizer_params["alpha_schedule_params"]
            alpha = alpha_min + (alpha_max - alpha_min) * min((step - 1) / max(total_steps - alpha_flat_epochs, 1), 1.0)

            k_steps = self.K
            rollout_batch_size = int(self.trainer_params["rollout_batch_size"])
            rollout_instances = max(1, rollout_batch_size)

            print(
                f"Step {step:3d}/{total_steps:3d}: beta={beta:.4f} (min={beta_min}, max={beta_max}, ratio={beta_ratio:.4f}) "
                f"| K={k_steps} | rho_x_init={self.rho_x_init:g} | rho_z_init={self.rho_z_init:g} "
                f"| T_f={self.consensus_weight_update_interval} | eps_cor={self.consensus_correlation_threshold:g} "
                f"| grad_accum={self.grad_accum} | select_best={self.select_best_mode} "
                f"| occ_bins={self.occ_time_bins} | occ_aux={self.occ_aux_weight:g}"
            )

            rollout_data = self.train_data(rollout_instances, self.model_params["k_sparse"])
            if not rollout_data:
                continue

            batch_size = len(rollout_data)
            n_nodes = rollout_data[0][1].size(0)
            generate = int(self.model_params["train_generate"])
            rollout_cache = self._build_base_batch_cache(rollout_data, device=self.device, include_pyg=True)
            dist_batch = rollout_cache["distances"]
            metadata_batch = rollout_cache["metadata"]

            gen_metrics_list = []
            rep_metrics_list = []

            def _step_with_clip(optimizer):
                if optimizer is None:
                    return
                params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
                if not params:
                    return
                torch.nn.utils.clip_grad_norm_(params, max_norm=3.0, norm_type=2)  # type: ignore
                optimizer.step()

            gen_rollout_stats = {
                "cost_mean": [float("nan")] * k_steps,
                "tw_late": [float("nan")] * k_steps,
            }
            rep_rollout_stats = {
                "cost_mean": [float("nan")] * k_steps,
                "tw_late": [float("nan")] * k_steps,
            }
            admm_stats = {
                "rho_x_k": [],
                "rho_z_k": [],
                "obj_x_k": [],
                "obj_z_k": [],
            }

            step_progress = tqdm(
                total=2 * k_steps,
                desc=f"Step {step}/{total_steps} Progress",
                leave=False,
            )
            rho_x = torch.full((batch_size,), self.rho_x_init, device=self.device, dtype=dist_batch.dtype)
            rho_z = torch.full((batch_size,), self.rho_z_init, device=self.device, dtype=dist_batch.dtype)
            u_x = torch.zeros((batch_size, self.occ_time_bins, n_nodes, n_nodes), device=self.device, dtype=dist_batch.dtype)
            u_z = torch.zeros((batch_size, self.occ_time_bins, n_nodes, n_nodes), device=self.device, dtype=dist_batch.dtype)
            v = torch.zeros((batch_size, self.occ_time_bins, n_nodes, n_nodes), device=self.device, dtype=dist_batch.dtype)
            admm_stats["rho_x_k"].append(float(rho_x.mean().item()))
            admm_stats["rho_z_k"].append(float(rho_z.mean().item()))
            anchor = None
            T_f = self.consensus_weight_update_interval

            if self.grad_accum:
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                if self.rep_optimizer is not None:
                    self.rep_optimizer.zero_grad(set_to_none=True)

            for k in range(k_steps):
                u_x_summary = self._occupancy_state_summary(u_x)
                u_z_summary = self._occupancy_state_summary(u_z)
                v_summary = self._occupancy_state_summary(v)

                with torch.no_grad():
                    x_cand = self._sample_model_paths(
                        self.net,
                        self.decoder,
                        rollout_data,
                        generate,
                        tw_mask=self.args.tw_mask_train,
                        use_pomo=self.args.train_use_pomo,
                        alpha=alpha,
                        desi=0,
                        rho=rho_x,
                        u_matrix=u_x_summary,
                        v_matrix=v_summary,
                        batch_cache=rollout_cache,
                    )

                gen_entries = self._build_buffer_entries(
                    rollout_data,
                    x_cand,
                    step_idx=k,
                    step_count=k_steps,
                    rho=rho_x,
                    u_matrix=u_x_summary,
                    v_matrix=v_summary,
                )
                gen_loss, gen_metrics = self.generator_train(
                    gen_entries,
                    generate=1,
                    alpha=alpha,
                    beta=beta,
                    u_state=u_x,
                    v_state=v,
                )
                if gen_metrics:
                    gen_metrics_list.append(gen_metrics)
                if gen_loss is not None and self.optimizer is not None:
                    if self.grad_accum:
                        (gen_loss / max(k_steps, 1)).backward()
                    else:
                        self.optimizer.zero_grad(set_to_none=True)
                        gen_loss.backward()
                        _step_with_clip(self.optimizer)

                x_selected = self._select_best_x(
                    x_cand,
                    dist_batch,
                    metadata_batch,
                    rho=rho_x,
                    u_state=u_x,
                    v_state=v,
                )
                if self.select_best_mode == "best_r":
                    with torch.no_grad():
                        x_sel = x_selected.unsqueeze(2) if x_selected.dim() == 2 else x_selected
                        costs_x = self._compute_route_costs(dist_batch.to(x_sel.device), x_sel)
                        consensus_x = self._consensus_penalty(
                            paths=x_sel,
                            rho=rho_x.to(device=x_sel.device, dtype=costs_x.dtype),
                            u_state=u_x.to(device=x_sel.device, dtype=costs_x.dtype),
                            v_state=v.to(device=x_sel.device, dtype=costs_x.dtype),
                            metadata_batch=metadata_batch,
                            project_to_window=False,
                        )
                        admm_stats["obj_x_k"].append(float((costs_x + consensus_x).mean().item()))
                gen_cost, gen_tw = self._summarize_rollout_stats(
                    x_selected,
                    dist_batch,
                    metadata_batch,
                )
                gen_rollout_stats["cost_mean"][k] = gen_cost
                gen_rollout_stats["tw_late"][k] = gen_tw
                step_progress.set_postfix_str(f"cost{k}")
                step_progress.update(1)

                with torch.no_grad():
                    z_cand = self._sample_model_paths(
                        self.rep_net,
                        self.rep_decoder,
                        rollout_data,
                        generate,
                        tw_mask=self.args.tw_mask_train,
                        use_pomo=self.args.train_use_pomo,
                        alpha=alpha,
                        desi=0,
                        rho=rho_z,
                        u_matrix=u_z_summary,
                        v_matrix=v_summary,
                        batch_cache=rollout_cache,
                    )

                rep_entries = self._build_buffer_entries(
                    rollout_data,
                    z_cand,
                    step_idx=k,
                    step_count=k_steps,
                    rho=rho_z,
                    u_matrix=u_z_summary,
                    v_matrix=v_summary,
                )
                rep_loss, rep_metrics = self.repair_train(
                    rep_entries,
                    generate=1,
                    alpha=alpha,
                    beta=beta,
                    u_state=u_z,
                    v_state=v,
                )
                if rep_metrics:
                    rep_metrics_list.append(rep_metrics)
                if rep_loss is not None and self.rep_optimizer is not None:
                    if self.grad_accum:
                        (rep_loss / max(k_steps, 1)).backward()
                    else:
                        self.rep_optimizer.zero_grad(set_to_none=True)
                        rep_loss.backward()
                        _step_with_clip(self.rep_optimizer)

                z_selected = self._select_best_z(
                    z_cand,
                    metadata_batch,
                    distances_batch=dist_batch,
                    rho=rho_z,
                    u_state=u_z,
                    v_state=v,
                )
                if self.select_best_mode == "best_r":
                    with torch.no_grad():
                        z_sel = z_selected.unsqueeze(2) if z_selected.dim() == 2 else z_selected
                        tw_z = self._compute_tw_lateness_penalty_batch(z_sel, metadata_batch)
                        consensus_z = self._consensus_penalty(
                            paths=z_sel,
                            rho=rho_z.to(device=z_sel.device, dtype=tw_z.dtype),
                            u_state=u_z.to(device=z_sel.device, dtype=tw_z.dtype),
                            v_state=v.to(device=z_sel.device, dtype=tw_z.dtype),
                            metadata_batch=metadata_batch,
                            project_to_window=True,
                        )
                        admm_stats["obj_z_k"].append(float((tw_z + consensus_z).mean().item()))
                rep_cost, rep_tw = self._summarize_rollout_stats(
                    z_selected,
                    dist_batch,
                    metadata_batch,
                )
                rep_rollout_stats["cost_mean"][k] = rep_cost
                rep_rollout_stats["tw_late"][k] = rep_tw

                y_x = self._materialize_occupancy_tensor(x_selected, metadata_batch, project_to_window=False)
                y_z = self._materialize_occupancy_tensor(z_selected, metadata_batch, project_to_window=True)
                rho_x_view = rho_x.view(-1, 1, 1, 1)
                rho_z_view = rho_z.view(-1, 1, 1, 1)
                hat_u_x = u_x + rho_x_view * (v - y_x)
                hat_u_z = u_z + rho_z_view * (v - y_z)
                v_next = self._update_global_consensus(rho_x, rho_z, u_x, u_z, y_x, y_z)
                u_x_next = u_x + rho_x_view * (v_next - y_x)
                u_z_next = u_z + rho_z_view * (v_next - y_z)

                current_state = {
                    "y_x": y_x,
                    "y_z": y_z,
                    "hat_u_x": hat_u_x,
                    "hat_u_z": hat_u_z,
                    "u_x": u_x_next,
                    "u_z": u_z_next,
                    "v": v_next,
                }
                if anchor is None:
                    anchor = {k_: v_.clone() for k_, v_ in current_state.items()}
                elif (k + 1) % T_f == 0:
                    delta_v = anchor["v"] - current_state["v"]
                    rho_x = self._update_consensus_weights(
                        current_state["y_x"] - anchor["y_x"],
                        current_state["hat_u_x"] - anchor["hat_u_x"],
                        current_state["u_x"] - anchor["u_x"],
                        delta_v,
                        rho_x,
                    )
                    rho_z = self._update_consensus_weights(
                        current_state["y_z"] - anchor["y_z"],
                        current_state["hat_u_z"] - anchor["hat_u_z"],
                        current_state["u_z"] - anchor["u_z"],
                        delta_v,
                        rho_z,
                    )
                    anchor = {k_: v_.clone() for k_, v_ in current_state.items()}

                u_x = u_x_next
                u_z = u_z_next
                v = v_next
                admm_stats["rho_x_k"].append(float(rho_x.mean().item()))
                admm_stats["rho_z_k"].append(float(rho_z.mean().item()))
                step_progress.set_postfix_str(f"repair{k}")
                step_progress.update(1)

            if self.grad_accum:
                if self.optimizer is not None:
                    _step_with_clip(self.optimizer)
                if self.rep_optimizer is not None:
                    _step_with_clip(self.rep_optimizer)

            gen_metrics = self._average_metrics(gen_metrics_list) if gen_metrics_list else {}
            if gen_metrics:
                gen_metrics.update(gen_rollout_stats)
                self._log_train_step("Generator", step, total_steps, gen_metrics)

            rep_metrics = self._average_metrics(rep_metrics_list) if rep_metrics_list else {}
            rep_metrics.update(rep_rollout_stats)
            if admm_stats["rho_x_k"]:
                rep_metrics["rho_x_k"] = admm_stats["rho_x_k"]
            if admm_stats["rho_z_k"]:
                rep_metrics["rho_z_k"] = admm_stats["rho_z_k"]
            if admm_stats["obj_x_k"]:
                rep_metrics["obj_x_k"] = admm_stats["obj_x_k"]
            if admm_stats["obj_z_k"]:
                rep_metrics["obj_z_k"] = admm_stats["obj_z_k"]
            if rep_metrics:
                self._log_train_step("Repair", step, total_steps, rep_metrics)

            step_progress.close()
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(step, total_steps)
            print("    Time Est.: Elapsed[{}], Remain[{}]".format(elapsed_time_str, remain_time_str))
            del rollout_cache, dist_batch, metadata_batch, rollout_data
            del gen_metrics_list, rep_metrics_list, gen_rollout_stats, rep_rollout_stats, gen_metrics, rep_metrics, step_progress

            all_done = (step == total_steps)
            model_save_checkpoint = self.trainer_params['model_save_checkpoint']
            validation_interval = self.trainer_params['validation_interval']

            if step == 1 or (step % validation_interval == 0):
                val_episodes = self.trainer_params["validation_warmup_episodes"]
                if all_done:
                    val_episodes = self.trainer_params["validation_final_episodes"]
                score, gap, unique_ratio, bpd, infeas_sol, infeas_inst, sample_records, metrics_summary = self.validation(
                    val_data,
                    opt_sol,
                    step,
                    val_episodes,
                    collect_details=all_done,
                )
                self.validation_steps.append(step)
                self.result_log["val_score"].append(score)
                self.result_log["val_gap"].append(gap)
                self.result_log["val_unique_ratio"].append(unique_ratio)
                self.result_log["val_bpd"].append(bpd if self.compute_val_bpd else None)
                self.result_log["val_infeas_sol"].append(infeas_sol)
                self.result_log["val_infeas_inst"].append(infeas_inst)
                if all_done and sample_records is not None and metrics_summary is not None:
                    self._save_validation_outputs(sample_records, metrics_summary)

                score_image_prefix = '{}/latest_val_score'.format(self.log_path)
                gap_image_prefix = '{}/latest_val_gap'.format(self.log_path)
                x_series = [self.validation_steps]
                y_score = [self.result_log["val_score"]]
                y_gap = [self.result_log["val_gap"]]
                labels = [val_problem]

                show(x_series, y_score, labels, title="Validation Score", xdes="Step", ydes="Score", path="{}.pdf".format(score_image_prefix))
                show(x_series, y_gap, labels, title="Validation Opt. Gap", xdes="Step", ydes="Opt. Gap (%)", path="{}.pdf".format(gap_image_prefix))

                show(x_series, y_score, labels, title="Validation Score", xdes="Step", ydes="Score", path="{}.png".format(score_image_prefix))
                show(x_series, y_gap, labels, title="Validation Opt. Gap", xdes="Step", ydes="Opt. Gap (%)", path="{}.png".format(gap_image_prefix))

                if metrics_summary is not None:
                    self._update_metric_artifacts(step, metrics_summary)

                if score < best_result:
                    print("Saving best model")
                    checkpoint_dict = {
                        'step': step,
                        'problem': self.args.problem,
                        'net_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log
                    }
                    if self.rep_net is not None:
                        checkpoint_dict['rep_net_state_dict'] = self.rep_net.state_dict()
                    if self.rep_optimizer is not None:
                        checkpoint_dict['rep_optimizer_state_dict'] = self.rep_optimizer.state_dict()
                    if self.rep_scheduler is not None:
                        checkpoint_dict['rep_scheduler_state_dict'] = self.rep_scheduler.state_dict()
                    torch.save(checkpoint_dict, '{}/best_model.pt'.format(self.log_path))
                    best_result = score
                elif all_done or (step % model_save_checkpoint == 0):
                    print(f"Saving checkpoint {step}")
                    checkpoint_dict = {
                        'step': step,
                        'problem': self.args.problem,
                        'net_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log
                    }
                    if self.rep_net is not None:
                        checkpoint_dict['rep_net_state_dict'] = self.rep_net.state_dict()
                    if self.rep_optimizer is not None:
                        checkpoint_dict['rep_optimizer_state_dict'] = self.rep_optimizer.state_dict()
                    if self.rep_scheduler is not None:
                        checkpoint_dict['rep_scheduler_state_dict'] = self.rep_scheduler.state_dict()
                    torch.save(checkpoint_dict, '{}/step-{}.pt'.format(self.log_path, step))

            self.scheduler.step()
            if self.rep_scheduler is not None:
                self.rep_scheduler.step()
     

def args2dict(args):
    env_params = {"problem_size": args.problem_size, 
                  "pomo_size": args.train_generate, # pomo_size
                  "device": args.device,
                  "problem": args.problem,
                  "hardness": args.hardness
    }
    if args.k_sparse_denom <= 1:
        k_sparse = max(args.problem_size - 1, 1)
    else:
        k_sparse = max(1, args.problem_size // args.k_sparse_denom)
    model_params = {
        "k_sparse": k_sparse,
        "device": args.device,
        "train_generate": args.train_generate,
        "val_generate": args.val_generate,
        "loss": args.loss,
    }
    
    beta_min_map = {50: 50, 100: 200, 200: 500, 400: 500, 500: 500, 1000: 2000}
    beta_max_map = {50: 500, 100: 1000, 200: 2000, 400: 2000, 500: 2000, 1000: 2000}
    default_beta_min = 50
    default_beta_max = 100
    beta_min_from_map = args.beta_min is None
    beta_max_from_map = args.beta_max is None
    if beta_min_from_map:
        args.beta_min = beta_min_map.get(args.problem_size, default_beta_min)
    if beta_max_from_map:
        args.beta_max = beta_max_map.get(args.problem_size, default_beta_max)

    optimizer_params = {
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "alpha_schedule_params": (args.alpha_min, args.alpha_max, args.alpha_flat_epochs),
        "beta_schedule_params": (args.beta_min, args.beta_max, args.beta_flat_epochs),
        "beta_from_map": {
            "min_from_map": beta_min_from_map,
            "max_from_map": beta_max_from_map,
        },
    }
    trainer_params = {
        "steps": args.steps,
        "rollout_batch_size": args.rollout_batch_size,
        "validation_interval": args.validation_interval,
        "validation_warmup_episodes": args.validation_warmup_episodes,
        "validation_final_episodes": args.validation_final_episodes,
        "model_save_checkpoint": args.model_save_checkpoint,
        "checkpoint": args.checkpoint,
    }

    return env_params, model_params, optimizer_params, trainer_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGFN for CVRPTW")
    # env_params
    parser.add_argument('--problem', type=str, default="TSPTW",
                        choices=["TSP", "TSPTW"])
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=5, help="the number of start node, should <= problem size")
    parser.add_argument('--hardness', type=str, default="medium", choices=["easy", "medium", "hard"])

    # model_type
    parser.add_argument('--model_type', type=str, default="AGFN",
                        choices=["SINGLE", "MTL", "MOE", "MOE_LIGHT", "AGFN"])
    parser.add_argument('--encoder_type', type=str, default="gnn", choices=["gnn", "transformer", "single_model"], help='type of encoder')
    parser.add_argument('--decoding_type', type=str, default="nar",
                        choices=["ar", "nar", "single_model"])
    
    # https://arxiv.org/pdf/2410.21066
    # Therefore, we remove the starting node stipulation in POMO and instead sample n solutions to calculate the baseline.
    parser.add_argument('--train_use_pomo', type=lambda x: str(x).lower() == "true", default=False,
                        help="If true, force first visited nodes(POMO-style) after depot during training rollout")
    parser.add_argument('--val_use_pomo', type=lambda x: str(x).lower() == "true", default=False,
                        help="If true, apply POMO-style starts during validation (pomo size = val_generate)")
    parser.add_argument('--sm_encoder_layer_num', type=int, default=6, help="single_model encoder layer count")
    parser.add_argument('--sm_head_num', type=int, default=8, help="single_model attention heads")
    parser.add_argument('--sm_qkv_dim', type=int, default=16, help="single_model qkv dimension per head")
    parser.add_argument('--sm_ff_hidden_dim', type=int, default=512, help="single_model FF hidden dimension")
    parser.add_argument('--sm_logit_clipping', type=float, default=10.0, help="single_model tanh logit clipping")
    parser.add_argument('--sm_tw_normalize', type=lambda x: str(x).lower() == "true", default=False, help="single_model time window normalization flag")
    parser.add_argument('--embedding_dim', type=int, default=128, help="node embedding dimension")

    parser.add_argument('--gen_matrix_output_space', type=str, default="probs", choices=["logit", "probs"],
                        help="Output space for generator NAR matrix scores (probs applies sigmoid, logit skips it)")
    
    # rollout/inference tw mask
    parser.add_argument('--tw_mask_train', type=str, default="off", choices=["on", "off"],
                        help="Enable (on) or disable (off) time-window mask during training rollout")
    parser.add_argument('--tw_mask_val', type=str, default="off", choices=["on", "off"],
                        help="Enable (on) or disable (off) time-window mask during validation")
    
    # loss and baseline
    parser.add_argument('--loss', type=str, default="gflow",
                        choices=["gflow", "reinforce"],
                        help="Training objective for generator, reinforce maximizes -penalty reward")
    parser.add_argument('--baseline_type', type=str, default="per_instance",
                        choices=["per_instance", "per_batch"],
                        help="Type of baseline for loss")
    parser.add_argument('--select_best_mode', type=str, default="feasible_min_cost",
                        choices=["feasible_min_cost", "best_R"],
                        help="Selection rule for best tour after each kernel step")
    parser.add_argument('--occ_time_bins', type=int, default=16,
                        help="Number of normalized time bins for the occupancy consensus tensor")
    parser.add_argument('--occ_aux_weight', type=float, default=0.05,
                        help="Weight of the direct occupancy alignment loss added to each kernel update")

    
    # instance augmentation
    parser.add_argument('--train_aug', type=lambda x: str(x).lower() == "true", default=False,
                        help="Apply 8-fold coordinate symmetry augmentation during training data generation")
    parser.add_argument('--val_aug', type=lambda x: str(x).lower() == "true", default=True,
                        help="Apply 8-fold coordinate symmetry augmentation during validation; aug routes are grouped per instance")

    # data_params
    parser.add_argument('--k_sparse_denom', type=int, default=1)
    parser.add_argument('--train_generate', type=int, default=50, help="number of rollout samples per input (training)")
    parser.add_argument('--val_generate', type=int, default=50, help="number of samples per input (validation/inference)")    

    # optimizer_params
    parser.add_argument('--optimizer', type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_min', type=float, default=1e-4, help="minimum LR for generator scheduler (default: 0.1 * lr)")
    
    parser.add_argument("--alpha_min", type=float, default=0.8, help='alpha')
    parser.add_argument("--alpha_max", type=float, default=1.0, help='alpha')
    parser.add_argument("--alpha_flat_epochs", type=int, default=5, help='alpha flat steps')
    
    parser.add_argument("--beta_min", type=float, default=50, help='Beta min')
    parser.add_argument("--beta_max", type=float, default=500, help='Beta max')
    parser.add_argument("--beta_flat_epochs", type=int, default=10, help='Beta flat steps') # defualt: 5

    # trainer_params
    parser.add_argument('--steps', type=int, default=3000, help="total training steps")
    parser.add_argument('--rollout_batch_size', type=int, default=128,
                        help="rollout batch size per step (B)")
    parser.add_argument('--validation_interval', type=int, default=50,
                        help="validation interval in steps")
    parser.add_argument('--validation_warmup_episodes', type=int, default=10,
                        help="number of validation instances used for non-final checkpoints")
    parser.add_argument('--validation_final_episodes', type=int, default=1000,
                        help="number of validation instances used at the final checkpoint")
    parser.add_argument('--val_bpd', type=str, choices=["on", "off"], default="off",
                        help="Enable (on) or disable (off) BPD computation during validation; disabling saves time.")
    parser.add_argument('--validation_iter', type=int, default=1,
                        help="Total validation iterations; 0 uses iter1 only")
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help="Number of validation instances processed together in one inference batch.")
    parser.add_argument('--model_save_checkpoint', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default=None, help="resume training")

    # repair steps
    parser.add_argument('--K', type=int, default=48, help="number of repair steps")
    parser.add_argument('--rho_x_init', type=float, default=0.5,
                        help="Initial consensus penalty for cost kernel")
    parser.add_argument('--rho_z_init', type=float, default=0.5,
                        help="Initial consensus penalty for repair kernel")
    parser.add_argument('--consensus_weight_update_interval', type=int, default=1,
                        help="Consensus weight update interval T_f")
    parser.add_argument('--consensus_correlation_threshold', type=float, default=0.1,
                        help="Correlation threshold epsilon_cor for consensus weight update")
    parser.add_argument('--grad_accum', type=lambda x: str(x).lower() == "true", default=False,
                        help="If true, accumulate gradients across k-steps and optimizer-step once per outer step")
    
    # penalty weights
    parser.add_argument('--cost_penalty_type', type=str, default="delta_value",
                        choices=["delta_value", "delta_rate"],
                        help="Cost term type: delta_value uses raw C, delta_rate uses raw C scaled by cost_penalty_weight_rate")
    parser.add_argument('--cost_penalty_weight', type=float, default=1.0,
                        help="Weight for cost in generator training")
    parser.add_argument('--cost_penalty_weight_rate', type=float, default=0.1,
                        help="Weight for cost when cost_penalty_type=delta_rate")
    parser.add_argument('--tw_penalty_weight', type=float, default=1, # 1
                        help="Deprecated; sets gflow TW penalty weight when specific value is not provided")
    parser.add_argument('--tw_penalty_weight_gflow', type=float, default=1,
                        help="Weight for time-window lateness penalty in gflow repair training")

    # settings (e.g., GPU)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--log_dir', type=str, default="./2_results")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--occ_gpu', type=float, default=0., help="occupy (X)%% GPU memory in advance.")

    args = parser.parse_args()
    args.val_bpd = args.val_bpd.lower() == "on"
    if args.lr_min is None:
        args.lr_min = args.lr * 0.1
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_timestamp = run_timestamp
    args.log_path = os.path.join(args.log_dir, f"{args.model_type}_{args.problem.lower()}_{args.problem_size}_{run_timestamp}")
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    log_file_name = f"{os.path.splitext(os.path.basename(__file__))[0]}.log"
    args.log_file = setup_logging(args.log_path, log_file_name)
    print(">> Log Path: {}".format(args.log_path))
    print(">> Log File: {}".format(args.log_file))
    EPS = 1e-10
    START_NODE = 0 if args.problem.upper() == "TSPTW" else None
    # COORD_SCALE = 100.0
    # TIME_WINDOW_SCALE = 5500 * (args.problem_size // 50)

    COORD_SCALE = 100
    TIME_WINDOW_SCALE = None  # set to None to scale by depot tw_end per instance
    GEN_SCALE = 100

    TSP_FAKE_TW_END = 1e6
    
    if args.problem == "ALL" and args.model_type == "Single":
        raise ValueError("Cannot solve multiple problems with Single model, please use MTL/MOE/MOE_LIGHT instead.")

    # Decide device minimally changed
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")

    # If we want to occupy GPU memory, only do it if device is CUDA
    if args.device.type == 'cuda' and args.occ_gpu > 0:
        occumpy_mem(args)
    
    env_params, model_params, optimizer_params, trainer_params = args2dict(args)
    pp.pprint(vars(args))
    args.config_snapshot = dict(vars(args))
    seed_everything(args.seed)

    # torch.set_printoptions(threshold=1000000)
    torch.set_printoptions(threshold=10000)
    process_start_time = datetime.now(pytz.timezone("Asia/Singapore"))
    print(">> USE_CUDA: {}".format(not args.no_cuda))

    print(">> Start {} Training using {} Model ...".format(args.problem, args.model_type))
    trainer = Trainer(args=args,
                      env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)
    start = time.time()
    trainer.run()
    end = time.time() - start
    TimeEstimator.cal_time(end)
    print(">> Finish {} Training using {} Model ...".format(args.problem, args.model_type))
