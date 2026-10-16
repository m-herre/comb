"""graph_scm synthetic prior for TabICLv2 / Comb pretraining (classification).

Vendored from tabicl 2.2.0 (BSD-3, see LICENSE-tabicl): ``tabicl/prior/graph_lib`` (all non-eval modules,
flattened in dependency order), ``GraphSCM`` from ``_graph_scm.py``, the scaling helpers of ``_reg2cls.py``,
and the ``graph_scm`` path of ``_dataset.py`` (``Prior`` statics, ``GraphPrior``, ``should_filter``,
``PriorDataset``). Every RNG-consuming call is kept in upstream's order, so with the same global numpy and
torch seeds the batches are bit-identical to upstream ``PriorDataset(prior_type="graph_scm", n_jobs=1)``.

Deviations from upstream:
  * ``RandomDataset.sample`` iterates the feature groups in the fixed order ``("y", "x")``, the order that
    upstream's ``list(set(...))`` produces under ``PYTHONHASHSEED=0``; upstream depends on the hash seed.
  * Only the serial ``n_jobs=1`` loop exists (as run inside each DataLoader worker by upstream's trainer).
  * Dropped paths that are off at the stage-script settings and consume no RNG there: v1/dummy priors,
    ``run_parallel``, ``replay_small``, ``log_n_features``, separate sub-group sizes, non-CPU devices,
    the recursion-tracking print in ``Context``, and ``Dataset``'s embedding outputs.
"""

from __future__ import annotations

import collections
import functools
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import threadpoolctl
import torch
import torch.nn.functional as F
from scipy.stats import loguniform
from sklearn.ensemble import ExtraTreesRegressor
from torch import Tensor
from torch.utils.data import IterableDataset

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)

# ==================== _config ====================

@dataclass
class PriorConfig:
    """Configuration options for the prior (upstream ``graph_lib/_config.py``, without argparse helpers)."""

    add_gaussian_noise: bool = False
    use_node_feature_importance: bool = True
    use_node_importance: bool = True
    use_node_l2_norm: bool = True
    allow_act_warping: bool = False
    allow_kumaraswamy_warping: bool = True
    disallow_y_warping: bool = False
    # maximum number of discretization levels in random discretization function
    max_discretization_cardinality: int = 256
    filter_unpredictable_datasets: bool = False
    filter_unpredictable_graphs: bool = False
    min_n_nodes: int = 2
    max_n_nodes: int = 32
    cauchy_dag_offset: float = 0.0
    meta_sampling_mode: Literal["meta", "local", "global"] = "meta"
    random_matrix_types: Literal["default", "gaussian"] = "default"
    fct_types: str = "default"
    multi_fct_types: Literal["default", "concat", "agg"] = "default"
    cat_modes: str = "default"
    random_weights_types: Literal["simple", "uniform"] = "simple"
    subsample_feature_nodes: bool = True
    seq_lens_multiple_of: int = 1
    use_corrected_num_converters: bool = False
    use_corrected_cat_sizes: bool = False
    ensure_iid: bool = False
    remove_trivial_datasets: bool = False
    trivial_dataset_threshold: float = 0.05
    use_corrected_cat_meta_sampling: bool = False

# The v2 stage scripts pass --graph_noise False --allow_act_warping False --min_n_nodes 2 --max_n_nodes 32
# --cauchy_dag_offset 0.0 (all dataclass defaults) plus the two filters below.
STAGE_CONFIG = PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True)

# Shape kwargs of upstream tabicl's scripts/train_v2_clf_stage{1,2,3}.sh (max_classes is passed separately). seq_len_per_gp=True in all three is the PriorDataset default.
STAGES: Dict[str, Dict[str, Any]] = {
    "stage1": dict(batch_size=64, batch_size_per_gp=4, min_features=1, max_features=100, min_seq_len=None,
                   max_seq_len=1024, log_seq_len=False, min_train_size=0.3, max_train_size=0.9),
    "stage2": dict(batch_size=64, batch_size_per_gp=1, min_features=1, max_features=100, min_seq_len=400,
                   max_seq_len=10240, log_seq_len=True, min_train_size=0.79, max_train_size=0.81),
    "stage3": dict(batch_size=64, batch_size_per_gp=1, min_features=1, max_features=100, min_seq_len=400,
                   max_seq_len=60000, log_seq_len=True, min_train_size=0.79, max_train_size=0.81),
}

# ==================== _base ====================

class GlobalSampler:
    """Samples scalars. Mode "meta" draws from a per-name distribution whose hyperparameters are sampled once
    per name, "global" samples once per name, "local" samples independently."""

    def __init__(self, seed: Optional[int] = 0, config: Optional[PriorConfig] = None):
        self.globals = dict()
        self.rng = np.random.default_rng(seed=seed if seed is not None else np.random.randint(np.iinfo(np.int32).max))
        self.config = config or PriorConfig()

    def numerical(self, name: str, low: float, high: float, use_log: bool = False, use_int: bool = False,
                  mode: Literal["meta", "global", "local"] = "meta", boundary_mass: bool = False,
                  size: Optional[Union[int, Tuple[int]]] = None):
        assert high >= low
        if mode == "global":
            if name not in self.globals:
                self.globals[name] = self.numerical(
                    name + "__global", low, high, use_log, use_int, mode="local", boundary_mass=boundary_mass, size=size
                )
            return self.globals[name]
        elif mode == "meta":
            if self.config.meta_sampling_mode != "meta":
                return self.numerical(name, low, high, use_log, use_int, self.config.meta_sampling_mode, boundary_mass,
                                      size)
            loc = self.numerical(name + "__meta_beta-loc", 0.0, 1.0, mode="global", boundary_mass=False)
            sum = self.numerical(name + "__meta_beta-sum", 0.1, 10000.0, use_log=True, mode="global",
                                 boundary_mass=False)
            alpha = loc * sum
            beta = (1 - loc) * sum
            value = self.rng.beta(alpha, beta, size=size)
        else:
            value = self.rng.random(size=size)

        if use_log:
            assert low > 0
            low = np.log(low)
            high = np.log(high)

        if boundary_mass:
            value = np.clip(1.2 * value - 0.1, 0.0, 1.0)

        value = low + (high - low) * value

        if use_log:
            value = np.exp(value)
        if use_int:
            value = np.rint(value).astype(int)
        return value

    def randint(self, name: str, low: int, high: int, use_log: bool = False,
                mode: Literal["meta", "global", "local"] = "meta", boundary_mass: bool = False) -> int:
        value = self.numerical(name, low, high, use_log=use_log, mode=mode, boundary_mass=boundary_mass)
        return np.clip(np.floor(value).astype(int), low, high - 1)

    def categorical(self, name: str, n_categories: int, mode: Literal["meta", "global", "local"] = "meta") -> int:
        if mode == "global":
            if name not in self.globals:
                self.globals[name] = self.categorical(name + "__global", n_categories, mode="local")
            return self.globals[name]
        elif mode == "meta":
            if self.config.meta_sampling_mode != "meta":
                return self.categorical(name, n_categories, self.config.meta_sampling_mode)
            ext_name = name + "__meta_weights"
            # upstream TabICLv2 queries the wrong key, so the weights are resampled on every call (with a
            # throwaway Context() that draws its seed from the global numpy RNG); kept for bit parity
            query_name = ext_name if self.config.use_corrected_cat_meta_sampling else name
            if query_name not in self.globals:
                self.globals[ext_name] = SimpleRandomWeights(Context()).sample(1, n_categories).squeeze(0)
                self.globals[ext_name] /= self.globals[ext_name].sum()

            return np.random.choice(np.arange(n_categories), p=self.globals[ext_name].numpy())

        return np.random.randint(n_categories)

    def choice(self, name: str, categories: List[Any], mode: Literal["meta", "global", "local"] = "meta") -> Any:
        return categories[self.categorical(name, len(categories), mode)]

    def boolean(self, name: str, mode: Literal["meta", "global", "local"] = "meta") -> bool:
        return self.categorical(name, 2, mode) == 1

class Context:
    """The sampler and config shared by all components of one dataset (CPU only)."""

    def __init__(self, sampler: Optional[GlobalSampler] = None, config: Optional[PriorConfig] = None,
                 seed: Optional[int] = None):
        self.device = "cpu"
        self.sampler = sampler or GlobalSampler(seed=seed, config=config)
        self.config = config or PriorConfig()

class PriorComponent:
    """Base class for components of the prior."""

    def __init__(self, context: Context):
        self.context = context
        self.sampler = context.sampler
        self.device = context.device
        self.config = context.config

class RandomTransformer(PriorComponent):
    """Acts like a scikit-learn transformer: the first ``__call__`` fits, later calls only transform."""

    def __init__(self, context: Context):
        super().__init__(context)
        self.fitted = False

    def _fit(self, *args, **kwargs):
        pass

    def _transform(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Any:
        if not self.fitted:
            self._fit(*args, **kwargs)
            self.fitted = True
        return self._transform(*args, **kwargs)

class RandomTensorTransformer(RandomTransformer):
    """RandomTransformer taking one tensor (n_samples, in_features)."""

class RandomTensorSequential(RandomTensorTransformer):
    def __init__(self, context: Context, tfms: List[RandomTensorTransformer]):
        super().__init__(context)
        self.tfms = tfms

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        for tfm in self.tfms:
            x = tfm(x)
        return x

class FeatureSpec:
    """One column: its group ("x" or "y") and its cardinality (0 = numerical)."""

    def __init__(self, *, group: str = "x", cat_size: int = 0):
        if "_" in group:
            raise ValueError("group must not contain an underscore")
        self.group = group
        self.cat_size = cat_size

class DatasetProperties:
    def __init__(self, n_train: int, n_test: int, cat_sizes: Dict[str, List[int]]):
        self.n_train = n_train
        self.n_test = n_test
        self.feature_specs = {f"{group}_{i}": FeatureSpec(group=group, cat_size=cat_size)
                              for group in cat_sizes for i, cat_size in enumerate(cat_sizes[group])}

class Dataset:
    def __init__(self, tensors: Dict[str, torch.Tensor], feature_specs: Dict[str, FeatureSpec], **kwargs):
        self.tensors = tensors
        self.feature_specs = feature_specs
        self.kwargs = kwargs

    def get_concat_tensors(self) -> Dict[str, torch.Tensor]:
        """Concatenates the columns into "x_num", "x_cat", "y_num", "y_cat"."""
        tensor_lists = collections.defaultdict(lambda: [])
        for feature_name, feature_spec in self.feature_specs.items():
            feat_type = "cat" if feature_spec.cat_size > 1 else "num"
            tensor_lists[f"{feature_spec.group}_{feat_type}"].append(self.tensors[feature_name])
        return {name: torch.cat(tensors, dim=-1) for name, tensors in tensor_lists.items()}

# ==================== _properties ====================

def sample_categorical_sizes(n_features: int, context: Context, max_cat_size: int = 100) -> List[int]:
    """Sample categorical sizes for features."""
    cat_fraction = np.clip(np.random.uniform(-0.5, 1.2), 0.0, 1.0)
    n_cat = round(n_features * cat_fraction)
    n_cont = n_features - n_cat

    n_local = context.sampler.randint("cat_sizes_n_local", 0, n_cat + 1, mode="local", boundary_mass=True)
    n_meta = n_cat - n_local
    if context.config.use_corrected_cat_sizes:
        cat_size_limit = context.sampler.randint("cat_size_limit", 2, max_cat_size + 1, mode="local", use_log=True)
    else:
        cat_size_limit = context.sampler.randint("cat_size_limit", 2, min(10, max_cat_size + 1), use_log=True)
    # only allow large categories in "local" mode, otherwise runtimes for some datasets can get very large
    return (
        [0] * n_cont
        + [context.sampler.randint("cat_size", 2, cat_size_limit + 1, use_log=True, mode="local")
           for _ in range(n_local)]
        + [context.sampler.randint("cat_size", 2, min(10, cat_size_limit + 1), use_log=True, mode="meta")
           for _ in range(n_meta)]
    )

# ==================== _weights ====================

class RandomWeights(PriorComponent):
    """Random nonnegative weight vectors."""

    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        if self.config.random_weights_types == "simple":
            return SimpleRandomWeights(self.context).sample(n_batch, n)
        elif self.config.random_weights_types == "uniform":
            return UniformRandomWeights(self.context).sample(n_batch, n)
        else:
            raise ValueError(f"Unknown option {self.config.random_weights_types=}")

class SimpleRandomWeights(RandomWeights):
    """Random polynomial decay + Gaussian noise on log-values."""

    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        # log-space weights: decay between n^{-log n} = 1/e and n^{-6}
        decay_rate = torch.as_tensor(self.sampler.numerical("simple_random_weights_decay_rate", 0.1 / np.log(1 + n), 6,
                                                            use_log=True, boundary_mass=True, size=n_batch),
                                     dtype=torch.float32, device=self.device)
        base_weights = torch.linspace(1.0, n, n, device=self.device)
        log_weights = -decay_rate[:, None] * torch.log(base_weights)
        std_scale = torch.as_tensor(
            self.sampler.numerical("simple_random_weights_std_scale", 0.0001, 10, use_log=True, size=n_batch),
            dtype=torch.float32, device=self.device)
        logits = log_weights + std_scale[:, None] * torch.randn(n_batch, n, device=self.device)
        logits = torch.stack([logits[i, torch.randperm(n)] for i in range(n_batch)], dim=0)
        weights = torch.as_tensor(torch.softmax(logits, dim=-1), dtype=torch.float32)
        weights = np.sqrt(n) * (weights / weights.norm(dim=-1, keepdim=True))
        return weights

class UniformRandomWeights(RandomWeights):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        return torch.rand(n_batch, n)

# ==================== _activation ====================

class RandomActivation(RandomTensorTransformer):
    """Random dimension-preserving function, wrapped in standardization and a random rescale."""

    def _fit(self, x: torch.Tensor):
        act_types = [RandomTorchActivation] * 4 + [RandomPowerReluActivation, RandomPowerActivation]
        base_act = self.sampler.choice("act_type", act_types)(self.context)
        self.act_ = RandomTensorSequential(
            self.context, [Standardize(self.context), RandomRescale(self.context), base_act, Standardize(self.context)])

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_(x)

def sgn(x):
    return 2 * (x >= 0.0).float() - 1.0

def heaviside(x):
    return (x >= 0.0).float()

def unit_interval_indicator(x):
    return (torch.abs(x) <= 1.0).float()

def rbf(x):
    return torch.exp(-(x**2))

def identity(x):
    return x

def log_clip_abs(x):
    return torch.log(torch.clamp(torch.abs(x), min=1e-6))

def is_max(x):
    return (x == torch.max(x, dim=-1, keepdim=True).values).float()

def rank(x):
    perm = torch.argsort(x, dim=-1)
    src = torch.arange(x.shape[-1], device=x.device)[None].expand(x.shape[0], -1)
    perms_inv = torch.empty_like(perm, device=x.device)
    perms_inv = torch.scatter(perms_inv, dim=1, index=perm, src=src)
    return perms_inv.float()

def argsort_float(x):
    return torch.argsort(x, dim=-1).float()

def modulo_act(x):
    return x - torch.floor(x)

acts = [
    # TabICLv1
    torch.tanh, F.leaky_relu, F.elu, identity, F.selu, F.silu, F.relu, F.softplus, F.relu6, F.hardtanh, sgn, heaviside,
    rbf, torch.exp, unit_interval_indicator, torch.sin, torch.square, torch.abs,
    # TabPFNv2
    log_clip_abs, rank, F.sigmoid, torch.round, modulo_act,
    # extra
    functools.partial(F.softmax, dim=-1), is_max, argsort_float, F.logsigmoid,
]

class RandomActivationPlain(RandomActivation):
    """Random activation without the standardization / rescaling."""

    def _fit(self, x: torch.Tensor):
        act_types = [RandomTorchActivation] * 8 + [RandomPowerReluActivation, RandomPowerActivation,
                                                   RandomNegPowerActivation, RandomIntPowerActivation]
        self.act_ = self.sampler.choice("act_type_plain", act_types)(self.context)

class RandomTorchActivation(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.act_ = self.sampler.choice("acts_torch", acts)

class RandomPowerReluActivation(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.exponent_ = self.sampler.numerical("relu_random_power_exponent", 0.1, 10.0, use_log=True)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** self.exponent_

class RandomPowerActivation(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.exponent_ = self.sampler.numerical("random_power_exponent", 0.1, 10.0, use_log=True)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return sgn(x) * (x.abs() ** self.exponent_)

class RandomNegPowerActivation(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.exponent_ = -self.sampler.numerical("neg_random_power_exponent", 0.1, 10.0, use_log=True)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x.abs() + 1e-3) ** self.exponent_

class RandomIntPowerActivation(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.exponent_ = self.sampler.randint("rand_int_power_exponent", 2, 6)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return x**self.exponent_

class Standardize(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.mean_ = x.mean(dim=0, keepdim=True)
        self.inv_std_ = 1.0 / (x.std(dim=0, keepdim=True, correction=0) + 1e-4)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_) * self.inv_std_

class L2Normalize(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.factor_ = 1.0 / (x.square().sum(dim=-1).mean().sqrt() + 1e-8)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.factor_ * x

class RandomRescale(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.scale_ = self.sampler.numerical("rr_scale", 1e-0, 1e1, use_log=True)
        # random datapoints as shifts, so that activations like ReLU are not always zero
        self.bias_ = -x[torch.randint(x.shape[0], size=(x.shape[1],)), torch.arange(x.shape[1])][None, :]

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale_ * (x + self.bias_)

class KumaraswamyWarping(RandomActivation):
    """Min-max scaling followed by the Kumaraswamy transform."""

    def _fit(self, x: torch.Tensor):
        self.min_ = x.min(dim=0).values
        self.max_ = x.max(dim=0).values
        self.a_ = self.sampler.numerical("kumaraswamy_a", 0.2, 5, use_log=True)
        self.b_ = self.sampler.numerical("kumaraswamy_b", 0.2, 5, use_log=True)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp((x - self.min_) / (self.max_ - self.min_ + 1e-30), 0.0, 1.0)
        return 1.0 - (1.0 - x ** self.a_) ** self.b_

# ==================== _matrix ====================

def row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / (1e-6 + matrix.norm(dim=-1, keepdim=True))

class RandomMatrix(PriorComponent):
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        if self.config.random_matrix_types == "default":
            matrix_types = [RandomGaussianMatrix, RandomWeightsMatrix, RandomSingularValuesMatrix, RandomKernelMatrix,
                            RandomActivationMatrix]
        elif self.config.random_matrix_types == "gaussian":
            matrix_types = [RandomGaussianMatrix]
        else:
            raise ValueError(f"Unknown {self.config.random_matrix_types=}")
        matrix_type = self.sampler.choice("random_matrix_type", matrix_types)
        mat = matrix_type(self.context).sample(n_batch, n, m)
        mat += 1e-6 * RandomGaussianMatrix(self.context).sample(n_batch, n, m)
        return row_normalize(mat)

class SimpleRandomMatrix(RandomMatrix):
    """RandomMatrix without RandomActivationMatrix (avoids infinite recursion inside it)."""

    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        matrix_types = [RandomGaussianMatrix, RandomWeightsMatrix, RandomSingularValuesMatrix, RandomKernelMatrix]
        matrix_type = self.sampler.choice("simple_random_matrix_type", matrix_types)
        mat = matrix_type(self.context).sample(n_batch, n, m)
        mat += 1e-6 * RandomGaussianMatrix(self.context).sample(n_batch, n, m)
        return row_normalize(mat)

class RandomGaussianMatrix(RandomMatrix):
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        return torch.randn(n_batch, n, m, device=self.device)

class RandomWeightsMatrix(RandomMatrix):
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        matrix = RandomWeights(self.context).sample(n_batch * n, m).reshape(n_batch, n, m)
        matrix *= torch.randn_like(matrix)
        return row_normalize(matrix)

class RandomSingularValuesMatrix(RandomMatrix):
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        k = min(n, m)
        U = RandomGaussianMatrix(self.context).sample(n_batch, n, k)
        diag = RandomWeights(self.context).sample(n_batch, k)
        V = RandomGaussianMatrix(self.context).sample(n_batch, k, m)
        return (U * diag[:, None, :]) @ V

class RandomKernelMatrix(RandomMatrix):
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        d = 3
        points = RandomGaussianMixturePoints(self.context).sample(n_batch * (n + m), d).reshape(n_batch, n + m, d)
        dists = torch.cdist(points[:, :n], points[:, n:])  # Laplace kernel
        dists *= self.sampler.numerical("random_kernel_matrix_dist_factor", 0.1, 10.0, use_log=True)
        return torch.exp(-dists) * torch.sign(RandomGaussianMatrix(self.context).sample(n_batch, n, m))

class RandomActivationMatrix(RandomMatrix):
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        # RandomActivation cannot handle batch size 1 (standardize), so use the plain one
        matrix = SimpleRandomMatrix(self.context).sample(n_batch, n, m).reshape(n_batch, n * m)
        matrix = RandomActivationPlain(self.context)(matrix).reshape(n_batch, n, m)
        matrix += 1e-3 * torch.randn_like(matrix)
        return matrix

# ==================== _function ====================

class RandomFunction(RandomTensorTransformer):
    """Random function of one tensor (n_samples, in_features) -> (n_samples, out_features)."""

    def __init__(self, context: Context, in_features: int, out_features: int):
        super().__init__(context)
        self.in_features = in_features
        self.out_features = out_features

    def _fit(self, x: torch.Tensor):
        fct_types_dict = {"mlp": RandomMLPFunction, "tree": RandomTreeFunction, "disc": RandomDiscretizationFunction,
                          "lin": RandomLinearFunction, "quad": RandomQuadraticFunction, "gp": RandomGPFunction,
                          "em": RandomEMAssignmentFunction, "prod": RandomProductFunction}
        presets_dict = {"default": "mlp,tree,disc,lin,quad,gp,em,prod", "tabpfnv2": "mlp,tree,disc"}
        fct_types: List[type] = []
        for name in self.config.fct_types.split(","):
            if name in presets_dict:
                fct_types.extend([fct_types_dict[type_name] for type_name in presets_dict[name].split(",")])
            else:
                fct_types.append(fct_types_dict[name])

        fct_type = self.sampler.choice("random_function_fct_type", fct_types)
        self.fct_ = fct_type(self.context, self.in_features, self.out_features)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fct_(x)

class CheapRandomFunction(RandomFunction):
    """Samples only from the cheaper function types (also avoids infinite recursion)."""

    def _fit(self, x: torch.Tensor):
        fct_types = [RandomGPFunction, RandomTreeFunction, RandomDiscretizationFunction, RandomLinearFunction,
                     RandomQuadraticFunction]
        fct_type = self.sampler.choice("cheap_random_function_fct_type", fct_types)
        self.fct_ = fct_type(self.context, self.in_features, self.out_features)

class RandomMLPFunction(RandomFunction):
    """Random MLP, optionally with activations at the start and end."""

    def _fit(self, x: torch.Tensor):
        width = self.sampler.randint("nn_width", 1, 128, use_log=True)
        n_layers = self.sampler.randint("nn_n_layers", 1, 4, use_log=True)
        self.start_act_ = self.sampler.boolean("nn_start_act")
        self.end_act_ = self.sampler.boolean("nn_end_act")
        layer_sizes = [x.shape[-1]] + ([width] * (n_layers - 1)) + [self.out_features]
        self.linears_ = [RandomLinearFunction(self.context, n_in, n_out)
                         for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.acts_ = [RandomActivation(self.context)
                      for _ in range(n_layers - 1 + int(self.start_act_) + int(self.end_act_))]

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.acts_
        if self.start_act_:
            x = acts[0](x)
            acts = acts[1:]

        if self.end_act_:
            for lin, act in zip(self.linears_, acts):
                x = act(lin(x))
        else:
            for lin, act in zip(self.linears_[:-1], acts):
                x = act(lin(x))
            x = self.linears_[-1](x)
        return x

class RandomTreeFunction(RandomFunction):
    """Ensemble of oblivious decision trees (as in CatBoost) with random leaf values."""

    def _fit(self, x: torch.Tensor):
        self.feature_imp_ = torch.clamp(x.std(dim=0, correction=0), 1e-8)
        self.feature_imp_[~torch.isfinite(self.feature_imp_)] = 1e-8
        self.n_trees_ = self.sampler.numerical("rt_n_trees", 1, 128, use_log=True, use_int=True)
        self.depth_ = self.sampler.randint("rt_depth", 1, 8)
        self.n_leaves_ = 2**self.depth_
        self.n_splits_ = self.n_trees_ * self.depth_
        self.n_leaf_values_ = self.n_trees_ * self.n_leaves_
        self.split_dims_ = torch.multinomial(self.feature_imp_, self.n_splits_, replacement=True).to(self.device)
        self.split_points_ = x[torch.randint(x.shape[0], size=(self.n_splits_,), device=self.device), self.split_dims_]
        self.leaf_values_ = RandomGaussianPoints(self.context).sample(self.n_leaf_values_, self.out_features).reshape(
            self.n_trees_, self.n_leaves_, self.out_features)
        self.idx_multipliers_ = 2 ** torch.arange(self.depth_, dtype=torch.long, device=self.device)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        split_sides = x[:, self.split_dims_] > self.split_points_
        split_sides = split_sides.reshape(x.shape[0], self.n_trees_, self.depth_)
        leaf_idxs = torch.einsum("btd,d->bt", split_sides.float(), self.idx_multipliers_.float()).long()
        tree_idxs = torch.arange(self.n_trees_, dtype=torch.long, device=x.device).expand(x.shape[0], self.n_trees_)
        result = self.leaf_values_[tree_idxs, leaf_idxs]
        return result.mean(dim=1)

class RandomDiscretizationFunction(RandomFunction):
    """Discretizes to the closest of a subset of points, then applies a linear function."""

    def __init__(self, context: Context, in_features: int, out_features: int, n_centers: Optional[int] = None):
        super().__init__(context, in_features, out_features)
        self.n_centers = n_centers

    def _fit(self, x: torch.Tensor):
        if x.shape[0] < 2:
            assert x.shape[0] == 1
            self.n_centers = 1
            self.centers_ = x
        else:
            if self.n_centers is None:
                self.n_centers = self.sampler.randint("rand_discr_n_centers", 2, min(
                    x.shape[0], self.config.max_discretization_cardinality), use_log=True)
            else:
                self.n_centers = min(self.n_centers, x.shape[0])
            self.centers_ = x[torch.randperm(len(x), device=x.device)[: self.n_centers]]

        self.p_ = self.sampler.numerical("cat_disc_p", 0.5, 4.0, use_log=True)
        self.targets_ = RandomLinearFunction(self.context, self.in_features, self.out_features)(self.centers_)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.centers_, self.p_)
        closest_idx = dists.argmin(dim=-1)
        return self.targets_[closest_idx]

class RandomGPFunction(RandomFunction):
    """Gaussian-process function (random Fourier features) with a random kernel and input transform."""

    def __init__(self, context: Context, in_features: int, out_features: int, n_frequencies: int = 256):
        super().__init__(context, in_features, out_features)
        self.n_frequencies = n_frequencies

    def _fit(self, x: torch.Tensor):
        a = self.sampler.numerical("gp_a", 2.0, 20.0, use_log=True)  # global decay rate a > 1

        if self.sampler.boolean("generalized_gp_use_product_kernel"):
            # inverse-CDF samples of the density (a-1)(1+x)^{-a} on [0, inf); axis-aligned kernel
            u = torch.rand(self.in_features, self.n_frequencies, device=x.device)
            u = torch.clamp(u, 1e-6, 1 - 1e-6)
            invcdf = torch.pow(u, 1 / (1 - a)) - 1.0
            self.freqs_ = invcdf
        else:
            u = torch.rand(self.n_frequencies, device=x.device)
            u = torch.clamp(u, 1e-6, 1 - 1e-6)
            invcdf = torch.pow(u, 1 / (1 - a)) - 1.0
            self.freqs_ = torch.randn(self.in_features, self.n_frequencies, device=x.device)
            self.freqs_ *= invcdf[None, :] / self.freqs_.norm(dim=0, keepdim=True)
            input_tfm = RandomGaussianMatrix(self.context).sample(1, self.in_features, self.in_features)[0]
            input_tfm = input_tfm * RandomWeights(self.context).sample(1, self.in_features).t()
            input_tfm *= self.sampler.numerical("rand_gen_gp_scale", 0.5, 10.0, use_log=True)
            self.freqs_ = input_tfm @ self.freqs_

        self.bias_ = 2 * np.pi * torch.rand(1, self.n_frequencies, device=x.device)
        self.weights_ = torch.randn(self.n_frequencies, self.out_features) / np.sqrt(self.n_frequencies)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x @ self.freqs_ + self.bias_) @ self.weights_

class RandomLinearFunction(RandomFunction):
    def _fit(self, x: torch.Tensor):
        # RandomMatrix is asymmetric and assumes the input on the right side, so transpose
        self.matrix_ = RandomMatrix(self.context).sample(1, self.out_features, self.in_features).squeeze(0).t()

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix_

class RandomQuadraticFunction(RandomFunction):
    """Random quadratic function of at most 20 (subsampled) input features."""

    def _fit(self, x: torch.Tensor):
        max_n_features = 20
        if x.shape[1] > max_n_features:
            self.idxs_ = torch.randperm(x.shape[1])[:max_n_features]
        else:
            self.idxs_ = torch.arange(x.shape[1])
        self.n_feat_ = len(self.idxs_) + 1  # +1 for constant term
        self.tensor_3d_ = RandomMatrix(self.context).sample(self.out_features, self.n_feat_, self.n_feat_)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x[:, self.idxs_], torch.ones(x.shape[0], 1)], dim=-1)
        return torch.einsum("oij,bi,bj->bo", self.tensor_3d_, x, x)

class RandomEMAssignmentFunction(RandomFunction):
    """Soft cluster assignment, roughly as in the EM algorithm."""

    def _fit(self, x: torch.Tensor):
        n_ind = self.sampler.randint("random_em_components", 2, max(16, 2 * self.out_features) + 1, use_log=True)
        self.x_ind_ = x[torch.randint(x.shape[0], size=(n_ind,))]
        self.x_ind_ += 1.0 * torch.randn_like(self.x_ind_)
        self.stds_ = torch.exp(self.sampler.numerical("gen_em_assignment_std_factor", 0.0, 1.0) * torch.randn(n_ind))
        self.consts = -torch.log(2 * torch.pi * self.stds_[None, :] ** 2) / 2
        self.out_func_ = RandomLinearFunction(self.context, in_features=n_ind, out_features=self.out_features)
        self.p_ = self.sampler.numerical("gen_em_assignment_p_inv", 1.0, 4.0, use_log=True)
        self.q_ = self.sampler.numerical("gen_em_assignment_q", 1.0, 2.0)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.x_ind_, p=self.p_)
        logits = self.consts - torch.clamp(dists / self.stds_[None, :], min=0.0) ** self.q_
        result = torch.softmax(logits, dim=-1)
        return self.out_func_(result)

class RandomProductFunction(RandomFunction):
    """Product of two cheap random functions."""

    def _fit(self, x: torch.Tensor):
        self.fcts_ = [
            CheapRandomFunction(self.context, in_features=self.in_features, out_features=self.out_features)
            for _ in range(2)
        ]

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcts_[0](x) * self.fcts_[1](x)

# ==================== _multi_function ====================

class RandomMultiFunction(RandomTransformer):
    """Random function of several input nodes."""

    def __init__(self, context: Context, out_features: int):
        super().__init__(context)
        self.out_features = out_features

    def _fit(self, x: List[torch.Tensor]):
        if self.config.multi_fct_types == "default":
            func_type = self.sampler.choice("multi_function_func_type",
                                            [RandomConcatMultiFunction, RandomAggregationMultiFunction])
        elif self.config.multi_fct_types == "concat":
            func_type = RandomConcatMultiFunction
        elif self.config.multi_fct_types == "agg":
            func_type = RandomAggregationMultiFunction
        else:
            raise ValueError(f"Unknown option {self.config.multi_fct_types=}")
        self.func_ = func_type(self.context, self.out_features)

    def _transform(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.func_(x)

class RandomConcatMultiFunction(RandomMultiFunction):
    def _fit(self, x: List[torch.Tensor]):
        in_features = sum([z.shape[-1] for z in x])
        self.func_ = RandomFunction(self.context, in_features, self.out_features)

    def _transform(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.func_(torch.cat(x, dim=-1))

class RandomAggregationMultiFunction(RandomMultiFunction):
    def _fit(self, x: List[torch.Tensor]):
        self.op_type_ = self.sampler.choice("aggregation_op_type", ["sum", "product", "max", "logsumexp"])
        self.funcs_ = [RandomFunction(self.context, tens.shape[-1], self.out_features) for tens in x]

    def _transform(self, x: List[torch.Tensor]) -> torch.Tensor:
        out_cat = torch.stack([f(tens) for f, tens in zip(self.funcs_, x)], dim=0)
        if self.op_type_ == "sum":
            return torch.sum(out_cat, dim=0)
        elif self.op_type_ == "product":
            return torch.prod(out_cat, dim=0)
        elif self.op_type_ == "max":
            return torch.max(out_cat, dim=0).values
        elif self.op_type_ == "logsumexp":
            return torch.logsumexp(out_cat, dim=0)
        else:
            raise ValueError(f'Unknown op_type "{self.op_type_}"')

# ==================== _points ====================

class RandomPoints(PriorComponent):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        base_points = self.sampler.choice(
            "random_base_points",
            [RandomUniformPoints, RandomCirclePoints, RandomGaussianPoints, RandomCovariancePoints])
        return RandomFunction(self.context, n, n)(base_points(self.context).sample(n_batch, n))

class RandomCovariancePoints(RandomPoints):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        matrix = torch.randn(n, n) * RandomWeights(self.context).sample(1, n)
        base_points = self.sampler.choice("random_covariance_base_points", [RandomUniformPoints, RandomGaussianPoints])
        return base_points(self.context).sample(n_batch, n) @ matrix.t()

class RandomGaussianPoints(RandomPoints):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        return torch.randn(n_batch, n, device=self.device)

class RandomUniformPoints(RandomPoints):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        return -1 + 2 * torch.rand(n_batch, n, device=self.device)

class RandomCirclePoints(RandomPoints):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        # angular part from Gaussian points; radial inverse CDF r = u^{1/n}
        x = torch.randn(n_batch, n)
        x /= torch.linalg.vector_norm(x, ord=2, keepdim=True, dim=-1)
        u = torch.rand(n_batch, 1)
        r = u ** (1 / n)
        return r * x

class RandomGaussianMixturePoints(RandomPoints):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        n_centers = self.sampler.randint("random_gmp_n_centers", 1, 16, use_log=True)
        weights = RandomWeights(self.context).sample(1, n_centers).squeeze(0)
        center_idxs = torch.multinomial(weights, num_samples=n_batch, replacement=True)
        centers = torch.randn(n_centers, n)
        matrices = torch.randn(n_centers, n, n) * RandomWeights(self.context).sample(n_centers, n)[:, None, :]
        # random factors with random mean and random std
        matrices = matrices * torch.exp(torch.randn(1) + torch.randn(1) * torch.randn(n_centers, 1, 1))
        return centers[center_idxs] + (matrices[center_idxs] @ torch.randn(n_batch, n, 1)).squeeze(-1)

# ==================== _converter ====================

class Converter(RandomTransformer):
    """Extracts a dataset column from part of a node matrix, and may modify that part of the node matrix.
    ``__call__(x)`` returns (values propagated in the graph, values used in the dataset)."""

    def get_n_features(self) -> int:
        raise NotImplementedError()

class NumericalConverter(Converter):
    def __init__(self, context: Context, disallow_warping: bool = False):
        super().__init__(context)
        self.disallow_warping = disallow_warping

    def get_n_features(self):
        return 1

    def _fit(self, x: torch.Tensor):
        modes = ["id"]
        if self.config.allow_kumaraswamy_warping and not self.disallow_warping:
            modes.append("kumar")
        if self.config.allow_act_warping and not self.disallow_warping:
            modes.append("act")
        self.mode_ = self.sampler.choice("num_conv_mode", modes)
        self.tfm_ = RandomActivation(self.context) if self.mode_ == "act" else KumaraswamyWarping(self.context)

    def _transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode_ == "id":
            return x, x
        elif self.mode_ in ["kumar", "act"]:
            if self.config.use_corrected_num_converters:
                return x, self.tfm_(x)
            else:
                # the wrong version, but the one used in the TabICLv2 experiments
                return self.tfm_(x), x
        else:
            raise NotImplementedError()

class CategoricalNeighborDiscretizer(Converter):
    def __init__(self, context: Context, n_values: int):
        super().__init__(context)
        self.n_values = n_values

    def _fit(self, x: torch.Tensor):
        perm = torch.randperm(len(x), device=x.device)
        self.centers_ = x[perm[: self.n_values]]
        self.p_ = self.sampler.numerical("cat_disc_p", 0.5, 4.0, use_log=True)

    def _transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = torch.cdist(x, self.centers_, p=self.p_)
        closest_idx = dists.argmin(dim=-1)
        return self.centers_[closest_idx], closest_idx[:, None]

class CategoricalSoftmaxDiscretizer(Converter):
    def __init__(self, context: Context, n_values: int):
        super().__init__(context)
        self.n_values = n_values

    def _fit(self, x: torch.Tensor):
        self.embeddings_ = RandomPoints(self.context).sample(self.n_values, self.n_values)
        self.standardize_ = Standardize(self.context)
        self.bias_ = torch.log(RandomWeights(self.context).sample(1, x.shape[1]) + 1e-4)
        self.factor_ = self.sampler.numerical("softmax_disc_factor", 0.1, 10, use_log=True)

    def _transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.factor_ * self.standardize_(x) + self.bias_
        x[~torch.isfinite(x)] = 0.0
        idxs = torch.multinomial(torch.softmax(x, dim=-1), num_samples=1, replacement=True).squeeze(-1)
        return self.embeddings_[idxs], idxs[:, None]

class CategoricalConverter(Converter):
    def __init__(self, context: Context, n_values: int):
        super().__init__(context)
        self.n_values = n_values

        all_cat_modes = ["neighbor_id", "neighbor_disc", "neighbor_func", "neighbor_int", "softmax_id", "softmax_disc",
                         "softmax_int"]
        if self.config.cat_modes == "default":
            cat_modes = all_cat_modes
        elif self.config.cat_modes in all_cat_modes:
            cat_modes = [self.config.cat_modes]
        elif self.config.cat_modes == "neighbor":
            cat_modes = [cm for cm in all_cat_modes if cm.startswith("neighbor")]
        elif self.config.cat_modes == "softmax":
            cat_modes = [cm for cm in all_cat_modes if cm.startswith("softmax")]
        else:
            raise ValueError(f"Unknown value {self.config.cat_modes=}")

        self.mode = self.sampler.choice("cat_mode", cat_modes)
        disc_type = CategoricalSoftmaxDiscretizer if self.mode.startswith("softmax") else CategoricalNeighborDiscretizer
        self.disc = disc_type(self.context, n_values=self.n_values)
        if self.mode.startswith("softmax"):
            self.n_features = n_values
        else:
            self.n_features = (n_values if self.sampler.boolean("cat_use_all_features")
                               else self.sampler.randint("cat_n_features", 1, n_values, use_log=True, mode="local"))
        self.post_func = (CheapRandomFunction(self.context, self.n_features, self.n_features)
                          if self.mode == "neighbor_func" else None)

    def get_n_features(self) -> int:
        return self.n_features

    def _transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_disc, x_idxs = self.disc(x)

        if self.mode in ["neighbor_id", "softmax_id"]:
            return x, x_idxs
        elif self.mode in ["neighbor_disc", "softmax_disc"]:
            return x_disc, x_idxs
        elif self.mode == "neighbor_func":
            return self.post_func(x_disc), x_idxs
        elif self.mode in ["neighbor_int", "softmax_int"]:
            return x_idxs.float(), x_idxs  # one column, broadcast into the node matrix
        else:
            raise ValueError(f'Unknown categorical mode "{self.mode}"')

# ==================== _node_function, _graph_function, _graph ====================

class RandomNodeFunction(RandomTransformer):
    """Computes the node matrix from the parents' matrices and extracts the node's dataset columns."""

    def __init__(self, context: Context, feature_specs: Dict[str, FeatureSpec]):
        super().__init__(context)
        self.feature_specs = feature_specs

    def _fit(self, x: List[torch.Tensor], n_samples: int):
        self.converters_ = dict()
        for feature_name, feature_spec in self.feature_specs.items():
            cat_size = feature_spec.cat_size
            self.converters_[feature_name] = (
                CategoricalConverter(self.context, n_values=cat_size) if cat_size > 0
                else NumericalConverter(self.context,
                                        disallow_warping=self.config.disallow_y_warping and feature_spec.group == "y"))

        self.n_features_ = self.sampler.randint("node_n_latent_features", 1, 32, use_log=True)
        if len(self.converters_) > 0:
            self.n_features_ += sum(converter.get_n_features() for converter in self.converters_.values())
        self.random_points_ = RandomPoints(self.context)
        self.random_func_ = RandomMultiFunction(self.context, out_features=self.n_features_)

        self.std_ = Standardize(self.context)
        self.l2_norm_ = L2Normalize(self.context)
        self.random_weights_ = RandomWeights(self.context)

    def _transform(self, x: List[torch.Tensor], n_samples: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(x) == 0:
            x = self.random_points_.sample(n_samples, self.n_features_)
        else:
            x = self.random_func_(x)

        if self.config.use_node_feature_importance:
            x = self.std_(x)
            weights = self.random_weights_.sample(1, x.shape[1]).squeeze(0)
            weights = weights / weights.square().mean().sqrt()
            x = x * weights[None, :]
        if self.config.use_node_l2_norm:
            x = self.l2_norm_(x)

        out_features = dict()
        start_idx = 0
        for name, converter in self.converters_.items():
            end_idx = start_idx + converter.get_n_features()
            assert end_idx <= self.n_features_
            tens, feat = converter(x[:, start_idx:end_idx])
            x[:, start_idx:end_idx] = tens
            out_features[name] = feat
            start_idx = end_idx

        if self.config.add_gaussian_noise:
            noise_level = self.sampler.numerical("gaussian_noise_sigma", low=1e-5, high=1e0, use_log=True)
            x = x + noise_level * torch.randn_like(x)

        if self.config.use_node_importance:
            x = x * self.sampler.numerical("node_importance", low=0.1, high=10.0, use_log=True)

        return x, out_features

class RandomGraphFunction(RandomTransformer):
    """Samples a dataset by propagating data through a DAG."""

    def __init__(self, context: Context, dag: List[List[int]], node_feature_specs: List[Dict[str, FeatureSpec]]):
        super().__init__(context=context)
        self.dag = dag
        self.node_feature_specs = node_feature_specs

    def _fit(self, n_samples: int):
        self.nodes_ = [RandomNodeFunction(self.context, feature_specs=specs) for specs in self.node_feature_specs]
        # prune nodes whose values are not needed
        self.should_compute_ = [False for _ in range(len(self.nodes_))]
        for node_idx in reversed(range(len(self.nodes_))):
            if len(self.node_feature_specs[node_idx]) >= 1:
                self.should_compute_[node_idx] = True
            if self.should_compute_[node_idx]:
                for parent in self.dag[node_idx]:
                    self.should_compute_[parent] = True

    def _transform(self, n_samples: int) -> Dict[str, torch.Tensor]:
        node_values = [None for _ in range(len(self.node_feature_specs))]
        features = dict()
        for node_idx in range(len(self.node_feature_specs)):
            if self.should_compute_[node_idx]:
                node_values[node_idx], out_features = self.nodes_[node_idx](
                    [node_values[parent] for parent in self.dag[node_idx]], n_samples
                )
                features = features | out_features
        return features

class RandomDAG(PriorComponent):
    def sample(self, n_nodes: int) -> List[List[int]]:
        """Returns parents[i], the parent node indices of each node i."""
        return RandomCauchyDAG(self.context).sample(n_nodes)

class RandomCauchyDAG(RandomDAG):
    """Random DAG whose connectivities are modelled by Cauchy random variables."""

    def sample(self, n_nodes: int) -> List[List[int]]:
        offset = np.random.standard_cauchy() + self.config.cauchy_dag_offset
        output_importances = torch.tensor(np.random.standard_cauchy(n_nodes))
        input_importances = torch.tensor(np.random.standard_cauchy(n_nodes))
        logits = offset + output_importances[None, :] + input_importances[:, None]
        adjacency_matrix = torch.rand_like(logits) <= torch.sigmoid(logits)
        return [[i_in for i_in in range(i_out) if adjacency_matrix[i_in, i_out]] for i_out in range(n_nodes)]

# ==================== _dataset (graph_lib) ====================

# upstream iterates list(set(groups)), whose order depends on PYTHONHASHSEED; this is the order under 0
FEATURE_GROUP_ORDER = ("y", "x")

def check_x_y_ancestors_overlap(graph: List[List[int]], node_feature_specs: List[Dict[str, FeatureSpec]]) -> bool:
    """False if x and y are independent according to the graph (no common ancestor), otherwise True."""
    ancestors_arrs = {key: np.zeros(len(graph), dtype=np.bool_) for key in ["x", "y"]}
    for node_idx in reversed(range(len(graph))):
        for feature_spec in node_feature_specs[node_idx].values():
            if feature_spec.group in ["x", "y"]:
                ancestors_arrs[feature_spec.group][node_idx] = True
        for group in ["x", "y"]:
            if ancestors_arrs[group][node_idx]:
                for parent_idx in graph[node_idx]:
                    ancestors_arrs[group][parent_idx] = True
    return np.any(np.logical_and(ancestors_arrs["x"], ancestors_arrs["y"]))

class RandomDataset(PriorComponent):
    def sample(self, data_prop: DatasetProperties) -> Dataset:
        while True:
            # ----- create the computation graph -----
            n_nodes = self.sampler.randint("n_nodes", self.config.min_n_nodes, self.config.max_n_nodes + 1,
                                           use_log=True)
            graph = RandomDAG(self.context).sample(n_nodes)

            node_feature_specs = [dict() for _ in range(n_nodes)]

            groups = {spec.group for spec in data_prop.feature_specs.values()}
            assert groups <= set(FEATURE_GROUP_ORDER), groups
            feature_groups = [g for g in FEATURE_GROUP_ORDER if g in groups]
            for feature_group in feature_groups:
                feature_specs = {k: v for k, v in data_prop.feature_specs.items() if v.group == feature_group}
                if self.config.subsample_feature_nodes:
                    n_feature_nodes = self.sampler.randint("n_feature_nodes", 1, n_nodes + 1)
                    feature_nodes = np.random.permutation(n_nodes)[:n_feature_nodes]
                else:
                    feature_nodes = np.arange(n_nodes)
                feature_node_idxs = np.random.choice(feature_nodes, replace=True, size=len(feature_specs))

                for idx, (feature_name, feature_spec) in enumerate(feature_specs.items()):
                    node_feature_specs[feature_node_idxs[idx]][feature_name] = feature_spec

            if (not self.config.filter_unpredictable_graphs) or check_x_y_ancestors_overlap(graph, node_feature_specs):
                break  # otherwise y and x are independent in the graph: try again

        graph_func = RandomGraphFunction(self.context, dag=graph, node_feature_specs=node_feature_specs)

        # ----- evaluate the computation graph -----
        n_samples = data_prop.n_train + data_prop.n_test
        if self.config.ensure_iid:
            graph_func(n_samples)  # fit the graph function on separate data
        tensors = graph_func(n_samples)
        return Dataset(tensors=tensors, feature_specs=data_prop.feature_specs, graph=graph, n_train=data_prop.n_train)

# ==================== _reg2cls: scaling helpers ====================

def torch_nanstd(input, dim=None, keepdim=False, ddof=0, *, dtype=None) -> Tensor:
    """Standard deviation ignoring NaNs, computed with NumPy."""
    np_input = input.cpu().numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(np_input, axis=dim, dtype=dtype, keepdims=keepdim, ddof=ddof)
    return torch.from_numpy(std).to(dtype=torch.float, device=input.device)

def standard_scaling(input: Tensor, clip_value: float = 100) -> Tensor:
    """Standardizes the columns of (T, H), ignoring NaNs, and clips to [-clip_value, clip_value]."""
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    scaled_input = (input - mean) / std
    return torch.clip(scaled_input, min=-clip_value, max=clip_value)

def outlier_removing(input: Tensor, threshold: float = 4.0) -> Tensor:
    """Clamps values beyond ``threshold`` robust standard deviations, per column of (T, H)."""
    # first pass: initial statistics
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)

    # second pass: statistics of the non-outliers
    masked_input = torch.where(mask, input, torch.nan)
    masked_mean = torch.nanmean(masked_input, dim=0)
    masked_std = torch_nanstd(masked_input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)

    # columns with <= 1 valid value after masking
    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)

    cut_off = masked_std * threshold
    lower, upper = masked_mean - cut_off, masked_mean + cut_off
    lower = torch.nan_to_num(lower, nan=-torch.inf)
    upper = torch.nan_to_num(upper, nan=torch.inf)
    return input.clamp(min=lower, max=upper)

# ==================== _graph_scm ====================

class GraphSCM:
    """Generates one dataset (X (seq_len, max_features), y (seq_len,)) from a random causal graph."""

    def __init__(self, seq_len: int = 1024, num_features: int = 100, max_features: int = 100,
                 num_classes: int = 2, permute_features: bool = True, permute_labels: bool = True,
                 config: Optional[PriorConfig] = None, **kwargs):
        self.seq_len = seq_len
        self.num_features = num_features
        self.max_features = max_features
        self.num_classes = num_classes
        self.permute_features = permute_features
        self.permute_labels = permute_labels
        self.config = config

    def __call__(self) -> Tuple[Tensor, Tensor]:
        context = Context(config=self.config)
        cat_sizes = {"x": sample_categorical_sizes(self.num_features, context, max_cat_size=200),
                     "y": [self.num_classes]}
        properties = DatasetProperties(n_train=self.seq_len, n_test=0, cat_sizes=cat_sizes)
        ds = RandomDataset(context).sample(properties)
        data = ds.get_concat_tensors()

        X_cat = data.get("x_cat", None)
        X_num = data.get("x_num", None)

        parts = [t for t in (X_cat, X_num) if t is not None]
        if not parts:
            raise ValueError("No features found in dataset")
        X = torch.cat(parts, dim=-1) if len(parts) == 2 else parts[0]

        X = outlier_removing(X.float(), threshold=4)
        X = standard_scaling(X)

        if self.permute_features:
            feat_perm = torch.randperm(self.num_features)
            X = X[..., feat_perm]

        if self.num_features < self.max_features:
            X = F.pad(X, (0, self.max_features - self.num_features), mode="constant", value=0.0)

        y = data["y_cat"].view(-1)
        if self.permute_labels:
            class_perm = torch.randperm(self.num_classes)
            y = class_perm[y.long()]

        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X = torch.zeros_like(X)
            y = torch.full_like(y, -100.0)

        return X, y

# ==================== _dataset (prior): Prior statics, GraphPrior, should_filter, PriorDataset ====================

def validate_train_size_range(min_train_size: Union[int, float], max_train_size: Union[int, float]) -> None:
    if not isinstance(min_train_size, (int, float)) or not isinstance(max_train_size, (int, float)):
        raise TypeError("Training sizes must be int or float")
    if isinstance(min_train_size, int) and isinstance(max_train_size, int):
        assert 0 < min_train_size < max_train_size, "0 < min_train_size < max_train_size"
    elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
        # min == max allowed: a fixed train fraction (stages 2 and 3)
        assert 0 < min_train_size <= max_train_size < 1, "0 < min_train_size <= max_train_size < 1"
    else:
        raise ValueError("Both training sizes must be of the same type (int or float)")

def sample_seq_len(min_seq_len: Optional[int], max_seq_len: int, log: bool = False) -> int:
    if min_seq_len is None:
        return max_seq_len
    if log:
        return int(loguniform.rvs(min_seq_len, max_seq_len))
    return np.random.randint(min_seq_len, max_seq_len)

def sample_train_size(min_train_size: Union[int, float], max_train_size: Union[int, float], seq_len: int) -> int:
    """An absolute position (int bounds) or a fraction of seq_len (float bounds)."""
    if isinstance(min_train_size, int) and isinstance(max_train_size, int):
        train_size = np.random.randint(min_train_size, max_train_size)
    elif isinstance(min_train_size, float):
        train_size = np.random.uniform(min_train_size, max_train_size)
        train_size = int(seq_len * train_size)
    else:
        raise ValueError("Invalid training size range.")
    return train_size

def adjust_max_features(seq_len: int, max_features: int) -> int:
    """Longer sequences get fewer features."""
    if seq_len <= 10240:
        return max_features
    for bound, cap in ((20000, 80), (30000, 60), (40000, 40), (50000, 30), (60000, 20), (65000, 15)):
        if seq_len <= bound:
            return min(cap, max_features)
    return 10

def delete_unique_features(X: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
    """Removes constant features of each (T, H) dataset in X, zero-padding back to H."""

    def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor]:
        num_features = xi.shape[-1]
        xi = xi[:, :di]
        unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
        di_new = sum(unique_mask)
        xi_new = F.pad(xi[:, unique_mask], pad=(0, num_features - di_new), mode="constant", value=0)
        return xi_new, torch.tensor(di_new, device=xi.device)

    filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
    X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]
    return X_new, d_new

def cls_sanity_check(X: Tensor, y: Tensor, train_size: int, n_attempts: int = 10, min_classes: int = 2) -> bool:
    """True if train and test contain the same >= min_classes classes, trying up to n_attempts random
    permutations (applied in place) to fix an invalid split."""

    def is_valid_split(yi: Tensor) -> bool:
        if train_size <= 0 or train_size >= yi.shape[0]:
            return False
        unique_tr = torch.unique(yi[:train_size])
        unique_te = torch.unique(yi[train_size:])
        return set(unique_tr.tolist()) == set(unique_te.tolist()) and len(unique_tr) >= min_classes

    for i, (xi, yi) in enumerate(zip(X, y)):
        if is_valid_split(yi):
            continue
        succeeded = False
        for _ in range(n_attempts):
            perm = torch.randperm(yi.shape[0])
            yi_perm = yi[perm]
            xi_perm = xi[perm]
            if is_valid_split(yi_perm):
                X[i], y[i] = xi_perm, yi_perm
                succeeded = True
                break
        if not succeeded:
            return False
    return True

def should_filter(X: torch.Tensor, y: torch.Tensor, config: PriorConfig, n_estimators: int = 25,
                  n_boot: int = 200) -> bool:
    """Filters datasets where an out-of-bag ExtraTrees is not significantly better than the mean predictor
    (and, optionally, trivially predictable ones). Uses fixed seeds only."""
    if not (config.remove_trivial_datasets or config.filter_unpredictable_datasets):
        return False

    y_int = y.to(torch.long)
    C = int(y_int.max().item() + 1)
    Y = F.one_hot(y_int, num_classes=C).to(torch.float32)  # (n, C)
    if C == 2:
        Y = Y[:, :1]  # drop one dimension to make it faster

    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    # random_state=0 does not give OOB scores for all samples for some 133 <= n <= 257
    et = ExtraTreesRegressor(n_estimators=n_estimators, bootstrap=True, oob_score=True, n_jobs=1, random_state=1,
                             max_depth=6).fit(X_np, np.squeeze(Y_np, axis=-1) if Y_np.shape[-1] == 1 else Y_np)

    Yhat = et.oob_prediction_
    if len(Yhat.shape) == 1:
        Yhat = Yhat[:, None]
    mask = ~np.isnan(Yhat).any(axis=1)  # valid OOB rows
    Yv, Yhatv = Y_np[mask], Yhat[mask]

    baseline = Y_np.mean(axis=0, keepdims=True)
    imp = ((Yv - baseline) ** 2).sum(axis=1) - ((Yv - Yhatv) ** 2).sum(axis=1)

    baseline_mse = ((Yv - baseline) ** 2).sum(axis=1).mean()
    extratrees_mse = ((Yv - Yhatv) ** 2).sum(axis=1).mean()

    # vectorized bootstrap (no retraining)
    rng = np.random.default_rng(0)
    n = imp.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    pval = float(np.mean(imp[idx].mean(axis=1) <= 0.0))

    trivial = np.sqrt(extratrees_mse) <= config.trivial_dataset_threshold * np.sqrt(baseline_mse)
    if config.remove_trivial_datasets and trivial:
        return True
    if config.filter_unpredictable_datasets and pval >= 0.05:
        return True
    return False

class PriorDataset(IterableDataset):
    """Infinite stream of graph_scm batches (upstream ``PriorDataset(prior_type="graph_scm", n_jobs=1)``).

    Datasets are generated in groups of ``batch_size_per_gp`` that share the number of features and, with
    ``seq_len_per_gp=True``, the sequence length and train size (then X and y are nested tensors)."""

    def __init__(self, batch_size: int, batch_size_per_gp: int, min_features: int, max_features: int,
                 max_classes: int, min_seq_len: Optional[int], max_seq_len: int, log_seq_len: bool,
                 min_train_size: Union[int, float], max_train_size: Union[int, float],
                 seq_len_per_gp: bool = True, config: PriorConfig = STAGE_CONFIG):
        super().__init__()
        assert min_features <= max_features, "Invalid feature range"
        validate_train_size_range(min_train_size, max_train_size)
        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.seq_len_per_gp = seq_len_per_gp
        self.config = config

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """Retries until a dataset has >= 1 non-constant feature, a valid class split, and
        passes ``should_filter``. Returns X (seq_len, max_features), y (seq_len,), d (active features)."""
        while True:
            X, y = GraphSCM(**params)()
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], dtype=torch.long)

            X, d = delete_unique_features(X, d)
            if not (d > 0).all():
                continue
            if not cls_sanity_check(X, y, params["train_size"]):
                continue
            if should_filter(X[0], y[0], self.config):
                continue
            return X.squeeze(0), y.squeeze(0), d.squeeze(0)

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Returns (X, y, d, seq_lens, train_sizes); X and y are nested tensors when seq_len_per_gp=True,
        else (B, T, H) and (B, T)."""
        batch_size = batch_size or self.batch_size
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        param_list = []
        if not self.seq_len_per_gp:
            global_seq_len = sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)
            global_train_size = sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)

        for gp_idx in range(num_gps):
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if self.seq_len_per_gp:
                gp_seq_len = sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)
                gp_train_size = sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len)
                gp_max_features = adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len, gp_train_size, gp_max_features = global_seq_len, global_train_size, self.max_features

            # the group is a single sub-group, which shares the number of features
            gp_num_features = round(np.random.uniform(self.min_features, gp_max_features))
            for _ in range(actual_gp_size):
                num_classes = np.random.randint(2, self.max_classes + 1)
                param_list.append(dict(config=self.config, seq_len=gp_seq_len,
                                       train_size=gp_train_size, max_features=gp_max_features,
                                       num_features=gp_num_features, num_classes=num_classes))

        with threadpoolctl.threadpool_limits(limits=1):
            n_threads_old = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                results = [self.generate_dataset(params) for params in param_list]
            finally:
                torch.set_num_threads(n_threads_old)

        X_list, y_list, d_list = zip(*results)
        if self.seq_len_per_gp:
            X = torch.nested.nested_tensor(list(X_list))
            y = torch.nested.nested_tensor(list(y_list))
        else:
            X = torch.stack(X_list)  # (B, T, H)
            y = torch.stack(y_list)  # (B, T)

        d = torch.stack(d_list)  # active features after removing constant ones
        seq_lens = torch.tensor([p["seq_len"] for p in param_list], dtype=torch.long)
        train_sizes = torch.tensor([p["train_size"] for p in param_list], dtype=torch.long)
        return X, y, d, seq_lens, train_sizes

    def __iter__(self):
        while True:
            yield self.get_batch()
