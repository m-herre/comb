"""Comb as a TabArena / AutoGluon model (needs a TabArena install; not an official TabArena entry).

Mirrors TabArena's own TabICLv2 and LightPFN models (seed passing, device handling, ensemble args), with the
released ``CombClassifier`` at its defaults; a ``cell_m`` hyperparameter sets the cell budget. Weights come from
``COMB_CHECKPOINT`` or the Hub.
"""
from __future__ import annotations

from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel


class CombModel(AbstractTorchModel):
    ag_key = "TA-COMB"
    ag_name = "TA-Comb"
    ag_priority = 65
    seed_name = "random_state"

    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 0
    _supported_problem_types = ["binary", "multiclass"]

    @classmethod
    def warmup(cls, *, num_gpus: float | None = None, **kwargs) -> None:
        """Weights and kernels (fp32 and fp16 paths) on fixed synthetic tables, never the task's data."""
        if num_gpus is not None and num_gpus <= 0:
            return
        import numpy as np
        from comb import CombClassifier

        rng = np.random.default_rng(0)
        for n in (200, 4000):
            X = rng.normal(size=(n, 20))
            CombClassifier(device="cuda").fit(X, (X[:, 0] > 0).astype(int)).predict_proba(X[:100])

    def _fit(self, X, y, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        from comb import CombClassifier

        params = dict(self._get_model_params())
        lo, hi = params.pop("cell_m_lo", None), params.pop("cell_m_hi", None)
        c = params.pop("cell_m_sqrt", None)
        if lo is not None:  # size rules: clip(N, lo, hi) (= fixed hi, since N <= M is exact) or clip(c sqrt(N), lo, hi)
            params["cell_m"] = int(min(max(c * len(X) ** 0.5 if c else len(X), lo), hi))
        self.model = CombClassifier(**params, device="cuda" if num_gpus else "cpu")
        self.model.fit(self.preprocess(X, y=y), y)

    @classmethod
    def _estimate_memory_usage_static(cls, *, X, hyperparameters: dict | None = None, **kwargs) -> int:
        return 3 * get_approximate_df_mem_usage(X).sum() + 1e9  # as TabArena's TabICLv2Model

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        args = super()._get_default_ag_args_ensemble(**kwargs)
        args.update({"fold_fitting_strategy": "sequential_local", "refit_folds": True})
        return args

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _class_tags(cls) -> dict:
        # CombClassifier pickles without its network (re-attached from the weights file on load), so ``save`` need not
        # move every fold model to the CPU and back
        return {**super()._class_tags(), "pickles_pretrained_weights": False}

    def get_device(self) -> str:
        return self.model.device_.type

    def _set_device(self, device: str):
        self.model.to(self.to_torch_device(device))


def gen(**config):
    from tabarena.utils.config_utils import ConfigGenerator

    return ConfigGenerator(model_cls=CombModel, manual_configs=[config], search_space={})
