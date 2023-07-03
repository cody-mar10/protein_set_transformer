from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import optuna

from pst.utils.cli import _KWARG_TYPE

from .config import load_config


class HyperparameterRegistryMixin:
    def __init__(
        self,
        trial: optuna.Trial,
        configfile: Path,
    ):
        self._hparams: _KWARG_TYPE = dict()
        self._trial = trial
        self._config = load_config(configfile)
        self._extract_suggest_type()
        self._extract_maps()

    def _extract_suggest_type(self):
        suggestions: defaultdict[str, dict[str, str]] = defaultdict(dict)
        for hparam_type, hparams in self._config.items():
            for hparam, opts in hparams.items():
                suggestion = opts.pop("suggest")
                suggestions[hparam_type][hparam] = suggestion

        self._suggestions = suggestions

    def _extract_maps(self):
        category_maps: dict[str, dict[str, dict[str, Any]]] = dict()
        for hparam_type, hparams in self._config.items():
            for hparam, opts in hparams.items():
                mapvalues = opts.pop("map", None)
                if mapvalues is None:
                    continue
                category_maps[hparam] = mapvalues

        self._category_maps = category_maps

    def _map_categories(self):
        for hparam, categories in self._category_maps.items():
            trialed_value = self._hparams.pop(hparam)
            mapped_values = categories[trialed_value]
            self._hparams.update(**mapped_values)

    def register_hparams(self):
        for hparam_type, hparams in self._config.items():
            for hparam, opts in hparams.items():
                suggest = self._suggestions[hparam_type][hparam]
                method = f"suggest_{suggest}"
                self._hparams[hparam] = getattr(self._trial, method)(
                    name=hparam, **opts
                )

        self._map_categories()
