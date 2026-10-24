# """GLINT core — Stages 0-8 causal discovery algorithm helpers.

# Dataset-agnostic: every knob below (TIERS, KNOWN_EDGES, B, M_DOWNSAMPLE,
# F_KEEP, MIN_CELL, ...) is a placeholder default, unconditionally
# rebound by bind_dataset() (inside causal_graph(), below) before any of it is
# actually used. The module-level values just need to exist so the functions
# below have something to import/reference at definition time; they carry no
# information about any particular dataset (this engine was originally
# prototyped against the Sachs et al. protein-signaling benchmark, and the
# placeholders below used to be that dataset's actual variable names and known
# edges — that data has been removed since it never described this pipeline's
# KPI data and was live only by coincidence of cell-execution order).
# """
# from __future__ import annotations

# import json
# import time
# from dataclasses import dataclass, field, asdict
# from itertools import combinations
# from pathlib import Path
# from typing import Any

# import networkx as nx
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.metrics import average_precision_score
# from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# # ---------------------------------------------------------------------------
# # Knobs — all rebound per-run by bind_dataset(); see its docstring.
# # ---------------------------------------------------------------------------
# B = 8
# M_DOWNSAMPLE = 20
# F_KEEP = 0.70
# FCI_ALPHA = 0.01
# EV_MAX = 1.0
# MIN_CELL = 300
# SEED = 0
# CV_FOLDS = 5
# MIN_POSITIVES = 0
# # TARGET and TARGET_BINARY are set as module globals by bind_dataset() so that
# # _outcome_n_pos() (which reads globals()) can find them from any call site.
# TARGET: str = ""
# TARGET_BINARY: bool = False

# # variable -> stage tier (1 = context/root ... 4 = outcome/sink). Populated by
# # bind_dataset() from this run's actual TIERS before any Stage 0-8 function
# # below reads it.
# TIERS: dict[str, int] = {}

# # (parent, child) pairs to force into the graph regardless of what discovery
# # finds. Populated by bind_dataset(); empty by default.
# KNOWN_EDGES: list[tuple[str, str]] = []

# # Reverse tier jumps and sink -> root edges. Derived from TIERS, so it's
# # recomputed by bind_dataset() alongside it; empty here since TIERS is empty.
# FORBIDDEN_EDGES: list[tuple[str, str]] = []


# @dataclass
# class EnvWorld:
#     name: str
#     tag: str  # natural | downsampled | near-experiment
#     data: pd.DataFrame


# @dataclass
# class CandidateRecord:
#     variable: str
#     parents: tuple[str, ...]
#     cv_score: float
#     prior: float
#     confounder_flags: list[str] = field(default_factory=list)


# @dataclass
# class EdgeRecord:
#     source: str
#     target: str
#     confidence: float = 0.0
#     strength: float = 0.0
#     trust: float = 0.0
#     direction: str = "→"
#     confounder: bool = False
#     audit: str = ""


# # ---------------------------------------------------------------------------
# # Stage 0
# # ---------------------------------------------------------------------------
# def build_schema(columns: list[str]) -> pd.DataFrame:
#     rows = []
#     for c in columns:
#         rows.append(
#             {
#                 "variable": c,
#                 "kind": "continuous",
#                 "time_status": "current",
#                 "role": "context" if TIERS[c] == 1 else ("outcome" if TIERS[c] == 4 else "mediator"),
#                 "tier": TIERS[c],
#                 "deterministic": False,
#                 "in_discovery": True,
#             }
#         )
#     return pd.DataFrame(rows)


# def basis_vars() -> list[str]:
#     return [v for v, t in TIERS.items() if t == 1]


# def is_forbidden(src: str, tgt: str) -> bool:
#     if TIERS[src] > TIERS[tgt]:
#         return True
#     return (src, tgt) in FORBIDDEN_EDGES


# # ---------------------------------------------------------------------------
# # Stage 1 — environments
# # ---------------------------------------------------------------------------
# def _bin_series(s: pd.Series, n_bins: int) -> pd.Series:
#     try:
#         return pd.qcut(s, q=n_bins, labels=False, duplicates="drop")
#     except ValueError:
#         return pd.cut(s, bins=n_bins, labels=False, duplicates="drop")


# def _is_binary_y(y) -> bool:
#     arr = np.asarray(y, dtype=float)
#     u = np.unique(arr[~np.isnan(arr)])
#     return len(u) == 2


# def _n_pos_series(s: pd.Series) -> int:
#     y = pd.to_numeric(s, errors="coerce")
#     if 1 in y.dropna().unique() or 1.0 in y.dropna().unique():
#         return int((y == 1).sum())
#     if y.nunique(dropna=True) == 2:
#         return int((y == y.max()).sum())
#     return 0


# def _outcome_n_pos(df: pd.DataFrame, col: str | None = None) -> int:
#     # TARGET is set as a module global by bind_dataset(); use globals() to read it.
#     name = col or globals().get("TARGET") or ""
#     if not name or name not in df.columns:
#         return 0
#     return _n_pos_series(df[name])


# def build_natural_strata(df: pd.DataFrame, min_cell: int = MIN_CELL) -> list[EnvWorld]:
#     """Natural strata from the tier-1 (context) columns, binned into tertiles.

#     This replaces the original Sachs-diagram version (hardcoded "X1"/"X2"
#     protein roots), which only ever ran in the notebooks by coincidence of
#     cell-execution order — bind_dataset()'s dataset-specific redefinition
#     always overwrote it before it was ever called. Keeping only this version,
#     at module scope, is what the notebooks actually executed; it's also what
#     bind_dataset() below expects (it just sets TIERS/MIN_CELL/etc. as globals
#     rather than redefining this function).
#     """
#     roots = [v for v, t in TIERS.items() if t == 1 and v in df.columns]
#     all_cols = list(df.columns)
#     if not roots:
#         roots = all_cols[: min(2, len(all_cols))]
#     work = df.copy()
#     work["_b0"] = _bin_series(work[roots[0]], 3)
#     work["_b1"] = _bin_series(work[roots[1]], 3) if len(roots) > 1 else 0
#     extra = next((c for c in all_cols if c not in roots), roots[0])
#     work["_b2"] = _bin_series(work[extra], 3)
#     work["_cell"] = work["_b0"].astype(str) + "_" + work["_b1"].astype(str) + "_" + work["_b2"].astype(str)
#     envs: list[EnvWorld] = []
#     leftovers = []
#     for cell, g in work.groupby("_cell", sort=True):
#         sub = g.drop(columns=["_b0", "_b1", "_b2", "_cell"])
#         if len(sub) >= min_cell and (MIN_POSITIVES <= 0 or _outcome_n_pos(sub) >= MIN_POSITIVES):
#             envs.append(EnvWorld(name=f"natural_{cell}", tag="natural", data=sub.reset_index(drop=True)))
#         else:
#             leftovers.append(sub)
#     if leftovers:
#         merged = pd.concat(leftovers, ignore_index=True)
#         if len(merged) >= max(20, min_cell // 2) and (MIN_POSITIVES <= 0 or _outcome_n_pos(merged) >= MIN_POSITIVES):
#             envs.append(EnvWorld(name="natural_merged_small", tag="natural", data=merged))
#     return envs


# def _root_stratum_ids(df: pd.DataFrame) -> np.ndarray:
#     roots = basis_vars()
#     if not roots:
#         return np.zeros(len(df), dtype=int)
#     b0 = _bin_series(df[roots[0]], 4).fillna(0).astype(int).values
#     if len(roots) > 1:
#         b1 = _bin_series(df[roots[1]], 4).fillna(0).astype(int).values
#     else:
#         b1 = np.zeros(len(df), dtype=int)
#     return b0 * 10 + b1


# def build_downsampled_envs(
#     df: pd.DataFrame,
#     m: int = M_DOWNSAMPLE,
#     f_keep: float = F_KEEP,
#     seed: int = SEED,
#     max_env_n: int = 2500,
# ) -> list[EnvWorld]:
#     """Fixed-fraction downsampling toward m target root-mix menus (diagram Stage 1.2)."""
#     rng = np.random.default_rng(seed)
#     stratum = _root_stratum_ids(df)
#     uniq = np.unique(stratum)
#     n = len(df)
#     envs: list[EnvWorld] = []
#     target_weights = []
#     target_weights.append(np.ones(len(uniq)) / len(uniq))
#     for i in range(min(len(uniq), max(1, m // 3))):
#         w = np.full(len(uniq), 0.05 / max(len(uniq) - 1, 1))
#         w[i] = 0.95
#         target_weights.append(w)
#     while len(target_weights) < m:
#         w = rng.dirichlet(np.ones(len(uniq)) * 0.5)
#         target_weights.append(w)
#     target_weights = target_weights[:m]

#     for k, tw in enumerate(target_weights):
#         keep_idx = []
#         for j, s in enumerate(uniq):
#             members = np.where(stratum == s)[0]
#             n_keep = int(round(f_keep * tw[j] * n))
#             n_keep = min(n_keep, len(members))
#             if n_keep <= 0:
#                 continue
#             chosen = rng.choice(members, size=n_keep, replace=False)
#             keep_idx.append(chosen)
#         if not keep_idx:
#             continue
#         idx = np.concatenate(keep_idx)
#         if len(idx) > max_env_n:
#             idx = rng.choice(idx, size=max_env_n, replace=False)
#         if len(idx) < 50:
#             continue
#         sub = df.iloc[idx].reset_index(drop=True)
#         if MIN_POSITIVES > 0:
#             n_pos = _outcome_n_pos(sub)
#             if n_pos < MIN_POSITIVES:
#                 tgt = globals().get("TARGET")
#                 if tgt and tgt in df.columns:
#                     pos_idx = np.where(pd.to_numeric(df[tgt], errors="coerce").eq(1).to_numpy())[0]
#                     need = MIN_POSITIVES - n_pos
#                     extra = np.setdiff1d(pos_idx, idx, assume_unique=False)
#                     if len(extra) >= need:
#                         extra = rng.choice(extra, size=need, replace=False)
#                         idx = np.concatenate([idx, extra])
#                         sub = df.iloc[idx].reset_index(drop=True)
#                         n_pos = _outcome_n_pos(sub)
#             if n_pos < MIN_POSITIVES:
#                 continue
#         envs.append(EnvWorld(name=f"down_{k:02d}", tag="downsampled", data=sub))
#     return envs


# def build_env_pool(
#     df: pd.DataFrame,
#     m: int = M_DOWNSAMPLE,
#     f_keep: float = F_KEEP,
#     seed: int = SEED,
# ) -> list[EnvWorld]:
#     """Combine natural strata with fixed-fraction downsampled environments."""
#     return build_natural_strata(df) + build_downsampled_envs(
#         df, m=m, f_keep=f_keep, seed=seed
#     )




# # ---------------------------------------------------------------------------
# # Stage 2 — candidates + 1-SE prune + FCI screen
# # ---------------------------------------------------------------------------
# COLLINEAR_PAIRS: set[tuple[str, str]] = set()
# POST_TREATMENT_FORBIDDEN: set[tuple[str, str]] = set()


# def tier_candidates(target: str, columns: list[str]) -> list[str]:
#     t = TIERS[target]
#     cands = []
#     for v in columns:
#         if v == target:
#             continue
#         if is_forbidden(v, target):
#             continue
#         if (v, target) in COLLINEAR_PAIRS or (target, v) in COLLINEAR_PAIRS:
#             continue
#         if (v, target) in POST_TREATMENT_FORBIDDEN:
#             continue
#         # earlier tier always; same tier only if known or adjacent cascade
#         if TIERS[v] < t:
#             cands.append(v)
#         elif TIERS[v] == t and ((v, target) in KNOWN_EDGES or (target, v) in KNOWN_EDGES):
#             cands.append(v)
#         elif (v, target) in KNOWN_EDGES:
#             cands.append(v)
#     # always inject known parents
#     for a, b in KNOWN_EDGES:
#         if b == target and a not in cands and a in columns:
#             cands.append(a)
#     return cands



# def _scale_X(X: np.ndarray) -> np.ndarray:
#     if X.ndim == 1:
#         X = X.reshape(-1, 1)
#     if X.shape[1] == 0:
#         return X
#     return StandardScaler().fit_transform(X)


# # KFold/StratifiedKFold.split() only depends on (n_rows, n_splits, seed) for
# # plain KFold, or (y's class labels, n_splits, seed) for StratifiedKFold --
# # never on which candidate columns are in X. _cv_r2 is called many times with
# # the same y (all of prune_parents_1se's drop-the-worst-candidate iterations
# # for one target share it; every bootstrap replicate reuses the same row
# # count n) and a fresh KFold/StratifiedKFold object recomputes the same split
# # from scratch every single call (np.unique/argsort/sort -- real, measured
# # cost on real data). Caching the split and passing it to cross_val_score as
# # an explicit list of (train_idx, test_idx) pairs -- which sklearn accepts in
# # place of a splitter object, using it as-is instead of calling .split()
# # again -- gives byte-identical folds without recomputing them.
# _KFOLD_CACHE: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = {}
# _KFOLD_CACHE_MAXSIZE = 256


# def _cached_kfold_splits(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
#     key = ("kfold", n, n_splits, seed)
#     cached = _KFOLD_CACHE.get(key)
#     if cached is not None:
#         return cached
#     if len(_KFOLD_CACHE) >= _KFOLD_CACHE_MAXSIZE:
#         _KFOLD_CACHE.clear()
#     splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros((n, 1))))
#     _KFOLD_CACHE[key] = splits
#     return splits


# def _cached_stratified_kfold_splits(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
#     # Keyed on y's actual bytes (not id(y)) so two different arrays that
#     # happen to reuse the same memory address across calls can never collide.
#     key = ("skfold", y.tobytes(), n_splits, seed)
#     cached = _KFOLD_CACHE.get(key)
#     if cached is not None:
#         return cached
#     if len(_KFOLD_CACHE) >= _KFOLD_CACHE_MAXSIZE:
#         _KFOLD_CACHE.clear()
#     splits = list(
#         StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y)
#     )
#     _KFOLD_CACHE[key] = splits
#     return splits


# def _gram_ols_cv_r2(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray | None:
#     """Closed-form OLS-with-intercept CV R^2, replacing cross_val_score(LinearRegression)
#     for the continuous target path. Same KFold(shuffle=True, random_state=seed) splits and
#     the same r2 definition sklearn uses, computed directly with np.linalg.solve instead of
#     paying sklearn's per-fold estimator clone/validation/dispatch overhead (the dominant
#     cost for a handful of variables). Falls back to None (caller uses cross_val_score) on
#     any numerical issue, e.g. a rank-deficient design on a degenerate/collinear subset.
#     """
#     n, p = X.shape
#     try:
#         splits = _cached_kfold_splits(n, n_splits, seed)
#     except Exception:
#         return None
#     Xa = np.hstack([np.ones((n, 1)), X])
#     scores = []
#     jitter = 1e-9 * np.eye(p + 1)
#     for train_idx, test_idx in splits:
#         Xtr, ytr = Xa[train_idx], y[train_idx]
#         Xte, yte = Xa[test_idx], y[test_idx]
#         try:
#             beta = np.linalg.solve(Xtr.T @ Xtr + jitter, Xtr.T @ ytr)
#         except np.linalg.LinAlgError:
#             return None
#         pred = Xte @ beta
#         ss_res = float(np.sum((yte - pred) ** 2))
#         ss_tot = float(np.sum((yte - yte.mean()) ** 2))
#         scores.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)
#     return np.asarray(scores, dtype=float)


# def _cv_r2(X: np.ndarray, y: np.ndarray, folds: int = CV_FOLDS, seed: int = SEED) -> tuple[float, float]:
#     y = np.asarray(y, dtype=float)
#     if X.ndim == 1:
#         X = X.reshape(-1, 1)
#     binary = _is_binary_y(y)
#     if X.shape[1] == 0:
#         if binary:
#             prev = float(np.mean(y == y.max()))
#             return prev, 0.0
#         return 0.0, 0.0
#     if binary:
#         n_pos = int((y == y.max()).sum())
#         n_neg = int(len(y) - n_pos)
#         n_splits = min(folds, n_pos, n_neg)
#         if n_splits < 2:
#             return -np.inf, np.inf
#         model = Pipeline(
#             [
#                 ("scaler", StandardScaler()),
#                 ("clf", LogisticRegression(solver="liblinear", max_iter=200)),
#             ]
#         )
#         try:
#             splits = _cached_stratified_kfold_splits(y, n_splits, seed)
#             scores = cross_val_score(model, X, y, cv=splits, scoring="average_precision")
#         except Exception:
#             return -np.inf, np.inf
#     else:
#         n_splits = min(folds, max(2, len(y) // 20))
#         scores = _gram_ols_cv_r2(X, y, n_splits=n_splits, seed=seed)
#         if scores is None:
#             model = LinearRegression()
#             try:
#                 splits = _cached_kfold_splits(len(y), n_splits, seed)
#                 scores = cross_val_score(model, X, y, cv=splits, scoring="r2")
#             except Exception:
#                 return -np.inf, np.inf
#     return float(np.mean(scores)), float(np.std(scores) / np.sqrt(len(scores)))


# def prune_parents_1se(
#     df: pd.DataFrame,
#     target: str,
#     candidates: list[str],
#     seed: int = SEED,
# ) -> list[str]:
#     """Drop a candidate if removing it does not lower CV fit by > 1 SE."""
#     y = df[target].values.astype(float)
#     remaining = list(candidates)
#     improved = True
#     while improved and remaining:
#         improved = False
#         full_mean, full_se = _cv_r2(df[remaining].values.astype(float), y, seed=seed)
#         worst = None
#         worst_drop = None
#         for j, v in enumerate(remaining):
#             reduced = [u for u in remaining if u != v]
#             red_mean, _ = _cv_r2(
#                 df[reduced].values.astype(float) if reduced else np.zeros((len(y), 0)),
#                 y,
#                 seed=seed,
#             )
#             # if full is not better than reduced by > 1 SE, v is inert
#             if full_mean <= red_mean + full_se:
#                 drop = full_mean - red_mean
#                 if worst is None or drop < worst_drop:
#                     worst = v
#                     worst_drop = drop
#         if worst is not None:
#             remaining.remove(worst)
#             improved = True
#     return remaining


# def fci_confounder_pairs(df: pd.DataFrame, vars_: list[str], alpha: float = FCI_ALPHA) -> list[tuple[str, str]]:
#     if len(vars_) < 2:
#         return []
#     try:
#         from causallearn.search.ConstraintBased.FCI import fci
#     except ImportError:
#         return []
#     sub = df[vars_].values.astype(float)
#     # standardize for numerical stability
#     sub = StandardScaler().fit_transform(sub)
#     try:
#         g, edges = fci(sub, "fisherz", alpha, verbose=False, show_progress=False)
#     except Exception:
#         return []
#     suspects = []
#     G = g.graph
#     for i in range(len(vars_)):
#         for j in range(i + 1, len(vars_)):
#             a, b = int(G[i, j]), int(G[j, i])
#             # bidirected / circle marks suggest possible latent
#             if (a == 1 and b == 1) or (a == 2 and b == 2) or (a == 2 and b == 1) or (a == 1 and b == 2):
#                 suspects.append((vars_[i], vars_[j]))
#     return suspects


# def prior_score(parents: list[str], target: str) -> float:
#     score = 0.0
#     for p in parents:
#         score += 1.0 / (1 + abs(TIERS[p] - TIERS[target]))
#         if (p, target) in KNOWN_EDGES:
#             score += 2.0
#     return score


# def stage2_candidates(df: pd.DataFrame, columns: list[str]) -> dict[str, CandidateRecord]:
#     records: dict[str, CandidateRecord] = {}
#     for tgt in columns:
#         if TIERS[tgt] == 1:
#             records[tgt] = CandidateRecord(tgt, (), 0.0, 0.0)
#             continue
#         cands = tier_candidates(tgt, columns)
#         pruned = prune_parents_1se(df, tgt, cands)
#         neigh = list(dict.fromkeys(pruned + [tgt]))
#         suspects = fci_confounder_pairs(df, neigh)
#         flags = [f"{a}↔{b}" for a, b in suspects]
#         mean, _ = _cv_r2(df[pruned].values.astype(float) if pruned else np.zeros((len(df), 0)), df[tgt].values)
#         records[tgt] = CandidateRecord(
#             variable=tgt,
#             parents=tuple(pruned),
#             cv_score=mean,
#             prior=prior_score(pruned, tgt),
#             confounder_flags=flags,
#         )
#     return records


# # ---------------------------------------------------------------------------
# # Stage 3 — mechanism transfer
# # ---------------------------------------------------------------------------
# def _fit_mechanism(train: pd.DataFrame, target: str, parents: list[str]) -> dict:
#     """Fit once on `train`; reused across many test sets by _predict_r2_bundle.

#     Model type is decided from the train labels only (previously it was
#     decided per-call from `_is_binary_y(y_tr) and _is_binary_y(y_te)`, which
#     made the fit itself depend on the test set and blocked reuse across
#     envs). In practice the train set here is always `pooled` — the full
#     frame — so its binary-ness is fixed; a downstream env whose target
#     happens to have <2 classes just gets a NaN score from predict, exactly
#     as the old code's own try/except path would produce for a degenerate
#     average_precision_score call.
#     """
#     y_tr = train[target].values.astype(float)
#     binary = _is_binary_y(y_tr)
#     if binary:
#         pos = float(np.max(y_tr))
#         if not parents:
#             return {"binary": True, "pos": pos, "const": float(np.mean(y_tr == pos))}
#         scaler = StandardScaler()
#         X_tr = scaler.fit_transform(train[parents].values.astype(float))
#         model = LogisticRegression(solver="liblinear", max_iter=200).fit(X_tr, y_tr)
#         return {"binary": True, "pos": pos, "scaler": scaler, "model": model}
#     if not parents:
#         return {"binary": False, "const": float(y_tr.mean())}
#     model = LinearRegression().fit(train[parents].values.astype(float), y_tr)
#     return {"binary": False, "model": model}


# def _predict_r2_bundle(bundle: dict, test: pd.DataFrame, target: str, parents: list[str]) -> float:
#     """Score a fitted `bundle` (from _fit_mechanism) against a held-out `test` set."""
#     y_te = test[target].values.astype(float)
#     if bundle["binary"]:
#         pos = bundle["pos"]
#         if "model" not in bundle:
#             pred = np.full(len(y_te), bundle["const"])
#         else:
#             X_te = bundle["scaler"].transform(test[parents].values.astype(float))
#             model = bundle["model"]
#             pred = model.predict_proba(X_te)[:, list(model.classes_).index(pos)]
#         try:
#             return float(average_precision_score(y_te == pos, pred))
#         except Exception:
#             return float("nan")
#     if "model" not in bundle:
#         pred = np.full_like(y_te, bundle["const"])
#     else:
#         pred = bundle["model"].predict(test[parents].values.astype(float))
#     ss_res = np.sum((y_te - pred) ** 2)
#     ss_tot = np.sum((y_te - y_te.mean()) ** 2) + 1e-12
#     return float(1.0 - ss_res / ss_tot)


# def _fit_predict_r2(train: pd.DataFrame, test: pd.DataFrame, target: str, parents: list[str]) -> float:
#     """One-shot fit+predict for callers that don't reuse the fit across many
#     test sets (e.g. _fit_predict_r2_holdout). mechanism_transfers below fits
#     once via _fit_mechanism and calls _predict_r2_bundle per env instead."""
#     return _predict_r2_bundle(_fit_mechanism(train, target, parents), test, target, parents)


# def _fit_predict_r2_holdout(df: pd.DataFrame, target: str, parents: list[str], seed: int = SEED) -> float:
#     n = len(df)
#     if n < 20:
#         return float("nan")
#     rng = np.random.default_rng(seed)
#     idx = rng.permutation(n)
#     cut = n // 2
#     test = df.iloc[idx[cut:]]
#     if len(test) < 10:
#         return float("nan")
#     train = df.iloc[idx[:cut]]
#     if _is_binary_y(df[target]) and (_outcome_n_pos(test, target) < 1 or _outcome_n_pos(train, target) < 1):
#         return float("nan")
#     return _fit_predict_r2(train, test, target, parents)


# def mechanism_transfers(
#     env_pool: list[EnvWorld],
#     target: str,
#     parents: list[str],
#     pooled: pd.DataFrame,
# ) -> tuple[bool, float, float]:
#     """Pooled mechanism must not be clearly beaten by env-specific holdout (diagram Stage 3.2).

#     Perf note: the pooled model only depends on (pooled, target, parents), which
#     are fixed for the whole call — it does not vary per env, so it is fit once
#     here instead of once per env (previously ~2x per env: once in the loop,
#     once again in the trailing `pooled_scores` comprehension, which recomputed
#     values the loop had already produced).
#     """
#     if len(pooled) < 50:
#         return False, -np.inf, -np.inf
#     pooled_bundle = _fit_mechanism(pooled, target, parents)
#     scores = []
#     pooled_scores = []
#     for env in env_pool:
#         if len(env.data) < 40 or target not in env.data.columns:
#             continue
#         pooled_r2 = _predict_r2_bundle(pooled_bundle, env.data, target, parents)
#         pooled_scores.append(pooled_r2)
#         if _is_binary_y(env.data[target]) and _outcome_n_pos(env.data, target) < 2:
#             continue
#         local_r2 = _fit_predict_r2_holdout(env.data, target, parents)
#         if np.isnan(local_r2):
#             continue
#         scores.append(pooled_r2 - local_r2)
#     if not scores:
#         return False, -np.inf, -np.inf
#     gap_stat = float(np.median(scores))
#     transfers = gap_stat >= -0.25
#     sel = float(np.mean(pooled_scores)) if pooled_scores else -np.inf
#     return transfers, sel, gap_stat


# def bic_penalized_score(n: int, n_params: int, r2: float, y_var: float) -> float:
#     # Gaussian BIC proxy from R²
#     # ll ≈ -n/2 * log(SSR/n) ; SSR = (1-R²)*SST ; SST ≈ n * y_var
#     ssr = max((1.0 - r2), 1e-8) * n * max(y_var, 1e-8)
#     ll = -0.5 * n * np.log(ssr / n + 1e-12)
#     return float(ll - 0.5 * n_params * np.log(n))


# def enumerate_parent_sets(cands: list[str], max_size: int = 4) -> list[tuple[str, ...]]:
#     sets: list[tuple[str, ...]] = [()]
#     for k in range(1, min(max_size, len(cands)) + 1):
#         sets.extend(combinations(cands, k))
#     return sets


# def stage3_select_parents(
#     df: pd.DataFrame,
#     env_pool: list[EnvWorld],
#     stage2: dict[str, CandidateRecord],
#     columns: list[str],
#     record_audit: bool = False,
# ) -> dict[str, tuple[str, ...]]:
#     chosen: dict[str, tuple[str, ...]] = {}
#     global STAGE3_AUDIT
#     if record_audit:
#         STAGE3_AUDIT = {}
#     naturals = [e for e in env_pool if e.tag == "natural"]
#     downs = [e for e in env_pool if e.tag == "downsampled"]
#     score_envs = naturals + downs[::2]
#     for tgt in columns:
#         if TIERS[tgt] == 1:
#             chosen[tgt] = ()
#             continue
#         base = list(stage2[tgt].parents) if tgt in stage2 else tier_candidates(tgt, columns)
#         for a, b in KNOWN_EDGES:
#             if b == tgt and a not in base and a in columns:
#                 base.append(a)
#         base = list(dict.fromkeys(base))
#         y_var = float(df[tgt].var())
#         n = len(df)
#         # Score every candidate set by the cheap test (BIC over a CV R^2) first,
#         # then only pay the expensive test (mechanism_transfers, ~len(score_envs)
#         # model fits) on sets in descending score order, stopping at the first
#         # one that passes. This picks exactly the same set the old code did —
#         # it scored *every* set the same way, but ran mechanism_transfers on
#         # all of them up front regardless of score, including ones that could
#         # never win. Sorting first and short-circuiting means low-scoring sets
#         # that fail (or that we never even need to check) don't pay for the
#         # expensive test at all. Z=() is never filtered by the transfer test
#         # in the original (guarded by `len(Z) > 0`), so it's exempted here too.
#         all_sets = enumerate_parent_sets(base, max_size=min(3, len(base)))
#         scored: list[tuple[float, tuple[str, ...]]] = []
#         for Z in all_sets:
#             mean, _se = _cv_r2(
#                 df[list(Z)].values.astype(float) if Z else np.zeros((n, 0)),
#                 df[tgt].values,
#             )
#             score = bic_penalized_score(n, len(Z) + 1, mean, y_var)
#             if len(Z) == 0 and base:
#                 score -= 5.0
#             scored.append((score, Z))
#         scored.sort(key=lambda sZ: -sZ[0])  # stable: ties keep enumeration order, same as `score > best_score`

#         best_set: tuple[str, ...] = ()
#         best_score = -np.inf
#         for score, Z in scored:
#             if len(Z) == 0:
#                 passes = True
#             else:
#                 ok, _sel, _gap = mechanism_transfers(score_envs, tgt, list(Z), df)
#                 passes = ok or all((p, tgt) in KNOWN_EDGES for p in Z)
#             if passes:
#                 best_set, best_score = Z, score
#                 break  # highest-scoring passing set — nothing later can beat it
#         kept = prune_parents_1se(df, tgt, list(best_set)) if best_set else []
#         for a, b in KNOWN_EDGES:
#             if b == tgt and a in columns and a not in kept:
#                 kept.append(a)
#         chosen[tgt] = tuple(kept)
#         if record_audit:
#             if base:
#                 ok_b, _, gap_b = mechanism_transfers(score_envs, tgt, list(base), df)
#             else:
#                 ok_b, gap_b = True, float("nan")
#             if kept:
#                 ok_c, _, gap_c = mechanism_transfers(score_envs, tgt, kept, df)
#             else:
#                 ok_c, gap_c = True, float("nan")
#             STAGE3_AUDIT[tgt] = {
#                 "stage2_parents": list(base),
#                 "stage2_transfer": "keep" if ok_b else "skip",
#                 "stage2_gap_stat": float(gap_b),
#                 "chosen": list(kept),
#                 "chosen_transfer": "keep" if ok_c else "skip",
#                 "chosen_gap_stat": float(gap_c),
#             }
#     return chosen


# # ---------------------------------------------------------------------------
# # Stage 4 — direction + assemble
# # ---------------------------------------------------------------------------
# def orient_pair(df: pd.DataFrame, a: str, b: str) -> str:
#     """Return 'a→b', 'b→a', or 'undirected' by CV fit + 1-SE."""
#     if is_forbidden(a, b) and not is_forbidden(b, a):
#         return f"{b}→{a}"
#     if is_forbidden(b, a) and not is_forbidden(a, b):
#         return f"{a}→{b}"
#     if TIERS[a] < TIERS[b]:
#         return f"{a}→{b}"
#     if TIERS[b] < TIERS[a]:
#         return f"{b}→{a}"
#     m_ab, se_ab = _cv_r2(df[[a]].values.astype(float), df[b].values)
#     m_ba, se_ba = _cv_r2(df[[b]].values.astype(float), df[a].values)
#     if m_ab > m_ba + se_ba:
#         return f"{a}→{b}"
#     if m_ba > m_ab + se_ab:
#         return f"{b}→{a}"
#     return "undirected"


# def assemble_graph(
#     columns: list[str],
#     parents: dict[str, tuple[str, ...]],
#     df: pd.DataFrame,
#     confounder_pairs: set[tuple[str, str]] | None = None,
# ) -> nx.DiGraph:
#     g = nx.DiGraph()
#     g.add_nodes_from(columns)
#     undirected = []
#     for tgt, pas in parents.items():
#         for p in pas:
#             if is_forbidden(p, tgt):
#                 continue
#             orient = orient_pair(df, p, tgt)
#             if orient == "undirected":
#                 undirected.append((p, tgt))
#             elif orient == f"{p}→{tgt}":
#                 g.add_edge(p, tgt)
#             elif orient == f"{tgt}→{p}":
#                 if not is_forbidden(tgt, p):
#                     g.add_edge(tgt, p)
#     # force known edges
#     for a, b in KNOWN_EDGES:
#         if a in columns and b in columns and not g.has_edge(a, b):
#             g.add_edge(a, b)
#     # break cycles greedily
#     while True:
#         try:
#             cycle = nx.find_cycle(g, orientation="original")
#         except nx.NetworkXNoCycle:
#             break
#         # remove weakest (last) edge on cycle
#         u, v, _ = cycle[-1]
#         g.remove_edge(u, v)
#     g.graph["undirected"] = undirected
#     g.graph["confounders"] = list(confounder_pairs or [])
#     return g


# def graph_to_adj(g: nx.DiGraph, columns: list[str]) -> np.ndarray:
#     d = len(columns)
#     idx = {c: i for i, c in enumerate(columns)}
#     adj = np.zeros((d, d), dtype=int)
#     for u, v in g.edges():
#         adj[idx[u], idx[v]] = 1
#     return adj


# # ---------------------------------------------------------------------------
# # Stage 5 — residual check + light refutation
# # ---------------------------------------------------------------------------
# def residual_independence_scores(df: pd.DataFrame, g: nx.DiGraph) -> dict[str, float]:
#     """|corr(residual, parent)| mean — low is good."""
#     out = {}
#     for node in g.nodes():
#         pas = list(g.predecessors(node))
#         y = df[node].values.astype(float)
#         if not pas:
#             out[node] = 0.0
#             continue
#         model = LinearRegression().fit(df[pas].values.astype(float), y)
#         resid = y - model.predict(df[pas].values.astype(float))
#         corrs = [abs(np.corrcoef(resid, df[p].values.astype(float))[0, 1]) for p in pas]
#         out[node] = float(np.nanmean(corrs))
#     return out


# def refute_edge(df: pd.DataFrame, src: str, tgt: str, seed: int = SEED) -> float:
#     """Fraction of simple refutation checks passed (placebo / subset)."""
#     rng = np.random.default_rng(seed)
#     passed = 0
#     total = 0
#     # baseline effect
#     m = LinearRegression().fit(df[[src]].values.astype(float), df[tgt].values.astype(float))
#     base = abs(float(m.coef_[0]))
#     # placebo: random covariate
#     total += 1
#     fake = rng.normal(size=len(df))
#     m2 = LinearRegression().fit(fake.reshape(-1, 1), df[tgt].values.astype(float))
#     if abs(float(m2.coef_[0])) < base:
#         passed += 1
#     # 30% subset stability
#     total += 1
#     idx = rng.choice(len(df), size=max(50, int(0.3 * len(df))), replace=False)
#     sub = df.iloc[idx]
#     m3 = LinearRegression().fit(sub[[src]].values.astype(float), sub[tgt].values.astype(float))
#     if abs(abs(float(m3.coef_[0])) - base) / (base + 1e-8) < 0.5:
#         passed += 1
#     return passed / total


# # ---------------------------------------------------------------------------
# # Stage 6 — stability selection
# # ---------------------------------------------------------------------------
# def stage2_candidates_light(df: pd.DataFrame, columns: list[str]) -> dict[str, CandidateRecord]:
#     """Stage-2 without FCI — used inside bootstrap loop for load."""
#     records: dict[str, CandidateRecord] = {}
#     for tgt in columns:
#         if TIERS[tgt] == 1:
#             records[tgt] = CandidateRecord(tgt, (), 0.0, 0.0)
#             continue
#         cands = tier_candidates(tgt, columns)
#         pruned = prune_parents_1se(df, tgt, cands)
#         mean, _ = _cv_r2(
#             df[pruned].values.astype(float) if pruned else np.zeros((len(df), 0)),
#             df[tgt].values,
#         )
#         records[tgt] = CandidateRecord(
#             variable=tgt,
#             parents=tuple(pruned),
#             cv_score=mean,
#             prior=prior_score(pruned, tgt),
#             confounder_flags=[],
#         )
#     return records


# def _import_gcm_fit():
#     """Lazy import so bootstrap workers never load DoWhy."""
#     try:
#         import gcm_fit
#         return gcm_fit
#     except ImportError:
#         from kpi_glint import gcm_fit
#         return gcm_fit


# def discover_once(
#     df: pd.DataFrame,
#     env_pool: list[EnvWorld] | None = None,
#     seed: int = SEED,
#     light: bool = False,
#     use_gcm: bool = True,
# ) -> nx.DiGraph:
#     """Run one discovery pass.

#     use_gcm controls whether the (expensive, DoWhy-dependent) GCM parent-set
#     refinement step runs. When False, gcm_fit is never imported and the
#     cheap-screen (sklearn) parent sets from stage3_select_parents are used
#     as-is to assemble the graph.
#     """
#     columns = list(df.columns)
#     if env_pool is None:
#         env_pool = build_env_pool(df, seed=seed)
#     s2 = stage2_candidates_light(df, columns) if light else stage2_candidates(df, columns)
#     if not light:
#         global LAST_STAGE2
#         LAST_STAGE2 = s2
#     parents = stage3_select_parents(df, env_pool, s2, columns, record_audit=not light)
#     if not light and use_gcm:
#         global LAST_GCM_AUDIT
#         gcm_fit = _import_gcm_fit()
#         gcm_fit.require_dowhy()
#         print("GCM refine on cheap-screen survivors", flush=True)
#         parents, LAST_GCM_AUDIT = gcm_fit.refine_parent_sets(
#             df, env_pool, parents, columns, root_tier=TIERS, seed=seed
#         )
#     elif not light and not use_gcm:
#         print("GCM refine skipped (use_gcm=False) — using cheap-screen parent sets as-is", flush=True)
#     conf = set()
#     for rec in s2.values():
#         for flag in rec.confounder_flags:
#             a, b = flag.split("↔")
#             conf.add(tuple(sorted((a, b))))
#     return assemble_graph(columns, parents, df, conf)


# # bootstrap_stability's B replicates are independent (each discover_once call
# # only reads df/env_pool, never mutates shared state) -- embarrassingly
# # parallel across processes. A worker process started via multiprocessing
# # does not inherit bind_dataset()'s already-mutated module globals (a spawned
# # process re-imports this module fresh, back at its placeholder defaults), so
# # _bootstrap_globals_snapshot/_bootstrap_worker_init explicitly copy exactly
# # the globals bind_dataset() sets -- no more, no less -- into each worker.
# def _bootstrap_globals_snapshot() -> dict:
#     return dict(
#         TIERS=TIERS,
#         KNOWN_EDGES=KNOWN_EDGES,
#         COLLINEAR_PAIRS=COLLINEAR_PAIRS,
#         POST_TREATMENT_FORBIDDEN=POST_TREATMENT_FORBIDDEN,
#         FORBIDDEN_EDGES=FORBIDDEN_EDGES,
#         B=B,
#         TARGET_BINARY=globals().get("TARGET_BINARY"),
#         MIN_POSITIVES=MIN_POSITIVES,
#         M_DOWNSAMPLE=M_DOWNSAMPLE,
#         F_KEEP=F_KEEP,
#         MIN_CELL=MIN_CELL,
#         SEED=SEED,
#         columns=globals().get("columns"),
#         n_vars=globals().get("n_vars"),
#     )


# _WORKER_DF: pd.DataFrame | None = None


# def _bootstrap_worker_init(state: dict, df: pd.DataFrame) -> None:
#     global TIERS, KNOWN_EDGES, COLLINEAR_PAIRS, POST_TREATMENT_FORBIDDEN, FORBIDDEN_EDGES
#     global B, TARGET_BINARY, MIN_POSITIVES, M_DOWNSAMPLE, F_KEEP, MIN_CELL, SEED
#     global columns, n_vars, _WORKER_DF
#     TIERS = state["TIERS"]
#     KNOWN_EDGES = state["KNOWN_EDGES"]
#     COLLINEAR_PAIRS = state["COLLINEAR_PAIRS"]
#     POST_TREATMENT_FORBIDDEN = state["POST_TREATMENT_FORBIDDEN"]
#     FORBIDDEN_EDGES = state["FORBIDDEN_EDGES"]
#     B = state["B"]
#     TARGET_BINARY = state["TARGET_BINARY"]
#     MIN_POSITIVES = state["MIN_POSITIVES"]
#     M_DOWNSAMPLE = state["M_DOWNSAMPLE"]
#     F_KEEP = state["F_KEEP"]
#     MIN_CELL = state["MIN_CELL"]
#     SEED = state["SEED"]
#     columns = state["columns"]
#     n_vars = state["n_vars"]
#     _WORKER_DF = df


# def _bootstrap_worker_run(args: tuple[int, int, np.ndarray]) -> tuple[int, list[tuple[str, str]], int, float]:
#     b, seed_b, boot_idx = args
#     t0 = time.time()
#     boot = _WORKER_DF.iloc[boot_idx].reset_index(drop=True)
#     g = discover_once(boot, env_pool=None, seed=seed_b, light=True)
#     edges = list(g.edges())
#     n_pos = _outcome_n_pos(boot)
#     return b, edges, n_pos, time.time() - t0


# def bootstrap_stability(
#     df: pd.DataFrame,
#     B: int = B,
#     seed: int = SEED,
#     cache_dir: Path | None = None,
#     rebuild_envs_each_boot: bool = False,
#     n_jobs: int = 1,
# ) -> tuple[dict[tuple[str, str], float], list[nx.DiGraph]]:
#     """Selection frequency over B bootstraps. Caches edge lists to cache_dir.

#     n_jobs: 1 runs sequentially in this process. Any other value runs the B
#     replicates across worker processes instead (-1 or 0 uses every available
#     core). This only changes WHICH process runs discover_once() for a given
#     replicate, never what data it gets: every replicate's resample indices
#     are drawn up front, in order, from the same single rng the original
#     sequential loop used, so replicate b sees the exact same rows whether
#     n_jobs is 1 or 8. Workers always call discover_once(..., light=True)
#     (sklearn only — no GCM).
#     """
#     rng = np.random.default_rng(seed)
#     columns = list(df.columns)
#     # build env pool once on full data (diagram uses fixed worlds; boots resample observations)
#     base_envs = build_env_pool(df, seed=seed)
#     freq: dict[tuple[str, str], int] = {}
#     graphs_by_b: dict[int, nx.DiGraph] = {}
#     cache_dir = Path(cache_dir) if cache_dir else None
#     if cache_dir:
#         cache_dir.mkdir(parents=True, exist_ok=True)

#     def _record(b: int, edges: list[tuple[str, str]], n_pos: int, elapsed: float) -> None:
#         g = nx.DiGraph()
#         g.add_nodes_from(columns)
#         g.add_edges_from(edges)
#         graphs_by_b[b] = g
#         for e in edges:
#             freq[e] = freq.get(e, 0) + 1
#         cache_path = cache_dir / f"boot_{b:03d}.json" if cache_dir else None
#         if cache_path:
#             cache_path.write_text(json.dumps(edges))
#         print(f"  bootstrap {b + 1}/{B}: |E|={len(edges)} n_pos={n_pos} ({elapsed:.1f}s)", flush=True)

#     # Same control flow as the original loop: draw idx (consuming rng) only
#     # for replicates that aren't already cached, in b order -- so the rng
#     # sees exactly the same sequence of .choice() calls either way.
#     boot_indices: dict[int, np.ndarray] = {}
#     pending: list[int] = []
#     for b in range(B):
#         cache_path = cache_dir / f"boot_{b:03d}.json" if cache_dir else None
#         if cache_path and cache_path.exists():
#             edges = [tuple(e) for e in json.loads(cache_path.read_text())]
#             g = nx.DiGraph()
#             g.add_nodes_from(columns)
#             g.add_edges_from(edges)
#             graphs_by_b[b] = g
#             for e in edges:
#                 freq[e] = freq.get(e, 0) + 1
#             continue
#         boot_indices[b] = rng.choice(len(df), size=len(df), replace=True)
#         pending.append(b)

#     if n_jobs == 1 or not pending:
#         for b in pending:
#             boot = df.iloc[boot_indices[b]].reset_index(drop=True)
#             t0 = time.time()
#             if rebuild_envs_each_boot:
#                 build_env_pool(boot, seed=seed + b + 1)  # kept for parity with the original call; unused, as before
#             g = discover_once(boot, env_pool=None, seed=seed + b, light=True)
#             edges = list(g.edges())
#             _record(b, edges, _outcome_n_pos(boot), time.time() - t0)
#     else:
#         import os
#         from concurrent.futures import ProcessPoolExecutor, as_completed

#         workers = (os.cpu_count() or 1) if n_jobs <= 0 else n_jobs
#         workers = max(1, min(workers, len(pending)))
#         state = _bootstrap_globals_snapshot()
#         with ProcessPoolExecutor(
#             max_workers=workers, initializer=_bootstrap_worker_init, initargs=(state, df)
#         ) as ex:
#             futures = [ex.submit(_bootstrap_worker_run, (b, seed + b, boot_indices[b])) for b in pending]
#             for fut in as_completed(futures):
#                 b, edges, n_pos, elapsed = fut.result()
#                 _record(b, edges, n_pos, elapsed)

#     graphs = [graphs_by_b[b] for b in range(B)]
#     freqs = {e: c / B for e, c in freq.items()}
#     return freqs, graphs


# def meinhausen_buhlmann_threshold(freqs: dict[tuple[str, str], float], avg_edges: float, ev_max: float = EV_MAX) -> float:
#     """Derive keep-threshold π_thr so E[V] ≤ ev_max (simplified MB bound).

#     Uses p = n_vars (the actual number of discovery columns), not the
#     Sachs diagram's hardcoded p=11 — see build_natural_strata's docstring
#     above for why only this version (previously nested in bind_dataset)
#     is kept.
#     """
#     p = max(n_vars, 2)
#     binom = p * (p - 1)
#     q = max(avg_edges, 1.0)
#     thr = 0.5 * (1.0 + (q ** 2) / (ev_max * binom))
#     return float(min(0.95, max(0.5, thr)))


# def edge_strengths(df: pd.DataFrame, g: nx.DiGraph) -> dict[tuple[str, str], float]:
#     strengths = {}
#     for node in g.nodes():
#         pas = list(g.predecessors(node))
#         if not pas:
#             continue
#         model = LinearRegression().fit(df[pas].values.astype(float), df[node].values.astype(float))
#         coefs = np.abs(model.coef_)
#         s = coefs / (coefs.sum() + 1e-12)
#         for p, w in zip(pas, s):
#             strengths[(p, node)] = float(w)
#     return strengths


# # ---------------------------------------------------------------------------
# # Stage 7 — experiment ranking
# # ---------------------------------------------------------------------------
# def rank_experiments(
#     undecided: list[tuple[str, str]],
#     df: pd.DataFrame,
#     g: nx.DiGraph,
# ) -> list[dict[str, Any]]:
#     ranks = []
#     for a, b in undecided:
#         # expected info proxy: variance of a in high-variance stratum of b's residual
#         info = float(df[a].var() * (1.0 + df[b].var()))
#         ranks.append(
#             {
#                 "intervene_on": a,
#                 "resolves_edge": f"{a}—{b}",
#                 "expected_info": info,
#                 "suggested": f"Perturb {a} and re-measure {b} transfer",
#             }
#         )
#     ranks.sort(key=lambda r: -r["expected_info"])
#     return ranks


# # ---------------------------------------------------------------------------
# # Driver: causal_graph(weakest_pillar_output_path, output_path)
# #
# # This used to be notebook 03's remaining cells (candidate-set setup,
# # bind_dataset, plotting helpers, discover/bootstrap/fallback/save). The
# # only change from the notebook version: it no longer computes the
# # weak-pillar-filter decision itself (that moved to weakest_pillar.py) and
# # no longer reads dataset/kpi_list.py or kpi_meta.csv directly — pillar and
# # parent info now come from weakest_pillar's selected_kpis.csv. Everything
# # else (build_matrix, bind_dataset, run_discovery, the no-parents-found
# # fallback, the sink audit, the plots) is unchanged.
# # ---------------------------------------------------------------------------
# import os as _os
# _os.environ.setdefault("MPLBACKEND", "Agg")

# import time
# import matplotlib.pyplot as plt


# def _default_artifacts_path(output_path: Path) -> Path:
#     parts = list(output_path.parts)
#     for i in range(len(parts) - 1, -1, -1):
#         if parts[i] == "output":
#             parts[i] = "artifacts"
#             return Path(*parts)
#     return output_path.parent / "artifacts" / output_path.name


# def _write_profile_log(profiler, log_path: Path, stage_name: str, elapsed: float) -> None:
#     """Write a full per-function cProfile report (sorted by cumulative time,
#     then by self/tottime) plus the raw .pstats dump next to it, so it can be
#     re-sorted or visualized later (snakeviz, gprof2dot, ...) without
#     re-running the pipeline. Pure reporting -- does not affect the run.
#     """
#     import io
#     import pstats

#     log_path = Path(log_path)
#     log_path.parent.mkdir(parents=True, exist_ok=True)
#     pstats_path = log_path.with_suffix(".pstats")
#     profiler.dump_stats(str(pstats_path))

#     buf = io.StringIO()
#     buf.write(f"=== {stage_name} profile ===\n")
#     buf.write(f"wall time: {elapsed:.2f}s\n\n")
#     buf.write("--- top 40 by cumulative time (this function + everything it calls) ---\n")
#     pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("cumulative").print_stats(40)
#     buf.write("\n--- top 40 by self time (time spent in this function alone) ---\n")
#     pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("tottime").print_stats(40)
#     log_path.write_text(buf.getvalue(), encoding="utf-8")
#     print(f"{stage_name} profile -> {log_path}  (raw stats -> {pstats_path})")


# def causal_graph(
#     weakest_pillar_output_path,
#     output_path,
#     artifacts_path=None,
#     profile: bool = False,
#     profile_log: str | None = None,
#     n_jobs: int = -1,
#     use_gcm: bool = True,
# ) -> dict:
#     """Thin entry point around _causal_graph_impl (notebook 03's discovery
#     logic, completely unchanged below). Set profile=True to run the same
#     call under cProfile and get a full per-function timing report written
#     next to the other artifacts -- this only wraps the call, it does not
#     change it. n_jobs controls how many worker processes run the bootstrap
#     replicates: -1 (default) uses every available core; 1 is sequential.

#     use_gcm (default True): when True, the DoWhy GCM refinement/scoring step
#     runs (parent-set refinement via gcm_fit.refine_parent_sets, arrow-strength
#     scoring, and graph falsification). When False, gcm_fit/DoWhy is never
#     imported -- the graph is assembled directly from the cheap sklearn/env-
#     based screen (stage2/stage3), and edge strengths come from a plain
#     linear-regression |coef| fallback instead of DoWhy's arrow_strength.
#     """
#     if not profile:
#         return _causal_graph_impl(weakest_pillar_output_path, output_path, artifacts_path, n_jobs=n_jobs, use_gcm=use_gcm)
#     import cProfile

#     profiler = cProfile.Profile()
#     t0 = time.time()
#     profiler.enable()
#     try:
#         result = _causal_graph_impl(weakest_pillar_output_path, output_path, artifacts_path, n_jobs=n_jobs, use_gcm=use_gcm)
#     finally:
#         profiler.disable()
#     elapsed = time.time() - t0
#     log_path = Path(profile_log) if profile_log else result["artifacts"] / "profile_causal_graph.log"
#     _write_profile_log(profiler, log_path, "causal_graph", elapsed)
#     return result


# def _causal_graph_impl(weakest_pillar_output_path, output_path, artifacts_path=None, n_jobs: int = -1, use_gcm: bool = True) -> dict:
#     """Run notebook 03's discovery logic end to end.

#     weakest_pillar_output_path: folder holding weakest_pillar()'s 3 main
#                   outputs (selected_kpis.csv, panel.csv, run.json).
#     output_path:  folder to write edges.csv, the one main output.
#     artifacts_path: folder for everything else (edges.json, metrics.json,
#                   graph.png, the discovery-matrix snapshot, bootstrap cache);
#                   defaults to swapping the last "output" path segment for
#                   "artifacts".
#     """
#     WP_OUTPUT = Path(weakest_pillar_output_path)
#     OUTPUT_MAIN = Path(output_path)
#     ARTIFACTS = Path(artifacts_path) if artifacts_path else _default_artifacts_path(OUTPUT_MAIN)
#     OUTPUT = ARTIFACTS  # historical name from the notebook; every `OUTPUT / ...` write below is an artifact
#     FIGURES = ARTIFACTS / "figures"
#     FIGURES.mkdir(parents=True, exist_ok=True)
#     OUTPUT.mkdir(parents=True, exist_ok=True)
#     OUTPUT_MAIN.mkdir(parents=True, exist_ok=True)

#     run = json.loads((WP_OUTPUT / "run.json").read_text())
#     panel = pd.read_csv(WP_OUTPUT / "panel.csv")
#     sel_df = pd.read_csv(WP_OUTPUT / "selected_kpis.csv")
#     print("panel", panel.shape, "weak pillars", run.get("pillars"))
#     print("selected", sel_df["kpi_id"].tolist())

#     TARGET = run["target"]
#     if TARGET not in panel.columns:
#         raise ValueError(
#             f"target {TARGET!r} is not on the panel. "
#             "Re-run weakest_pillar with this KPI added to the catalog."
#         )
#     pillars = list(run.get("pillars") or [])
#     inbound = [c for c in run.get("inbound") or [] if c in panel.columns]
#     DROP = set(run.get("drop_from_discovery") or run.get("drop_from_influence") or [])
#     pillar_of = dict(zip(sel_df["kpi_id"], sel_df["pillar"]))
#     parent_of = dict(zip(sel_df["kpi_id"], sel_df["parent"]))
#     SELECTED = [k for k in sel_df["kpi_id"].tolist() if k in panel.columns]

#     # rec_targets: all recommended KPIs (from weakest_pillar's multi-target loop).
#     # These get tier 3 in build_matrix — they are near-outcome nodes that can both
#     # cause and be caused by other KPIs, unlike the single tier-4 sink (TARGET).
#     REC_TARGETS: set[str] = set(run.get("rec_targets") or [TARGET])

#     # rec_target flag column may or may not exist in older selected_kpis.csv files
#     if "rec_target" in sel_df.columns:
#         _rec_from_csv = set(sel_df.loc[sel_df["rec_target"].fillna(False).astype(bool), "kpi_id"])
#         REC_TARGETS = REC_TARGETS | _rec_from_csv

#     kpi_cols = [
#         k for k in sel_df.loc[sel_df["primary_candidate"], "kpi_id"].tolist()
#         if k in panel.columns
#     ]
#     print("discovery candidate KPIs (weakest_pillar's primary_candidate list):", kpi_cols)
#     print("rec_targets (tier-3 near-outcome nodes):", sorted(REC_TARGETS))

#     def class_coverage_ok(df: pd.DataFrame, target: str, col: str, thresh: float = 0.5) -> bool:
#         y = pd.to_numeric(df[target], errors="coerce")
#         s = pd.to_numeric(df[col], errors="coerce")
#         y_ok = y.dropna()
#         if y_ok.nunique() > 2:
#             return True
#         for cls in y_ok.unique():
#             mask = y == cls
#             if mask.any() and float(s.loc[mask].isna().mean()) > thresh:
#                 return False
#         return True


#     def build_matrix(cand_kpis: list[str], verbose: bool = True) -> tuple[list[str], list[str], pd.DataFrame, dict[str, int]]:
#         """Turn a candidate KPI list into the (core_kpis, cols, disc matrix, tiers) that
#         GLINT discovery runs on. Reused as-is for the weak-pillar fallback below."""
#         core = [k for k in cand_kpis if k != TARGET and k not in DROP]
#         dropped_recode = [k for k in cand_kpis if k != TARGET and k in DROP]
#         if dropped_recode and verbose:
#             print("dropped recodes", dropped_recode)
#         dropped_cov = [k for k in core if not class_coverage_ok(panel, TARGET, k)]
#         if dropped_cov and verbose:
#             print("dropped (missing on a target class):", dropped_cov)
#         core = [k for k in core if k not in dropped_cov]

#         c_cols: list[str] = []
#         for c in core + inbound + [TARGET]:
#             if c in panel.columns and c not in c_cols and c not in DROP:
#                 c_cols.append(c)

#         w = panel[c_cols].copy()
#         for c in w.columns:
#             if w[c].dtype == object or str(w[c].dtype) in ("string", "category"):
#                 w[c] = w[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
#             else:
#                 w[c] = pd.to_numeric(w[c], errors="coerce")

#         bins = [c for c in w.columns if w[c].nunique(dropna=True) <= 2]
#         rider_drop = []
#         for c in list(w.columns):
#             if c == TARGET:
#                 continue
#             if any(not class_coverage_ok(w, b, c) for b in bins):
#                 rider_drop.append(c)
#         if rider_drop:
#             if verbose:
#                 print("dropped (binary-rider class coverage):", rider_drop)
#             w = w.drop(columns=rider_drop)
#             c_cols = [c for c in c_cols if c not in rider_drop]
#             core = [k for k in core if k not in rider_drop]

#         d = w.dropna().reset_index(drop=True)
#         if verbose:
#             print("discovery matrix", d.shape, "columns", list(d.columns))
#         if TARGET in d.columns and verbose:
#             panel_pos = int(pd.to_numeric(panel[TARGET], errors="coerce").eq(1).sum()) if TARGET in panel.columns else 0
#             disc_pos = int(pd.to_numeric(d[TARGET], errors="coerce").eq(1).sum())
#             print(
#                 f"TARGET {TARGET}: n={len(d)} n_pos={disc_pos} "
#                 f"prevalence={disc_pos / max(len(d), 1):.4f} "
#                 f"panel_pos={panel_pos} lost_to_dropna={panel_pos - disc_pos}"
#             )
#             print("target value counts:\n", d[TARGET].value_counts(dropna=False).head().to_string())

#         t: dict[str, int] = {}
#         core_set = set(core)
#         in_set = set(inbound)
#         graph_set = set(d.columns)
#         for c in d.columns:
#             if c == TARGET:
#                 # Tier 4: the single causal-graph sink (explicit or inferred target)
#                 t[c] = 4
#             elif c in in_set:
#                 # Tier 1: context / inbound variables (roots)
#                 t[c] = 1
#             elif c in REC_TARGETS:
#                 # Tier 3: recommended KPIs are near-outcome nodes — they can be
#                 # caused by operational KPIs (tier 2) and can themselves cause
#                 # other tier-3 or tier-4 nodes.  This replaces the old flat-tier-2
#                 # assignment that lumped all non-target KPIs together.
#                 t[c] = 3
#             elif parent_of.get(c) in graph_set and parent_of.get(c) != c:
#                 # Tier 3: derived / child KPI (expression depends on a parent KPI)
#                 t[c] = 3
#             elif c in core_set:
#                 t[c] = 2
#             else:
#                 t[c] = 2

#         pd.DataFrame({"variable": list(d.columns), "tier": [t[c] for c in d.columns]}).to_csv(
#             ARTIFACTS / "schema.csv", index=False
#         )
#         d.to_csv(ARTIFACTS / "discovery_panel.csv", index=False)
#         if verbose:
#             print(pd.Series(t).sort_values())
#         return core, c_cols, d, t


#     # The full selected set (pre weak-pillar filter) — kept around as the fallback
#     # candidate pool below: if the weak-pillar pool turns out to have no parents
#     # for TARGET at all, we retry discovery against this instead.
#     kpi_cols_fallback = list(SELECTED)

#     core_kpis, cols, disc, tiers = build_matrix(kpi_cols)


#     # Bind this run's tiers/knobs (rebinds the empty engine-module placeholders).
#     # Wrapped as a function so cell 12 can rebind against the fallback candidate
#     # set (kpi_cols_fallback) if the weak-pillar pool turns out to have no
#     # parents for TARGET. build_natural_strata / meinhausen_buhlmann_threshold
#     # below read these as globals, so re-running this function is enough — the
#     # functions themselves don't need to be redefined.
#     def bind_dataset(d: pd.DataFrame, t: dict[str, int]) -> None:
#         global TIERS, KNOWN_EDGES, COLLINEAR_PAIRS, POST_TREATMENT_FORBIDDEN, FORBIDDEN_EDGES
#         global B, TARGET, TARGET_BINARY, MIN_POSITIVES, M_DOWNSAMPLE, F_KEEP, MIN_CELL, SEED
#         global columns, n_vars

#         TIERS = dict(t)
#         KNOWN_EDGES = []

#         # Read collinear pairs from run.json (set by weakest_pillar from kpis.jsonld).
#         # run["collinear"] is a list of [a, b] pairs written by weakest_pillar when
#         # it serialises spec.collinear; fall back to an empty set if not present.
#         COLLINEAR_PAIRS = {tuple(pair) for pair in (run.get("collinear") or [])}

#         # Post-outcome pillars: any pillar whose stage == "post_outcome" in kpis.jsonld.
#         # These KPIs are forbidden from being causes of ANY tier-3/4 operational node
#         # (not just the single TARGET sink), because in the multi-target model every
#         # recommended KPI can be an outcome endpoint.
#         post_outcome_pillars: set[str] = set(run.get("post_outcome_pillars") or [])
#         post_outcome_kpis = {k for k, p in pillar_of.items() if p in post_outcome_pillars}
#         POST_TREATMENT_FORBIDDEN = set()
#         if post_outcome_pillars:
#             # Tier-3/4 nodes whose pillar is NOT post_outcome are operational outcomes
#             # — post_outcome KPIs must not point at them.
#             operational_outcomes = {
#                 c for c in d.columns
#                 if TIERS.get(c, 2) >= 3
#                 and pillar_of.get(c) not in post_outcome_pillars
#             }
#             POST_TREATMENT_FORBIDDEN = {
#                 (r, op)
#                 for r in post_outcome_kpis if r in d.columns
#                 for op in operational_outcomes
#             }

#         FORBIDDEN_EDGES = [(c, p) for p, t_p in TIERS.items() for c, t_c in TIERS.items() if t_c > t_p]
#         B = int(run.get("bootstrap_B") or 8)
#         # Ensure TARGET is always a real module-level global so _outcome_n_pos() can
#         # find it via globals() even before bind_dataset() is called during startup.
#         TARGET = run["target"]
#         TARGET_BINARY = TARGET in d.columns and int(pd.to_numeric(d[TARGET], errors="coerce").nunique(dropna=True)) == 2
#         MIN_POSITIVES = 20 if TARGET_BINARY else 0
#         M_DOWNSAMPLE = 10
#         F_KEEP = 0.70
#         MIN_CELL = max(30, min(120, len(d) // 100))
#         SEED = 0
#         columns = list(d.columns)
#         n_vars = len(columns)


#     bind_dataset(disc, tiers)


#     # build_natural_strata / meinhausen_buhlmann_threshold read TIERS, columns,
#     # MIN_CELL, MIN_POSITIVES, n_vars as globals (set just above) — no need to
#     # redefine them here; the single module-level versions pick up the new
#     # values automatically on the next call.
#     print("tiers bound", {k: TIERS[k] for k in columns})
#     print("B", B, "min_cell", MIN_CELL, "min_positives", MIN_POSITIVES, "target_binary", TARGET_BINARY)
#     print("POST_TREATMENT_FORBIDDEN", sorted(POST_TREATMENT_FORBIDDEN))


#     # GLINT graph plots — GT overlay when truth exists, else plain discovered structure.
#     # (Path, Any already imported at module level above.)

#     from matplotlib.lines import Line2D

#     COLORS = {
#         "true": "#2CA02C",
#         "spur": "#D62728",
#         "rev": "#FF7F0E",
#         "miss": "#7F7F7F",
#         "edge": "#2B6CB0",
#         "node": "#E8F1FB",
#         "node_iso": "#F0F0F0",
#         "node_edge": "#2C3E50",
#     }


#     def _label(v: str, labels: dict[str, str] | None) -> str:
#         if not labels:
#             return v
#         return labels.get(v, v)


#     def _pairs_to_edges(pairs: list) -> list[tuple[str, str]]:
#         out = []
#         for p in pairs:
#             if isinstance(p, (list, tuple)) and len(p) >= 2:
#                 out.append((str(p[0]), str(p[1])))
#         return out


#     def draw_overlay(
#         ax,
#         score: dict[str, Any],
#         title: str,
#         all_vars: list[str],
#         labels: dict[str, str] | None = None,
#         node_pos: dict[str, tuple[float, float]] | None = None,
#     ):
#         G = nx.DiGraph()
#         G.add_nodes_from(all_vars)
#         if node_pos is None:
#             pos = nx.circular_layout(G)
#         else:
#             pos = {v: node_pos[v] for v in all_vars if v in node_pos}
#             missing = [v for v in all_vars if v not in pos]
#             if missing:
#                 extra = nx.circular_layout(missing)
#                 pos.update(extra)

#         nx.draw_networkx_nodes(
#             G, pos, ax=ax, node_color="#E8F1FB", node_size=900, edgecolors="#333333", linewidths=1.0
#         )
#         nx.draw_networkx_labels(
#             G,
#             pos,
#             labels={v: _label(v, labels) for v in all_vars},
#             ax=ax,
#             font_size=8,
#             font_weight="bold",
#         )

#         def _draw(pairs, color, style="solid", alpha=0.9, width=2.0):
#             edges = _pairs_to_edges(pairs)
#             if not edges:
#                 return
#             nx.draw_networkx_edges(
#                 G,
#                 pos,
#                 edgelist=edges,
#                 ax=ax,
#                 edge_color=color,
#                 style=style,
#                 width=width,
#                 alpha=alpha,
#                 arrows=True,
#                 arrowsize=14,
#                 connectionstyle="arc3,rad=0.08",
#                 min_source_margin=12,
#                 min_target_margin=12,
#             )

#         _draw(score["etrue"], COLORS["true"])
#         _draw(score["espur"], COLORS["spur"])
#         _draw(score["efals"], COLORS["rev"])
#         _draw(score["emiss"], COLORS["miss"], style="dashed", alpha=0.7, width=1.5)

#         ax.set_title(title, fontsize=10)
#         ax.set_axis_off()
#         ax.margins(0.15)


#     def save_colored_graph(
#         metrics: dict,
#         columns: list[str],
#         labels: dict[str, str],
#         out_path: Path,
#         title: str,
#     ):
#         """Green=true, red=spurious, orange=reversed, dashed gray=missing (bnlearn / GT runs)."""
#         d = len(columns)
#         all_vars = [f"X{i + 1}" for i in range(d)]
#         xlabels = {f"X{i + 1}": labels.get(columns[i], columns[i]) for i in range(d)}
#         score = {
#             "etrue": metrics.get("true_edges") or [],
#             "espur": metrics.get("spur_edges") or [],
#             "efals": metrics.get("fals_edges") or [],
#             "emiss": metrics.get("miss_edges") or [],
#         }
#         fig, ax = plt.subplots(figsize=(7.5, 6.0))
#         draw_overlay(ax, score, title, all_vars, labels=xlabels)
#         legend = [
#             Line2D([0], [0], color=COLORS["true"], lw=2, label=f"true ({metrics['etrue']})"),
#             Line2D([0], [0], color=COLORS["spur"], lw=2, label=f"spurious ({metrics['espur']})"),
#             Line2D([0], [0], color=COLORS["rev"], lw=2, label=f"reversed ({metrics['efals']})"),
#             Line2D([0], [0], color=COLORS["miss"], lw=2, ls="--", label=f"missing ({metrics['emiss']})"),
#         ]
#         ax.legend(
#             handles=legend,
#             loc="lower center",
#             ncol=2,
#             frameon=False,
#             fontsize=8,
#             bbox_to_anchor=(0.5, -0.08),
#         )
#         fig.tight_layout()
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(out_path, dpi=160, bbox_inches="tight")
#         plt.close(fig)


#     def _tier_tree_layout(
#         columns: list[str],
#         tiers: dict[str, int],
#         edges: list[tuple[str, str]],
#     ) -> dict[str, tuple[float, float]]:
#         """Left→right parent→child columns by tier; isolates sit in a bottom band.

#         Within each tier, order nodes with a barycenter pass to cut crossings.
#         """
#         G = nx.DiGraph()
#         G.add_nodes_from(columns)
#         G.add_edges_from(edges)
#         active = {n for n in columns if G.degree(n) > 0}
#         isolates = [n for n in columns if n not in active]

#         by_tier: dict[int, list[str]] = {}
#         for n in active:
#             by_tier.setdefault(int(tiers.get(n, 1)), []).append(n)
#         tier_ids = sorted(by_tier.keys())

#         # Seed order: out-degree then name
#         order: dict[int, list[str]] = {
#             t: sorted(by_tier[t], key=lambda n: (-G.out_degree(n), -G.in_degree(n), n))
#             for t in tier_ids
#         }

#         def _bary(nodes: list[str], nbrs_fn) -> list[str]:
#             rank = {n: i for i, n in enumerate(nodes)}

#             def key(n: str):
#                 nbrs = [rank[u] for u in nbrs_fn(n) if u in rank]
#                 return (sum(nbrs) / len(nbrs) if nbrs else len(nodes) / 2.0, n)

#             return sorted(nodes, key=key)

#         # Forward + backward barycenter sweeps
#         for _ in range(4):
#             for i in range(1, len(tier_ids)):
#                 prev = {n: j for j, n in enumerate(order[tier_ids[i - 1]])}
#                 order[tier_ids[i]] = _bary(
#                     order[tier_ids[i]],
#                     lambda n, p=prev: [u for u in G.predecessors(n) if u in p],
#                 )
#             for i in range(len(tier_ids) - 2, -1, -1):
#                 nxt = {n: j for j, n in enumerate(order[tier_ids[i + 1]])}
#                 order[tier_ids[i]] = _bary(
#                     order[tier_ids[i]],
#                     lambda n, p=nxt: [v for v in G.successors(n) if v in p],
#                 )

#         x_gap = 3.6
#         y_gap = 1.55
#         pos: dict[str, tuple[float, float]] = {}
#         for xi, t in enumerate(tier_ids):
#             nodes = order[t]
#             n = len(nodes)
#             ys = np.linspace((n - 1) / 2.0, -(n - 1) / 2.0, n) if n > 1 else [0.0]
#             for y, node in zip(ys, nodes):
#                 pos[node] = (xi * x_gap, float(y) * y_gap)

#         if isolates:
#             iso_y = min((p[1] for p in pos.values()), default=0.0) - 2.4
#             n = len(isolates)
#             xs = np.linspace(0.0, max(len(tier_ids) - 1, 1) * x_gap, n) if n > 1 else [0.0]
#             for x, node in zip(xs, sorted(isolates)):
#                 pos[node] = (float(x), iso_y)

#         return pos


#     def save_structure_graph(
#         out_path: Path,
#         columns: list[str],
#         edges: list[tuple[str, str]],
#         *,
#         tiers: dict[str, int] | None = None,
#         labels: dict[str, str] | None = None,
#         strengths: dict[tuple[str, str], float] | None = None,
#         truth: np.ndarray | None = None,
#         title: str = "",
#         subtitle: str | None = None,
#     ):
#         """Hierarchical parent→child plot with strength labels.

#         - No / empty truth → single-color discovered edges (no true/spur/rev legend).
#         - Non-empty truth → GT overlay colors + missing dashed.
#         """
#         labels = labels or {c: c for c in columns}
#         strengths = strengths or {}
#         tiers = tiers or {c: 1 for c in columns}
#         idx = {c: i for i, c in enumerate(columns)}

#         has_gt = truth is not None and int(np.sum(truth != 0)) > 0

#         G = nx.DiGraph()
#         G.add_nodes_from(columns)
#         G.add_edges_from(edges)
#         pos = _tier_tree_layout(columns, tiers, edges)

#         active = [n for n in columns if G.degree(n) > 0]
#         isolates = [n for n in columns if n not in active]

#         n_tiers = max((tiers.get(c, 1) for c in active), default=1) - min(
#             (tiers.get(c, 1) for c in active), default=1
#         ) + 1
#         fig_w = max(10.0, 2.8 * max(n_tiers, 2) + 2)
#         fig_h = max(7.0, 0.55 * max(len(active), 4) + (2.5 if isolates else 0.8))
#         fig, ax = plt.subplots(figsize=(fig_w, fig_h))

#         if active:
#             nx.draw_networkx_nodes(
#                 G,
#                 pos,
#                 nodelist=active,
#                 ax=ax,
#                 node_color=COLORS["node"],
#                 node_size=1600,
#                 edgecolors=COLORS["node_edge"],
#                 linewidths=1.2,
#             )
#         if isolates:
#             nx.draw_networkx_nodes(
#                 G,
#                 pos,
#                 nodelist=isolates,
#                 ax=ax,
#                 node_color=COLORS["node_iso"],
#                 node_size=1100,
#                 edgecolors="#AAAAAA",
#                 linewidths=1.0,
#                 alpha=0.85,
#             )
#         nx.draw_networkx_labels(
#             G,
#             pos,
#             labels={n: labels.get(n, n) for n in columns},
#             ax=ax,
#             font_size=7.5,
#             font_weight="bold",
#         )

#         edge_labels: dict[tuple[str, str], str] = {}
#         for a, b in edges:
#             s = strengths.get((a, b), strengths.get((str(a), str(b)), None))
#             if s is None:
#                 continue
#             edge_labels[(a, b)] = f"{float(s):.2f}"

#         def _edge_meta(a: str, b: str) -> tuple[str, float]:
#             """Return (color, width) for an edge."""
#             w = 1.2 + 3.5 * float(strengths.get((a, b), 0.35))
#             if not has_gt:
#                 return COLORS["edge"], w
#             ia, ib = idx[a], idx[b]
#             if truth[ia, ib] != 0:
#                 return COLORS["true"], w
#             if truth[ib, ia] != 0:
#                 return COLORS["rev"], w
#             return COLORS["spur"], w

#         # Slight fan of curvatures so parallel parents don't sit on one line
#         for i, (a, b) in enumerate(edges):
#             color, width = _edge_meta(a, b)
#             rad = 0.06 + 0.04 * ((i % 5) - 2)
#             nx.draw_networkx_edges(
#                 G,
#                 pos,
#                 edgelist=[(a, b)],
#                 ax=ax,
#                 edge_color=color,
#                 width=width,
#                 arrows=True,
#                 arrowstyle="-|>",
#                 arrowsize=14,
#                 connectionstyle=f"arc3,rad={rad}",
#                 min_source_margin=18,
#                 min_target_margin=18,
#                 alpha=0.9,
#             )

#         if has_gt:
#             miss = []
#             for i, u in enumerate(columns):
#                 for j, v in enumerate(columns):
#                     if truth[i, j] != 0 and (u, v) not in set(edges):
#                         miss.append((u, v))
#             if miss:
#                 # add missing endpoints already in G
#                 nx.draw_networkx_edges(
#                     G,
#                     pos,
#                     edgelist=miss,
#                     ax=ax,
#                     edge_color=COLORS["miss"],
#                     style="dashed",
#                     width=1.3,
#                     arrows=True,
#                     arrowsize=12,
#                     connectionstyle="arc3,rad=0.05",
#                     alpha=0.65,
#                     min_source_margin=14,
#                     min_target_margin=14,
#                 )

#         if edge_labels:
#             # Manual mid-edge labels with staggered offsets (nx edge_labels collide heavily)
#             for i, ((a, b), txt) in enumerate(edge_labels.items()):
#                 if a not in pos or b not in pos:
#                     continue
#                 x0, y0 = pos[a]
#                 x1, y1 = pos[b]
#                 t = 0.45 + 0.08 * ((i % 3) - 1)
#                 mx = x0 + t * (x1 - x0)
#                 my = y0 + t * (y1 - y0)
#                 # perpendicular nudge
#                 dx, dy = x1 - x0, y1 - y0
#                 norm = (dx * dx + dy * dy) ** 0.5 + 1e-9
#                 ox, oy = -dy / norm, dx / norm
#                 nudge = 0.18 * (1 if i % 2 == 0 else -1)
#                 ax.text(
#                     mx + nudge * ox,
#                     my + nudge * oy,
#                     txt,
#                     fontsize=7,
#                     color="#1A365D",
#                     ha="center",
#                     va="center",
#                     zorder=5,
#                     bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#CBD5E0", "alpha": 0.92},
#                 )

#         full_title = title
#         if subtitle:
#             full_title = f"{title}\n{subtitle}"
#         ax.set_title(full_title, fontsize=11, pad=12)
#         ax.set_axis_off()
#         ax.margins(0.08)

#         if has_gt:
#             legend = [
#                 Line2D([0], [0], color=COLORS["true"], lw=2, label="true"),
#                 Line2D([0], [0], color=COLORS["spur"], lw=2, label="spurious"),
#                 Line2D([0], [0], color=COLORS["rev"], lw=2, label="reversed"),
#                 Line2D([0], [0], color=COLORS["miss"], lw=2, ls="--", label="missing"),
#                 Line2D([0], [0], color="none", label="labels = edge strength"),
#             ]
#         else:
#             legend = [
#                 Line2D([0], [0], color=COLORS["edge"], lw=2, label="discovered edge"),
#                 Line2D(
#                     [0],
#                     [0],
#                     marker="o",
#                     color="w",
#                     markerfacecolor=COLORS["node"],
#                     markeredgecolor=COLORS["node_edge"],
#                     markersize=10,
#                     label="in graph",
#                 ),
#                 Line2D(
#                     [0],
#                     [0],
#                     marker="o",
#                     color="w",
#                     markerfacecolor=COLORS["node_iso"],
#                     markeredgecolor="#AAAAAA",
#                     markersize=10,
#                     label="no edges",
#                 ),
#                 Line2D([0], [0], color="none", label="edge labels = strength"),
#             ]
#         ax.legend(
#             handles=legend,
#             loc="lower center",
#             ncol=4,
#             frameon=False,
#             fontsize=8,
#             bbox_to_anchor=(0.5, -0.06),
#         )

#         fig.tight_layout()
#         out_path = Path(out_path)
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(out_path, dpi=170, bbox_inches="tight")
#         plt.close(fig)


#     def run_discovery(d: pd.DataFrame):
#         env_pool_ = build_env_pool(d, m=M_DOWNSAMPLE, f_keep=F_KEEP, seed=SEED)
#         env_pos_ = pd.DataFrame(
#             [
#                 {
#                     "name": e.name,
#                     "tag": e.tag,
#                     "n": len(e.data),
#                     "n_pos": _outcome_n_pos(e.data),
#                 }
#                 for e in env_pool_
#             ]
#         )
#         print(f"envs={len(env_pool_)} natural={sum(e.tag=='natural' for e in env_pool_)}")
#         if len(env_pos_):
#             print(env_pos_.to_string(index=False))
#             print("env n_pos min/median", int(env_pos_.n_pos.min()), float(env_pos_.n_pos.median()))
#         env_pos_.to_csv(ARTIFACTS / "env_positive_counts.csv", index=False)
#         graph_ = discover_once(d, env_pool=env_pool_, seed=SEED, light=False, use_gcm=use_gcm)
#         print("full |E|", graph_.number_of_edges())
#         boot_dir_ = ARTIFACTS / "boot"
#         freqs_, boot_graphs_ = bootstrap_stability(d, B=B, seed=SEED, cache_dir=boot_dir_, n_jobs=n_jobs)
#         avg_e_ = float(np.mean([gg.number_of_edges() for gg in boot_graphs_])) if boot_graphs_ else 0.0
#         thr_ = meinhausen_buhlmann_threshold(freqs_, avg_e_)
#         stable_ = [(a, b) for (a, b), f in freqs_.items() if f >= thr_]
#         if use_gcm:
#             gcm_fit = _import_gcm_fit()
#             gcm_meta_ = {}
#             print("GCM scoring assembled graph (arrow_strength + falsify)", flush=True)
#             try:
#                 strengths_, gcm_ll = gcm_fit.arrow_strengths_normalized(d, graph_, seed=SEED)
#                 gcm_meta_["gcm_ll"] = gcm_ll
#             except Exception as exc:
#                 print(f"GCM arrow_strength failed ({exc}); using sklearn |coef| strengths", flush=True)
#                 strengths_ = edge_strengths(d, graph_)
#                 gcm_meta_["gcm_ll"] = float("nan")
#                 gcm_meta_["gcm_strength_error"] = str(exc)
#             try:
#                 falsify_info = gcm_fit.falsify_dag(d, graph_, seed=SEED)
#                 gcm_meta_["falsify_p"] = falsify_info.get("p_value")
#                 gcm_meta_["falsified"] = falsify_info.get("falsified")
#                 gcm_meta_["falsifiable"] = falsify_info.get("falsifiable")
#             except Exception as exc:
#                 print(f"GCM falsify_graph failed ({exc})", flush=True)
#                 gcm_meta_["falsify_p"] = None
#                 gcm_meta_["falsify_error"] = str(exc)
#         else:
#             print("GCM scoring skipped (use_gcm=False) — using sklearn |coef| strengths", flush=True)
#             strengths_ = edge_strengths(d, graph_)
#             gcm_meta_ = {
#                 "gcm_ll": None,
#                 "falsify_p": None,
#                 "falsified": None,
#                 "falsifiable": None,
#                 "gcm_skipped": True,
#             }
#         graph_.graph["gcm"] = gcm_meta_
#         print(f"keep π={thr_:.3f} avg|E|={avg_e_:.1f} stable={len(stable_)}")
#         return env_pool_, graph_, freqs_, boot_graphs_, thr_, stable_, strengths_


#     t0 = time.time()
#     env_pool, graph, freqs, boot_graphs, thr, stable, strengths = run_discovery(disc)

#     # --- fallback: if the weak-pillar pool found no parents at all for ANY
#     # tier-3/4 node (rec targets + TARGET), retry discovery against the full
#     # selected-KPI set from Step 0/02 instead of quietly shipping a disconnected graph.
#     fallback_used = False
#     incoming_to_target = [a for (a, b) in stable if b == TARGET]
#     # Tier-3 nodes are the rec targets; tier-4 is TARGET itself.
#     outcome_nodes = {c for c in columns if TIERS.get(c, 2) >= 3}
#     stable_set = set(stable)
#     any_outcome_has_parent = any(
#         (a, b) in stable_set
#         for (a, b) in stable_set
#         if b in outcome_nodes
#     )
#     if not any_outcome_has_parent and pillars and set(kpi_cols) != set(kpi_cols_fallback):
#         print(
#             f"no parents found for any tier-3/4 node {sorted(outcome_nodes)} "
#             f"in the weak-pillar pool {pillars} — "
#             "falling back to the full selected KPI set from Step 0 (pre weak-pillar filter)."
#         )
#         fallback_used = True
#         kpi_cols = list(kpi_cols_fallback)
#         core_kpis, cols, disc, tiers = build_matrix(kpi_cols)
#         bind_dataset(disc, tiers)
#         env_pool, graph, freqs, boot_graphs, thr, stable, strengths = run_discovery(disc)
#         incoming_to_target = [a for (a, b) in stable if b == TARGET]
#         outcome_nodes = {c for c in columns if TIERS.get(c, 2) >= 3}
#         stable_set = set(stable)
#         any_outcome_has_parent = any(
#             (a, b) in stable_set for (a, b) in stable_set if b in outcome_nodes
#         )
#         if not any_outcome_has_parent:
#             print(f"still no parents found for any tier-3/4 node even against the full selected set.")

#     print("=== sink audit ===")
#     s2_map = globals().get("LAST_STAGE2") or {}
#     s3_map = globals().get("STAGE3_AUDIT") or {}
#     gcm_map = globals().get("LAST_GCM_AUDIT") or {}
#     n_boot = len(boot_graphs) or B
#     incoming_tgt = [a for (a, b) in freqs if b == TARGET and a in disc.columns]
#     rec_tgt = s2_map.get(TARGET)
#     stage2_tgt = list(rec_tgt.parents) if rec_tgt is not None else []
#     corr_cols = [c for c in dict.fromkeys(incoming_tgt + stage2_tgt) if c in disc.columns]
#     if len(corr_cols) >= 2:
#         print(f"candidate corr into {TARGET} (vote-split check):")
#         print(disc[corr_cols].corr().round(4).to_string())
#     audit_children = [c for c in columns if TIERS.get(c, 1) > 1]
#     if TARGET in audit_children:
#         audit_children = [TARGET] + [c for c in audit_children if c != TARGET]
#     for child in audit_children:
#         rec = s2_map.get(child)
#         parents2 = list(rec.parents) if rec is not None else []
#         info = s3_map.get(child) or {}
#         gap = info.get("stage2_gap_stat", float("nan"))
#         xfer = info.get("stage2_transfer", "n/a")
#         chosen = info.get("chosen", [])
#         chosen_xfer = info.get("chosen_transfer", "n/a")
#         chosen_gap = info.get("chosen_gap_stat", float("nan"))
#         incoming = [(a, float(f), int(round(float(f) * n_boot))) for (a, b), f in freqs.items() if b == child]
#         incoming.sort(key=lambda r: -r[1])
#         gcm_info = gcm_map.get(child) or {}
#         gcm_xfer = gcm_info.get("transfer", "n/a" if gcm_info.get("skipped") else gcm_info.get("error", "n/a"))
#         gcm_out = gcm_info.get("parents_out", gcm_info.get("parents", []))
#         print(
#             f"{child} Stage-2 parents={parents2} transfer={xfer} gap_stat={gap} "
#             f"Stage-3 chosen={chosen} transfer={chosen_xfer} gap_stat={chosen_gap} "
#             f"GCM transfer={gcm_xfer} parents={gcm_out}"
#         )
#         print(f"{child} bootstrap into child: {incoming if incoming else 'none'} / {n_boot}")

#     edge_rows = [
#         {
#             "source": a,
#             "target": b,
#             "confidence": float(freqs.get((a, b), 1.0)),
#             "strength": float(strengths.get((a, b), 0.0)),
#         }
#         for a, b in stable
#     ]
#     edges = pd.DataFrame(edge_rows)
#     edges.to_csv(OUTPUT_MAIN / "edges.csv", index=False)  # <-- the one main output
#     graph_json = {
#         "directed": True,
#         "nodes": [{"id": c, "tier": int(tiers.get(c, 2))} for c in columns],
#         "edges": edge_rows,
#     }
#     (ARTIFACTS / "edges.json").write_text(json.dumps(graph_json, indent=2), encoding="utf-8")
#     gcm_meta = dict(graph.graph.get("gcm") or {})

#     def _json_num(v):
#         if isinstance(v, (float, np.floating)) and not np.isfinite(v):
#             return None
#         if isinstance(v, (np.floating, np.integer)):
#             return v.item()
#         return v

#     metrics = {
#         "pillars": pillars,
#         "fallback_used": fallback_used,
#         "candidate_kpis": kpi_cols,
#         "n_rows": int(len(disc)),
#         "n_vars": n_vars,
#         "n_envs": len(env_pool),
#         "B": B,
#         "keep_threshold": thr,
#         "n_stable_edges": len(stable),
#         "n_full_edges": int(graph.number_of_edges()),
#         "seconds": time.time() - t0,
#         "gcm_ll": _json_num(gcm_meta.get("gcm_ll")),
#         "falsify_p": _json_num(gcm_meta.get("falsify_p")),
#         "falsified": gcm_meta.get("falsified"),
#         "falsifiable": gcm_meta.get("falsifiable"),
#     }
#     if gcm_meta.get("gcm_strength_error"):
#         metrics["gcm_strength_error"] = gcm_meta["gcm_strength_error"]
#     if gcm_meta.get("falsify_error"):
#         metrics["falsify_error"] = gcm_meta["falsify_error"]
#     (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
#     pd.DataFrame([metrics]).to_csv(ARTIFACTS / "glint_summary.csv", index=False)
#     print(edges)

#     tag = "FALLBACK:full_selected" if fallback_used else "+".join(pillars)
#     subtitle = (
#         "no parents found for target in the weak-pillar pool — fell back to the full selected set"
#         if fallback_used
#         else "joint parent search on weak-pillar KPIs + context + target (labels = strength)"
#     )
#     out_png = FIGURES / "03_glint_graph.png"
#     save_structure_graph(
#         out_png,
#         columns,
#         list(stable),
#         tiers=tiers,
#         labels={c: c for c in columns},
#         strengths={(r["source"], r["target"]): r["strength"] for r in edge_rows},
#         truth=None,
#         title=f"GLINT merged [{tag}]  |E|={len(stable)}",
#         subtitle=subtitle,
#     )
#     # also keep a copy next to edges
#     save_structure_graph(
#         ARTIFACTS / "graph.png",
#         columns,
#         list(stable),
#         tiers=tiers,
#         labels={c: c for c in columns},
#         strengths={(r["source"], r["target"]): r["strength"] for r in edge_rows},
#         truth=None,
#         title=f"GLINT merged [{tag}]  |E|={len(stable)}",
#         subtitle=subtitle,
#     )
#     print(metrics)
#     return {"output": OUTPUT_MAIN, "artifacts": ARTIFACTS, "edges": edges, "metrics": metrics}


# if __name__ == "__main__":
#     import argparse

#     ap = argparse.ArgumentParser(description="Run GLINT discovery on weakest_pillar's output.")
#     ap.add_argument("weakest_pillar_output_path", help="Folder with selected_kpis.csv, panel.csv, run.json")
#     ap.add_argument("output_path", help="Folder to write edges.csv")
#     ap.add_argument("--artifacts-path", default=None)
#     ap.add_argument("--profile", action="store_true", help="Write a per-function cProfile timing report to artifacts/")
#     ap.add_argument(
#         "--n-jobs",
#         type=int,
#         default=-1,
#         help="Worker processes for the bootstrap loop. -1 (default) = all cores. 1 = sequential.",
#     )
#     ap.add_argument(
#         "--gcm",
#         dest="use_gcm",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#         help="Use DoWhy GCM for parent-set refinement and arrow-strength scoring (default: true). "
#              "Pass --no-gcm to skip gcm_fit/DoWhy entirely and assemble the graph straight from "
#              "the cheap sklearn screen, using linear-regression |coef| edge strengths instead.",
#     )
#     args = ap.parse_args()
#     causal_graph(
#         args.weakest_pillar_output_path,
#         args.output_path,
#         artifacts_path=args.artifacts_path,
#         profile=args.profile,
#         n_jobs=args.n_jobs,
#         use_gcm=args.use_gcm,
#     )

"""GLINT core — Stages 0-8 causal discovery algorithm helpers.

Dataset-agnostic: every knob below (TIERS, KNOWN_EDGES, B, M_DOWNSAMPLE,
F_KEEP, MIN_CELL, ...) is a placeholder default, unconditionally
rebound by bind_dataset() (inside causal_graph(), below) before any of it is
actually used. The module-level values just need to exist so the functions
below have something to import/reference at definition time; they carry no
information about any particular dataset (this engine was originally
prototyped against the Sachs et al. protein-signaling benchmark, and the
placeholders below used to be that dataset's actual variable names and known
edges — that data has been removed since it never described this pipeline's
KPI data and was live only by coincidence of cell-execution order).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Knobs — all rebound per-run by bind_dataset(); see its docstring.
# ---------------------------------------------------------------------------
B = 8
M_DOWNSAMPLE = 20
F_KEEP = 0.70
FCI_ALPHA = 0.01
EV_MAX = 1.0
MIN_CELL = 300
SEED = 0
CV_FOLDS = 5
MIN_POSITIVES = 0
# TARGET and TARGET_BINARY are set as module globals by bind_dataset() so that
# _outcome_n_pos() (which reads globals()) can find them from any call site.
# In multi-target mode (no single declared target), TARGET is None.
TARGET: str | None = ""
TARGET_BINARY: bool = False

# variable -> stage tier.  KG-based scheme (from kpis.jsonld depth):
#   Tier 0 = root context vars (TOTAL_PRICE, N_ITEMS, ...)
#   Tier 1 = derived context vars (ORDER_COMPLEXITY = f(N_ITEMS, N_SELLERS))
#   Tier 2 = root KPIs (no parent in KG: DELAY_DAYS, IS_CANCELED, ...)
#   Tier 3 = child KPIs (has a parent KPI in KG: IS_ON_TIME, IS_NEGATIVE_REVIEW)
# Populated by bind_dataset() from this run's actual TIERS before any
# Stage 0-8 function below reads it.
TIERS: dict[str, int] = {}

# Set of rec_target KPI IDs, used by build_schema for role assignment.
# Populated by bind_dataset().
REC_TARGETS_SET: set[str] = set()

# (parent, child) pairs to force into the graph regardless of what discovery
# finds. Populated by bind_dataset(); empty by default.
KNOWN_EDGES: list[tuple[str, str]] = []

# Reverse tier jumps and sink -> root edges. Derived from TIERS, so it's
# recomputed by bind_dataset() alongside it; empty here since TIERS is empty.
FORBIDDEN_EDGES: list[tuple[str, str]] = []


@dataclass
class EnvWorld:
    name: str
    tag: str  # natural | downsampled | near-experiment
    data: pd.DataFrame


@dataclass
class CandidateRecord:
    variable: str
    parents: tuple[str, ...]
    cv_score: float
    prior: float
    confounder_flags: list[str] = field(default_factory=list)


@dataclass
class EdgeRecord:
    source: str
    target: str
    confidence: float = 0.0
    strength: float = 0.0
    trust: float = 0.0
    direction: str = "→"
    confounder: bool = False
    audit: str = ""


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------
def build_schema(columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        rows.append(
            {
                "variable": c,
                "kind": "continuous",
                "time_status": "current",
                "role": "context" if TIERS[c] <= 1 else ("outcome" if c in REC_TARGETS_SET else "mediator"),
                "tier": TIERS[c],
                "deterministic": False,
                "in_discovery": True,
            }
        )
    return pd.DataFrame(rows)


def basis_vars() -> list[str]:
    return [v for v, t in TIERS.items() if t <= 1]


def is_forbidden(src: str, tgt: str) -> bool:
    if TIERS[src] > TIERS[tgt]:
        return True
    return (src, tgt) in FORBIDDEN_EDGES


# ---------------------------------------------------------------------------
# Stage 1 — environments
# ---------------------------------------------------------------------------
def _bin_series(s: pd.Series, n_bins: int) -> pd.Series:
    try:
        return pd.qcut(s, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.cut(s, bins=n_bins, labels=False, duplicates="drop")


def _is_binary_y(y) -> bool:
    arr = np.asarray(y, dtype=float)
    u = np.unique(arr[~np.isnan(arr)])
    return len(u) == 2


def _n_pos_series(s: pd.Series) -> int:
    y = pd.to_numeric(s, errors="coerce")
    if 1 in y.dropna().unique() or 1.0 in y.dropna().unique():
        return int((y == 1).sum())
    if y.nunique(dropna=True) == 2:
        return int((y == y.max()).sum())
    return 0


def _outcome_n_pos(df: pd.DataFrame, col: str | None = None) -> int:
    # TARGET is set as a module global by bind_dataset(); use globals() to read it.
    name = col or globals().get("TARGET") or ""
    if not name or name not in df.columns:
        return 0
    return _n_pos_series(df[name])


def build_natural_strata(df: pd.DataFrame, min_cell: int = MIN_CELL) -> list[EnvWorld]:
    """Natural strata from the tier-1 (context) columns, binned into tertiles.

    This replaces the original Sachs-diagram version (hardcoded "X1"/"X2"
    protein roots), which only ever ran in the notebooks by coincidence of
    cell-execution order — bind_dataset()'s dataset-specific redefinition
    always overwrote it before it was ever called. Keeping only this version,
    at module scope, is what the notebooks actually executed; it's also what
    bind_dataset() below expects (it just sets TIERS/MIN_CELL/etc. as globals
    rather than redefining this function).
    """
    roots = [v for v, t in TIERS.items() if t == 1 and v in df.columns]
    all_cols = list(df.columns)
    if not roots:
        roots = all_cols[: min(2, len(all_cols))]
    work = df.copy()
    work["_b0"] = _bin_series(work[roots[0]], 3)
    work["_b1"] = _bin_series(work[roots[1]], 3) if len(roots) > 1 else 0
    extra = next((c for c in all_cols if c not in roots), roots[0])
    work["_b2"] = _bin_series(work[extra], 3)
    work["_cell"] = work["_b0"].astype(str) + "_" + work["_b1"].astype(str) + "_" + work["_b2"].astype(str)
    envs: list[EnvWorld] = []
    leftovers = []
    for cell, g in work.groupby("_cell", sort=True):
        sub = g.drop(columns=["_b0", "_b1", "_b2", "_cell"])
        if len(sub) >= min_cell and (MIN_POSITIVES <= 0 or _outcome_n_pos(sub) >= MIN_POSITIVES):
            envs.append(EnvWorld(name=f"natural_{cell}", tag="natural", data=sub.reset_index(drop=True)))
        else:
            leftovers.append(sub)
    if leftovers:
        merged = pd.concat(leftovers, ignore_index=True)
        if len(merged) >= max(20, min_cell // 2) and (MIN_POSITIVES <= 0 or _outcome_n_pos(merged) >= MIN_POSITIVES):
            envs.append(EnvWorld(name="natural_merged_small", tag="natural", data=merged))
    return envs


def _root_stratum_ids(df: pd.DataFrame) -> np.ndarray:
    roots = basis_vars()
    if not roots:
        return np.zeros(len(df), dtype=int)
    b0 = _bin_series(df[roots[0]], 4).fillna(0).astype(int).values
    if len(roots) > 1:
        b1 = _bin_series(df[roots[1]], 4).fillna(0).astype(int).values
    else:
        b1 = np.zeros(len(df), dtype=int)
    return b0 * 10 + b1


def build_downsampled_envs(
    df: pd.DataFrame,
    m: int = M_DOWNSAMPLE,
    f_keep: float = F_KEEP,
    seed: int = SEED,
    max_env_n: int = 2500,
) -> list[EnvWorld]:
    """Fixed-fraction downsampling toward m target root-mix menus (diagram Stage 1.2)."""
    rng = np.random.default_rng(seed)
    stratum = _root_stratum_ids(df)
    uniq = np.unique(stratum)
    n = len(df)
    envs: list[EnvWorld] = []
    target_weights = []
    target_weights.append(np.ones(len(uniq)) / len(uniq))
    for i in range(min(len(uniq), max(1, m // 3))):
        w = np.full(len(uniq), 0.05 / max(len(uniq) - 1, 1))
        w[i] = 0.95
        target_weights.append(w)
    while len(target_weights) < m:
        w = rng.dirichlet(np.ones(len(uniq)) * 0.5)
        target_weights.append(w)
    target_weights = target_weights[:m]

    for k, tw in enumerate(target_weights):
        keep_idx = []
        for j, s in enumerate(uniq):
            members = np.where(stratum == s)[0]
            n_keep = int(round(f_keep * tw[j] * n))
            n_keep = min(n_keep, len(members))
            if n_keep <= 0:
                continue
            chosen = rng.choice(members, size=n_keep, replace=False)
            keep_idx.append(chosen)
        if not keep_idx:
            continue
        idx = np.concatenate(keep_idx)
        if len(idx) > max_env_n:
            idx = rng.choice(idx, size=max_env_n, replace=False)
        if len(idx) < 50:
            continue
        sub = df.iloc[idx].reset_index(drop=True)
        if MIN_POSITIVES > 0:
            n_pos = _outcome_n_pos(sub)
            if n_pos < MIN_POSITIVES:
                tgt = globals().get("TARGET")
                if tgt and tgt in df.columns:
                    pos_idx = np.where(pd.to_numeric(df[tgt], errors="coerce").eq(1).to_numpy())[0]
                    need = MIN_POSITIVES - n_pos
                    extra = np.setdiff1d(pos_idx, idx, assume_unique=False)
                    if len(extra) >= need:
                        extra = rng.choice(extra, size=need, replace=False)
                        idx = np.concatenate([idx, extra])
                        sub = df.iloc[idx].reset_index(drop=True)
                        n_pos = _outcome_n_pos(sub)
            if n_pos < MIN_POSITIVES:
                continue
        envs.append(EnvWorld(name=f"down_{k:02d}", tag="downsampled", data=sub))
    return envs


def build_env_pool(
    df: pd.DataFrame,
    m: int = M_DOWNSAMPLE,
    f_keep: float = F_KEEP,
    seed: int = SEED,
) -> list[EnvWorld]:
    """Combine natural strata with fixed-fraction downsampled environments."""
    return build_natural_strata(df) + build_downsampled_envs(
        df, m=m, f_keep=f_keep, seed=seed
    )




# ---------------------------------------------------------------------------
# Stage 2 — candidates + 1-SE prune + FCI screen
# ---------------------------------------------------------------------------
COLLINEAR_PAIRS: set[tuple[str, str]] = set()
POST_TREATMENT_FORBIDDEN: set[tuple[str, str]] = set()


def tier_candidates(target: str, columns: list[str]) -> list[str]:
    t = TIERS[target]
    cands = []
    for v in columns:
        if v == target:
            continue
        if is_forbidden(v, target):
            continue
        if (v, target) in COLLINEAR_PAIRS or (target, v) in COLLINEAR_PAIRS:
            continue
        if (v, target) in POST_TREATMENT_FORBIDDEN:
            continue
        # earlier tier always; same tier only if known or adjacent cascade
        if TIERS[v] < t:
            cands.append(v)
        elif TIERS[v] == t and ((v, target) in KNOWN_EDGES or (target, v) in KNOWN_EDGES):
            cands.append(v)
        elif (v, target) in KNOWN_EDGES:
            cands.append(v)
    # always inject known parents
    for a, b in KNOWN_EDGES:
        if b == target and a not in cands and a in columns:
            cands.append(a)
    return cands



def _scale_X(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        return X
    return StandardScaler().fit_transform(X)


# KFold/StratifiedKFold.split() only depends on (n_rows, n_splits, seed) for
# plain KFold, or (y's class labels, n_splits, seed) for StratifiedKFold --
# never on which candidate columns are in X. _cv_r2 is called many times with
# the same y (all of prune_parents_1se's drop-the-worst-candidate iterations
# for one target share it; every bootstrap replicate reuses the same row
# count n) and a fresh KFold/StratifiedKFold object recomputes the same split
# from scratch every single call (np.unique/argsort/sort -- real, measured
# cost on real data). Caching the split and passing it to cross_val_score as
# an explicit list of (train_idx, test_idx) pairs -- which sklearn accepts in
# place of a splitter object, using it as-is instead of calling .split()
# again -- gives byte-identical folds without recomputing them.
_KFOLD_CACHE: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = {}
_KFOLD_CACHE_MAXSIZE = 256


def _cached_kfold_splits(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    key = ("kfold", n, n_splits, seed)
    cached = _KFOLD_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_KFOLD_CACHE) >= _KFOLD_CACHE_MAXSIZE:
        _KFOLD_CACHE.clear()
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros((n, 1))))
    _KFOLD_CACHE[key] = splits
    return splits


def _cached_stratified_kfold_splits(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    # Keyed on y's actual bytes (not id(y)) so two different arrays that
    # happen to reuse the same memory address across calls can never collide.
    key = ("skfold", y.tobytes(), n_splits, seed)
    cached = _KFOLD_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_KFOLD_CACHE) >= _KFOLD_CACHE_MAXSIZE:
        _KFOLD_CACHE.clear()
    splits = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y)
    )
    _KFOLD_CACHE[key] = splits
    return splits


def _gram_ols_cv_r2(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray | None:
    """Closed-form OLS-with-intercept CV R^2, replacing cross_val_score(LinearRegression)
    for the continuous target path. Same KFold(shuffle=True, random_state=seed) splits and
    the same r2 definition sklearn uses, computed directly with np.linalg.solve instead of
    paying sklearn's per-fold estimator clone/validation/dispatch overhead (the dominant
    cost for a handful of variables). Falls back to None (caller uses cross_val_score) on
    any numerical issue, e.g. a rank-deficient design on a degenerate/collinear subset.
    """
    n, p = X.shape
    try:
        splits = _cached_kfold_splits(n, n_splits, seed)
    except Exception:
        return None
    Xa = np.hstack([np.ones((n, 1)), X])
    scores = []
    jitter = 1e-9 * np.eye(p + 1)
    for train_idx, test_idx in splits:
        Xtr, ytr = Xa[train_idx], y[train_idx]
        Xte, yte = Xa[test_idx], y[test_idx]
        try:
            beta = np.linalg.solve(Xtr.T @ Xtr + jitter, Xtr.T @ ytr)
        except np.linalg.LinAlgError:
            return None
        pred = Xte @ beta
        ss_res = float(np.sum((yte - pred) ** 2))
        ss_tot = float(np.sum((yte - yte.mean()) ** 2))
        scores.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)
    return np.asarray(scores, dtype=float)


def _cv_r2(X: np.ndarray, y: np.ndarray, folds: int = CV_FOLDS, seed: int = SEED) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    binary = _is_binary_y(y)
    if X.shape[1] == 0:
        if binary:
            prev = float(np.mean(y == y.max()))
            return prev, 0.0
        return 0.0, 0.0
    if binary:
        n_pos = int((y == y.max()).sum())
        n_neg = int(len(y) - n_pos)
        n_splits = min(folds, n_pos, n_neg)
        if n_splits < 2:
            return -np.inf, np.inf
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(solver="liblinear", max_iter=200)),
            ]
        )
        try:
            splits = _cached_stratified_kfold_splits(y, n_splits, seed)
            scores = cross_val_score(model, X, y, cv=splits, scoring="average_precision")
        except Exception:
            return -np.inf, np.inf
    else:
        n_splits = min(folds, max(2, len(y) // 20))
        scores = _gram_ols_cv_r2(X, y, n_splits=n_splits, seed=seed)
        if scores is None:
            model = LinearRegression()
            try:
                splits = _cached_kfold_splits(len(y), n_splits, seed)
                scores = cross_val_score(model, X, y, cv=splits, scoring="r2")
            except Exception:
                return -np.inf, np.inf
    return float(np.mean(scores)), float(np.std(scores) / np.sqrt(len(scores)))


def prune_parents_1se(
    df: pd.DataFrame,
    target: str,
    candidates: list[str],
    seed: int = SEED,
) -> list[str]:
    """Drop a candidate if removing it does not lower CV fit by > 1 SE."""
    y = df[target].values.astype(float)
    remaining = list(candidates)
    improved = True
    while improved and remaining:
        improved = False
        full_mean, full_se = _cv_r2(df[remaining].values.astype(float), y, seed=seed)
        worst = None
        worst_drop = None
        for j, v in enumerate(remaining):
            reduced = [u for u in remaining if u != v]
            red_mean, _ = _cv_r2(
                df[reduced].values.astype(float) if reduced else np.zeros((len(y), 0)),
                y,
                seed=seed,
            )
            # if full is not better than reduced by > 1 SE, v is inert
            if full_mean <= red_mean + full_se:
                drop = full_mean - red_mean
                if worst is None or drop < worst_drop:
                    worst = v
                    worst_drop = drop
        if worst is not None:
            remaining.remove(worst)
            improved = True
    return remaining


def fci_confounder_pairs(df: pd.DataFrame, vars_: list[str], alpha: float = FCI_ALPHA) -> list[tuple[str, str]]:
    if len(vars_) < 2:
        return []
    try:
        from causallearn.search.ConstraintBased.FCI import fci
    except ImportError:
        return []
    sub = df[vars_].values.astype(float)
    # standardize for numerical stability
    sub = StandardScaler().fit_transform(sub)
    try:
        g, edges = fci(sub, "fisherz", alpha, verbose=False, show_progress=False)
    except Exception:
        return []
    suspects = []
    G = g.graph
    for i in range(len(vars_)):
        for j in range(i + 1, len(vars_)):
            a, b = int(G[i, j]), int(G[j, i])
            # bidirected / circle marks suggest possible latent
            if (a == 1 and b == 1) or (a == 2 and b == 2) or (a == 2 and b == 1) or (a == 1 and b == 2):
                suspects.append((vars_[i], vars_[j]))
    return suspects


def prior_score(parents: list[str], target: str) -> float:
    score = 0.0
    for p in parents:
        score += 1.0 / (1 + abs(TIERS[p] - TIERS[target]))
        if (p, target) in KNOWN_EDGES:
            score += 2.0
    return score


def stage2_candidates(df: pd.DataFrame, columns: list[str]) -> dict[str, CandidateRecord]:
    records: dict[str, CandidateRecord] = {}
    for tgt in columns:
        if TIERS[tgt] == 1:
            records[tgt] = CandidateRecord(tgt, (), 0.0, 0.0)
            continue
        cands = tier_candidates(tgt, columns)
        pruned = prune_parents_1se(df, tgt, cands)
        neigh = list(dict.fromkeys(pruned + [tgt]))
        suspects = fci_confounder_pairs(df, neigh)
        flags = [f"{a}↔{b}" for a, b in suspects]
        mean, _ = _cv_r2(df[pruned].values.astype(float) if pruned else np.zeros((len(df), 0)), df[tgt].values)
        records[tgt] = CandidateRecord(
            variable=tgt,
            parents=tuple(pruned),
            cv_score=mean,
            prior=prior_score(pruned, tgt),
            confounder_flags=flags,
        )
    return records


# ---------------------------------------------------------------------------
# Stage 3 — mechanism transfer
# ---------------------------------------------------------------------------
def _fit_mechanism(train: pd.DataFrame, target: str, parents: list[str]) -> dict:
    """Fit once on `train`; reused across many test sets by _predict_r2_bundle.

    Model type is decided from the train labels only (previously it was
    decided per-call from `_is_binary_y(y_tr) and _is_binary_y(y_te)`, which
    made the fit itself depend on the test set and blocked reuse across
    envs). In practice the train set here is always `pooled` — the full
    frame — so its binary-ness is fixed; a downstream env whose target
    happens to have <2 classes just gets a NaN score from predict, exactly
    as the old code's own try/except path would produce for a degenerate
    average_precision_score call.
    """
    y_tr = train[target].values.astype(float)
    binary = _is_binary_y(y_tr)
    if binary:
        pos = float(np.max(y_tr))
        if not parents:
            return {"binary": True, "pos": pos, "const": float(np.mean(y_tr == pos))}
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[parents].values.astype(float))
        model = LogisticRegression(solver="liblinear", max_iter=200).fit(X_tr, y_tr)
        return {"binary": True, "pos": pos, "scaler": scaler, "model": model}
    if not parents:
        return {"binary": False, "const": float(y_tr.mean())}
    model = LinearRegression().fit(train[parents].values.astype(float), y_tr)
    return {"binary": False, "model": model}


def _predict_r2_bundle(bundle: dict, test: pd.DataFrame, target: str, parents: list[str]) -> float:
    """Score a fitted `bundle` (from _fit_mechanism) against a held-out `test` set."""
    y_te = test[target].values.astype(float)
    if bundle["binary"]:
        pos = bundle["pos"]
        if "model" not in bundle:
            pred = np.full(len(y_te), bundle["const"])
        else:
            X_te = bundle["scaler"].transform(test[parents].values.astype(float))
            model = bundle["model"]
            pred = model.predict_proba(X_te)[:, list(model.classes_).index(pos)]
        try:
            return float(average_precision_score(y_te == pos, pred))
        except Exception:
            return float("nan")
    if "model" not in bundle:
        pred = np.full_like(y_te, bundle["const"])
    else:
        pred = bundle["model"].predict(test[parents].values.astype(float))
    ss_res = np.sum((y_te - pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _fit_predict_r2(train: pd.DataFrame, test: pd.DataFrame, target: str, parents: list[str]) -> float:
    """One-shot fit+predict for callers that don't reuse the fit across many
    test sets (e.g. _fit_predict_r2_holdout). mechanism_transfers below fits
    once via _fit_mechanism and calls _predict_r2_bundle per env instead."""
    return _predict_r2_bundle(_fit_mechanism(train, target, parents), test, target, parents)


def _fit_predict_r2_holdout(df: pd.DataFrame, target: str, parents: list[str], seed: int = SEED) -> float:
    n = len(df)
    if n < 20:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = n // 2
    test = df.iloc[idx[cut:]]
    if len(test) < 10:
        return float("nan")
    train = df.iloc[idx[:cut]]
    if _is_binary_y(df[target]) and (_outcome_n_pos(test, target) < 1 or _outcome_n_pos(train, target) < 1):
        return float("nan")
    return _fit_predict_r2(train, test, target, parents)


def mechanism_transfers(
    env_pool: list[EnvWorld],
    target: str,
    parents: list[str],
    pooled: pd.DataFrame,
) -> tuple[bool, float, float]:
    """Pooled mechanism must not be clearly beaten by env-specific holdout (diagram Stage 3.2).

    Perf note: the pooled model only depends on (pooled, target, parents), which
    are fixed for the whole call — it does not vary per env, so it is fit once
    here instead of once per env (previously ~2x per env: once in the loop,
    once again in the trailing `pooled_scores` comprehension, which recomputed
    values the loop had already produced).
    """
    if len(pooled) < 50:
        return False, -np.inf, -np.inf
    pooled_bundle = _fit_mechanism(pooled, target, parents)
    scores = []
    pooled_scores = []
    for env in env_pool:
        if len(env.data) < 40 or target not in env.data.columns:
            continue
        pooled_r2 = _predict_r2_bundle(pooled_bundle, env.data, target, parents)
        pooled_scores.append(pooled_r2)
        if _is_binary_y(env.data[target]) and _outcome_n_pos(env.data, target) < 2:
            continue
        local_r2 = _fit_predict_r2_holdout(env.data, target, parents)
        if np.isnan(local_r2):
            continue
        scores.append(pooled_r2 - local_r2)
    if not scores:
        return False, -np.inf, -np.inf
    gap_stat = float(np.median(scores))
    transfers = gap_stat >= -0.25
    sel = float(np.mean(pooled_scores)) if pooled_scores else -np.inf
    return transfers, sel, gap_stat


def bic_penalized_score(n: int, n_params: int, r2: float, y_var: float) -> float:
    # Gaussian BIC proxy from R²
    # ll ≈ -n/2 * log(SSR/n) ; SSR = (1-R²)*SST ; SST ≈ n * y_var
    ssr = max((1.0 - r2), 1e-8) * n * max(y_var, 1e-8)
    ll = -0.5 * n * np.log(ssr / n + 1e-12)
    return float(ll - 0.5 * n_params * np.log(n))


def enumerate_parent_sets(cands: list[str], max_size: int = 4) -> list[tuple[str, ...]]:
    sets: list[tuple[str, ...]] = [()]
    for k in range(1, min(max_size, len(cands)) + 1):
        sets.extend(combinations(cands, k))
    return sets


def stage3_select_parents(
    df: pd.DataFrame,
    env_pool: list[EnvWorld],
    stage2: dict[str, CandidateRecord],
    columns: list[str],
    record_audit: bool = False,
) -> dict[str, tuple[str, ...]]:
    chosen: dict[str, tuple[str, ...]] = {}
    global STAGE3_AUDIT
    if record_audit:
        STAGE3_AUDIT = {}
    naturals = [e for e in env_pool if e.tag == "natural"]
    downs = [e for e in env_pool if e.tag == "downsampled"]
    score_envs = naturals + downs[::2]
    for tgt in columns:
        if TIERS[tgt] == 1:
            chosen[tgt] = ()
            continue
        base = list(stage2[tgt].parents) if tgt in stage2 else tier_candidates(tgt, columns)
        for a, b in KNOWN_EDGES:
            if b == tgt and a not in base and a in columns:
                base.append(a)
        base = list(dict.fromkeys(base))
        y_var = float(df[tgt].var())
        n = len(df)
        # Score every candidate set by the cheap test (BIC over a CV R^2) first,
        # then only pay the expensive test (mechanism_transfers, ~len(score_envs)
        # model fits) on sets in descending score order, stopping at the first
        # one that passes. This picks exactly the same set the old code did —
        # it scored *every* set the same way, but ran mechanism_transfers on
        # all of them up front regardless of score, including ones that could
        # never win. Sorting first and short-circuiting means low-scoring sets
        # that fail (or that we never even need to check) don't pay for the
        # expensive test at all. Z=() is never filtered by the transfer test
        # in the original (guarded by `len(Z) > 0`), so it's exempted here too.
        all_sets = enumerate_parent_sets(base, max_size=min(3, len(base)))
        scored: list[tuple[float, tuple[str, ...]]] = []
        for Z in all_sets:
            mean, _se = _cv_r2(
                df[list(Z)].values.astype(float) if Z else np.zeros((n, 0)),
                df[tgt].values,
            )
            score = bic_penalized_score(n, len(Z) + 1, mean, y_var)
            if len(Z) == 0 and base:
                score -= 5.0
            scored.append((score, Z))
        scored.sort(key=lambda sZ: -sZ[0])  # stable: ties keep enumeration order, same as `score > best_score`

        best_set: tuple[str, ...] = ()
        best_score = -np.inf
        for score, Z in scored:
            if len(Z) == 0:
                passes = True
            else:
                ok, _sel, _gap = mechanism_transfers(score_envs, tgt, list(Z), df)
                passes = ok or all((p, tgt) in KNOWN_EDGES for p in Z)
            if passes:
                best_set, best_score = Z, score
                break  # highest-scoring passing set — nothing later can beat it
        kept = prune_parents_1se(df, tgt, list(best_set)) if best_set else []
        for a, b in KNOWN_EDGES:
            if b == tgt and a in columns and a not in kept:
                kept.append(a)
        chosen[tgt] = tuple(kept)
        if record_audit:
            if base:
                ok_b, _, gap_b = mechanism_transfers(score_envs, tgt, list(base), df)
            else:
                ok_b, gap_b = True, float("nan")
            if kept:
                ok_c, _, gap_c = mechanism_transfers(score_envs, tgt, kept, df)
            else:
                ok_c, gap_c = True, float("nan")
            STAGE3_AUDIT[tgt] = {
                "stage2_parents": list(base),
                "stage2_transfer": "keep" if ok_b else "skip",
                "stage2_gap_stat": float(gap_b),
                "chosen": list(kept),
                "chosen_transfer": "keep" if ok_c else "skip",
                "chosen_gap_stat": float(gap_c),
            }
    return chosen


# ---------------------------------------------------------------------------
# Stage 4 — direction + assemble
# ---------------------------------------------------------------------------
def orient_pair(df: pd.DataFrame, a: str, b: str) -> str:
    """Return 'a→b', 'b→a', or 'undirected' by CV fit + 1-SE."""
    if is_forbidden(a, b) and not is_forbidden(b, a):
        return f"{b}→{a}"
    if is_forbidden(b, a) and not is_forbidden(a, b):
        return f"{a}→{b}"
    if TIERS[a] < TIERS[b]:
        return f"{a}→{b}"
    if TIERS[b] < TIERS[a]:
        return f"{b}→{a}"
    m_ab, se_ab = _cv_r2(df[[a]].values.astype(float), df[b].values)
    m_ba, se_ba = _cv_r2(df[[b]].values.astype(float), df[a].values)
    if m_ab > m_ba + se_ba:
        return f"{a}→{b}"
    if m_ba > m_ab + se_ab:
        return f"{b}→{a}"
    return "undirected"


def assemble_graph(
    columns: list[str],
    parents: dict[str, tuple[str, ...]],
    df: pd.DataFrame,
    confounder_pairs: set[tuple[str, str]] | None = None,
) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(columns)
    undirected = []
    for tgt, pas in parents.items():
        for p in pas:
            if is_forbidden(p, tgt):
                continue
            orient = orient_pair(df, p, tgt)
            if orient == "undirected":
                undirected.append((p, tgt))
            elif orient == f"{p}→{tgt}":
                g.add_edge(p, tgt)
            elif orient == f"{tgt}→{p}":
                if not is_forbidden(tgt, p):
                    g.add_edge(tgt, p)
    # force known edges
    for a, b in KNOWN_EDGES:
        if a in columns and b in columns and not g.has_edge(a, b):
            g.add_edge(a, b)
    # break cycles greedily
    while True:
        try:
            cycle = nx.find_cycle(g, orientation="original")
        except nx.NetworkXNoCycle:
            break
        # remove weakest (last) edge on cycle
        u, v, _ = cycle[-1]
        g.remove_edge(u, v)
    g.graph["undirected"] = undirected
    g.graph["confounders"] = list(confounder_pairs or [])
    return g


def graph_to_adj(g: nx.DiGraph, columns: list[str]) -> np.ndarray:
    d = len(columns)
    idx = {c: i for i, c in enumerate(columns)}
    adj = np.zeros((d, d), dtype=int)
    for u, v in g.edges():
        adj[idx[u], idx[v]] = 1
    return adj


# ---------------------------------------------------------------------------
# Stage 5 — residual check + light refutation
# ---------------------------------------------------------------------------
def residual_independence_scores(df: pd.DataFrame, g: nx.DiGraph) -> dict[str, float]:
    """|corr(residual, parent)| mean — low is good."""
    out = {}
    for node in g.nodes():
        pas = list(g.predecessors(node))
        y = df[node].values.astype(float)
        if not pas:
            out[node] = 0.0
            continue
        model = LinearRegression().fit(df[pas].values.astype(float), y)
        resid = y - model.predict(df[pas].values.astype(float))
        corrs = [abs(np.corrcoef(resid, df[p].values.astype(float))[0, 1]) for p in pas]
        out[node] = float(np.nanmean(corrs))
    return out


def refute_edge(df: pd.DataFrame, src: str, tgt: str, seed: int = SEED) -> float:
    """Fraction of simple refutation checks passed (placebo / subset)."""
    rng = np.random.default_rng(seed)
    passed = 0
    total = 0
    # baseline effect
    m = LinearRegression().fit(df[[src]].values.astype(float), df[tgt].values.astype(float))
    base = abs(float(m.coef_[0]))
    # placebo: random covariate
    total += 1
    fake = rng.normal(size=len(df))
    m2 = LinearRegression().fit(fake.reshape(-1, 1), df[tgt].values.astype(float))
    if abs(float(m2.coef_[0])) < base:
        passed += 1
    # 30% subset stability
    total += 1
    idx = rng.choice(len(df), size=max(50, int(0.3 * len(df))), replace=False)
    sub = df.iloc[idx]
    m3 = LinearRegression().fit(sub[[src]].values.astype(float), sub[tgt].values.astype(float))
    if abs(abs(float(m3.coef_[0])) - base) / (base + 1e-8) < 0.5:
        passed += 1
    return passed / total


# ---------------------------------------------------------------------------
# Stage 6 — stability selection
# ---------------------------------------------------------------------------
def stage2_candidates_light(df: pd.DataFrame, columns: list[str]) -> dict[str, CandidateRecord]:
    """Stage-2 without FCI — used inside bootstrap loop for load."""
    records: dict[str, CandidateRecord] = {}
    for tgt in columns:
        if TIERS[tgt] == 1:
            records[tgt] = CandidateRecord(tgt, (), 0.0, 0.0)
            continue
        cands = tier_candidates(tgt, columns)
        pruned = prune_parents_1se(df, tgt, cands)
        mean, _ = _cv_r2(
            df[pruned].values.astype(float) if pruned else np.zeros((len(df), 0)),
            df[tgt].values,
        )
        records[tgt] = CandidateRecord(
            variable=tgt,
            parents=tuple(pruned),
            cv_score=mean,
            prior=prior_score(pruned, tgt),
            confounder_flags=[],
        )
    return records


def _import_gcm_fit():
    """Lazy import so bootstrap workers never load DoWhy."""
    try:
        import gcm_fit
        return gcm_fit
    except ImportError:
        from kpi_glint import gcm_fit
        return gcm_fit


def discover_once(
    df: pd.DataFrame,
    env_pool: list[EnvWorld] | None = None,
    seed: int = SEED,
    light: bool = False,
    use_gcm: bool = True,
) -> nx.DiGraph:
    """Run one discovery pass.

    use_gcm controls whether the (expensive, DoWhy-dependent) GCM parent-set
    refinement step runs. When False, gcm_fit is never imported and the
    cheap-screen (sklearn) parent sets from stage3_select_parents are used
    as-is to assemble the graph.
    """
    columns = list(df.columns)
    if env_pool is None:
        env_pool = build_env_pool(df, seed=seed)
    s2 = stage2_candidates_light(df, columns) if light else stage2_candidates(df, columns)
    if not light:
        global LAST_STAGE2
        LAST_STAGE2 = s2
    parents = stage3_select_parents(df, env_pool, s2, columns, record_audit=not light)
    if not light and use_gcm:
        global LAST_GCM_AUDIT
        gcm_fit = _import_gcm_fit()
        gcm_fit.require_dowhy()
        print("GCM refine on cheap-screen survivors", flush=True)
        parents, LAST_GCM_AUDIT = gcm_fit.refine_parent_sets(
            df, env_pool, parents, columns, root_tier=TIERS, seed=seed
        )
    elif not light and not use_gcm:
        print("GCM refine skipped (use_gcm=False) — using cheap-screen parent sets as-is", flush=True)
    conf = set()
    for rec in s2.values():
        for flag in rec.confounder_flags:
            a, b = flag.split("↔")
            conf.add(tuple(sorted((a, b))))
    return assemble_graph(columns, parents, df, conf)


# bootstrap_stability's B replicates are independent (each discover_once call
# only reads df/env_pool, never mutates shared state) -- embarrassingly
# parallel across processes. A worker process started via multiprocessing
# does not inherit bind_dataset()'s already-mutated module globals (a spawned
# process re-imports this module fresh, back at its placeholder defaults), so
# _bootstrap_globals_snapshot/_bootstrap_worker_init explicitly copy exactly
# the globals bind_dataset() sets -- no more, no less -- into each worker.
def _bootstrap_globals_snapshot() -> dict:
    return dict(
        TIERS=TIERS,
        KNOWN_EDGES=KNOWN_EDGES,
        COLLINEAR_PAIRS=COLLINEAR_PAIRS,
        POST_TREATMENT_FORBIDDEN=POST_TREATMENT_FORBIDDEN,
        FORBIDDEN_EDGES=FORBIDDEN_EDGES,
        B=B,
        TARGET_BINARY=globals().get("TARGET_BINARY"),
        MIN_POSITIVES=MIN_POSITIVES,
        M_DOWNSAMPLE=M_DOWNSAMPLE,
        F_KEEP=F_KEEP,
        MIN_CELL=MIN_CELL,
        SEED=SEED,
        columns=globals().get("columns"),
        n_vars=globals().get("n_vars"),
    )


_WORKER_DF: pd.DataFrame | None = None


def _bootstrap_worker_init(state: dict, df: pd.DataFrame) -> None:
    global TIERS, KNOWN_EDGES, COLLINEAR_PAIRS, POST_TREATMENT_FORBIDDEN, FORBIDDEN_EDGES
    global B, TARGET_BINARY, MIN_POSITIVES, M_DOWNSAMPLE, F_KEEP, MIN_CELL, SEED
    global columns, n_vars, _WORKER_DF
    TIERS = state["TIERS"]
    KNOWN_EDGES = state["KNOWN_EDGES"]
    COLLINEAR_PAIRS = state["COLLINEAR_PAIRS"]
    POST_TREATMENT_FORBIDDEN = state["POST_TREATMENT_FORBIDDEN"]
    FORBIDDEN_EDGES = state["FORBIDDEN_EDGES"]
    B = state["B"]
    TARGET_BINARY = state["TARGET_BINARY"]
    MIN_POSITIVES = state["MIN_POSITIVES"]
    M_DOWNSAMPLE = state["M_DOWNSAMPLE"]
    F_KEEP = state["F_KEEP"]
    MIN_CELL = state["MIN_CELL"]
    SEED = state["SEED"]
    columns = state["columns"]
    n_vars = state["n_vars"]
    _WORKER_DF = df


def _bootstrap_worker_run(args: tuple[int, int, np.ndarray]) -> tuple[int, list[tuple[str, str]], int, float]:
    b, seed_b, boot_idx = args
    t0 = time.time()
    boot = _WORKER_DF.iloc[boot_idx].reset_index(drop=True)
    g = discover_once(boot, env_pool=None, seed=seed_b, light=True)
    edges = list(g.edges())
    n_pos = _outcome_n_pos(boot)
    return b, edges, n_pos, time.time() - t0


def bootstrap_stability(
    df: pd.DataFrame,
    B: int = B,
    seed: int = SEED,
    cache_dir: Path | None = None,
    rebuild_envs_each_boot: bool = False,
    n_jobs: int = 1,
) -> tuple[dict[tuple[str, str], float], list[nx.DiGraph]]:
    """Selection frequency over B bootstraps. Caches edge lists to cache_dir.

    n_jobs: 1 runs sequentially in this process. Any other value runs the B
    replicates across worker processes instead (-1 or 0 uses every available
    core). This only changes WHICH process runs discover_once() for a given
    replicate, never what data it gets: every replicate's resample indices
    are drawn up front, in order, from the same single rng the original
    sequential loop used, so replicate b sees the exact same rows whether
    n_jobs is 1 or 8. Workers always call discover_once(..., light=True)
    (sklearn only — no GCM).
    """
    rng = np.random.default_rng(seed)
    columns = list(df.columns)
    # build env pool once on full data (diagram uses fixed worlds; boots resample observations)
    base_envs = build_env_pool(df, seed=seed)
    freq: dict[tuple[str, str], int] = {}
    graphs_by_b: dict[int, nx.DiGraph] = {}
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _record(b: int, edges: list[tuple[str, str]], n_pos: int, elapsed: float) -> None:
        g = nx.DiGraph()
        g.add_nodes_from(columns)
        g.add_edges_from(edges)
        graphs_by_b[b] = g
        for e in edges:
            freq[e] = freq.get(e, 0) + 1
        cache_path = cache_dir / f"boot_{b:03d}.json" if cache_dir else None
        if cache_path:
            cache_path.write_text(json.dumps(edges))
        print(f"  bootstrap {b + 1}/{B}: |E|={len(edges)} n_pos={n_pos} ({elapsed:.1f}s)", flush=True)

    # Same control flow as the original loop: draw idx (consuming rng) only
    # for replicates that aren't already cached, in b order -- so the rng
    # sees exactly the same sequence of .choice() calls either way.
    boot_indices: dict[int, np.ndarray] = {}
    pending: list[int] = []
    for b in range(B):
        cache_path = cache_dir / f"boot_{b:03d}.json" if cache_dir else None
        if cache_path and cache_path.exists():
            edges = [tuple(e) for e in json.loads(cache_path.read_text())]
            g = nx.DiGraph()
            g.add_nodes_from(columns)
            g.add_edges_from(edges)
            graphs_by_b[b] = g
            for e in edges:
                freq[e] = freq.get(e, 0) + 1
            continue
        boot_indices[b] = rng.choice(len(df), size=len(df), replace=True)
        pending.append(b)

    if n_jobs == 1 or not pending:
        for b in pending:
            boot = df.iloc[boot_indices[b]].reset_index(drop=True)
            t0 = time.time()
            if rebuild_envs_each_boot:
                build_env_pool(boot, seed=seed + b + 1)  # kept for parity with the original call; unused, as before
            g = discover_once(boot, env_pool=None, seed=seed + b, light=True)
            edges = list(g.edges())
            _record(b, edges, _outcome_n_pos(boot), time.time() - t0)
    else:
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed

        workers = (os.cpu_count() or 1) if n_jobs <= 0 else n_jobs
        workers = max(1, min(workers, len(pending)))
        state = _bootstrap_globals_snapshot()
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_bootstrap_worker_init, initargs=(state, df)
        ) as ex:
            futures = [ex.submit(_bootstrap_worker_run, (b, seed + b, boot_indices[b])) for b in pending]
            for fut in as_completed(futures):
                b, edges, n_pos, elapsed = fut.result()
                _record(b, edges, n_pos, elapsed)

    graphs = [graphs_by_b[b] for b in range(B)]
    freqs = {e: c / B for e, c in freq.items()}
    return freqs, graphs


def meinhausen_buhlmann_threshold(freqs: dict[tuple[str, str], float], avg_edges: float, ev_max: float = EV_MAX) -> float:
    """Derive keep-threshold π_thr so E[V] ≤ ev_max (simplified MB bound).

    Uses p = n_vars (the actual number of discovery columns), not the
    Sachs diagram's hardcoded p=11 — see build_natural_strata's docstring
    above for why only this version (previously nested in bind_dataset)
    is kept.
    """
    p = max(n_vars, 2)
    binom = p * (p - 1)
    q = max(avg_edges, 1.0)
    thr = 0.5 * (1.0 + (q ** 2) / (ev_max * binom))
    return float(min(0.95, max(0.5, thr)))


def edge_strengths(df: pd.DataFrame, g: nx.DiGraph) -> dict[tuple[str, str], float]:
    strengths = {}
    for node in g.nodes():
        pas = list(g.predecessors(node))
        if not pas:
            continue
        model = LinearRegression().fit(df[pas].values.astype(float), df[node].values.astype(float))
        coefs = np.abs(model.coef_)
        s = coefs / (coefs.sum() + 1e-12)
        for p, w in zip(pas, s):
            strengths[(p, node)] = float(w)
    return strengths


# ---------------------------------------------------------------------------
# Stage 7 — experiment ranking
# ---------------------------------------------------------------------------
def rank_experiments(
    undecided: list[tuple[str, str]],
    df: pd.DataFrame,
    g: nx.DiGraph,
) -> list[dict[str, Any]]:
    ranks = []
    for a, b in undecided:
        # expected info proxy: variance of a in high-variance stratum of b's residual
        info = float(df[a].var() * (1.0 + df[b].var()))
        ranks.append(
            {
                "intervene_on": a,
                "resolves_edge": f"{a}—{b}",
                "expected_info": info,
                "suggested": f"Perturb {a} and re-measure {b} transfer",
            }
        )
    ranks.sort(key=lambda r: -r["expected_info"])
    return ranks


# ---------------------------------------------------------------------------
# Driver: causal_graph(weakest_pillar_output_path, output_path)
#
# This used to be notebook 03's remaining cells (candidate-set setup,
# bind_dataset, plotting helpers, discover/bootstrap/fallback/save). The
# only change from the notebook version: it no longer computes the
# weak-pillar-filter decision itself (that moved to weakest_pillar.py) and
# no longer reads dataset/kpi_list.py or kpi_meta.csv directly — pillar and
# parent info now come from weakest_pillar's selected_kpis.csv. Everything
# else (build_matrix, bind_dataset, run_discovery, the no-parents-found
# fallback, the sink audit, the plots) is unchanged.
# ---------------------------------------------------------------------------
import os as _os
_os.environ.setdefault("MPLBACKEND", "Agg")

import time
import matplotlib.pyplot as plt


def _default_artifacts_path(output_path: Path) -> Path:
    parts = list(output_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "output":
            parts[i] = "artifacts"
            return Path(*parts)
    return output_path.parent / "artifacts" / output_path.name


def _write_profile_log(profiler, log_path: Path, stage_name: str, elapsed: float) -> None:
    """Write a full per-function cProfile report (sorted by cumulative time,
    then by self/tottime) plus the raw .pstats dump next to it, so it can be
    re-sorted or visualized later (snakeviz, gprof2dot, ...) without
    re-running the pipeline. Pure reporting -- does not affect the run.
    """
    import io
    import pstats

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pstats_path = log_path.with_suffix(".pstats")
    profiler.dump_stats(str(pstats_path))

    buf = io.StringIO()
    buf.write(f"=== {stage_name} profile ===\n")
    buf.write(f"wall time: {elapsed:.2f}s\n\n")
    buf.write("--- top 40 by cumulative time (this function + everything it calls) ---\n")
    pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("cumulative").print_stats(40)
    buf.write("\n--- top 40 by self time (time spent in this function alone) ---\n")
    pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("tottime").print_stats(40)
    log_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"{stage_name} profile -> {log_path}  (raw stats -> {pstats_path})")


def causal_graph(
    weakest_pillar_output_path,
    output_path,
    artifacts_path=None,
    profile: bool = False,
    profile_log: str | None = None,
    n_jobs: int = -1,
    use_gcm: bool = True,
) -> dict:
    """Thin entry point around _causal_graph_impl (notebook 03's discovery
    logic, completely unchanged below). Set profile=True to run the same
    call under cProfile and get a full per-function timing report written
    next to the other artifacts -- this only wraps the call, it does not
    change it. n_jobs controls how many worker processes run the bootstrap
    replicates: -1 (default) uses every available core; 1 is sequential.

    use_gcm (default True): when True, the DoWhy GCM refinement/scoring step
    runs (parent-set refinement via gcm_fit.refine_parent_sets, arrow-strength
    scoring, and graph falsification). When False, gcm_fit/DoWhy is never
    imported -- the graph is assembled directly from the cheap sklearn/env-
    based screen (stage2/stage3), and edge strengths come from a plain
    linear-regression |coef| fallback instead of DoWhy's arrow_strength.
    """
    if not profile:
        return _causal_graph_impl(weakest_pillar_output_path, output_path, artifacts_path, n_jobs=n_jobs, use_gcm=use_gcm)
    import cProfile

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    try:
        result = _causal_graph_impl(weakest_pillar_output_path, output_path, artifacts_path, n_jobs=n_jobs, use_gcm=use_gcm)
    finally:
        profiler.disable()
    elapsed = time.time() - t0
    log_path = Path(profile_log) if profile_log else result["artifacts"] / "profile_causal_graph.log"
    _write_profile_log(profiler, log_path, "causal_graph", elapsed)
    return result


def _causal_graph_impl(weakest_pillar_output_path, output_path, artifacts_path=None, n_jobs: int = -1, use_gcm: bool = True) -> dict:
    """Run notebook 03's discovery logic end to end.

    weakest_pillar_output_path: folder holding weakest_pillar()'s 3 main
                  outputs (selected_kpis.csv, panel.csv, run.json).
    output_path:  folder to write edges.csv, the one main output.
    artifacts_path: folder for everything else (edges.json, metrics.json,
                  graph.png, the discovery-matrix snapshot, bootstrap cache);
                  defaults to swapping the last "output" path segment for
                  "artifacts".
    """
    WP_OUTPUT = Path(weakest_pillar_output_path)
    OUTPUT_MAIN = Path(output_path)
    ARTIFACTS = Path(artifacts_path) if artifacts_path else _default_artifacts_path(OUTPUT_MAIN)
    OUTPUT = ARTIFACTS  # historical name from the notebook; every `OUTPUT / ...` write below is an artifact
    FIGURES = ARTIFACTS / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MAIN.mkdir(parents=True, exist_ok=True)

    run = json.loads((WP_OUTPUT / "run.json").read_text())
    panel = pd.read_csv(WP_OUTPUT / "panel.csv")
    sel_df = pd.read_csv(WP_OUTPUT / "selected_kpis.csv")
    print("panel", panel.shape, "weak pillars", run.get("pillars"))
    print("selected", sel_df["kpi_id"].tolist())

    # Multi-target mode: TARGET may be None when no single target was declared
    # in kpis.jsonld.  In that case all rec_targets are outcome nodes (no
    # single tier-4 sink).
    TARGET = run.get("target")  # None in multi-target mode
    if TARGET and TARGET not in panel.columns:
        raise ValueError(
            f"target {TARGET!r} is not on the panel. "
            "Re-run weakest_pillar with this KPI added to the catalog."
        )
    pillars = list(run.get("pillars") or [])
    inbound = [c for c in run.get("inbound") or [] if c in panel.columns]
    DROP = set(run.get("drop_from_discovery") or run.get("drop_from_influence") or [])
    pillar_of = dict(zip(sel_df["kpi_id"], sel_df["pillar"]))
    parent_of = dict(zip(sel_df["kpi_id"], sel_df["parent"]))
    SELECTED = [k for k in sel_df["kpi_id"].tolist() if k in panel.columns]

    # rec_targets: all recommended KPIs (from weakest_pillar's multi-target loop).
    # Under KG-based tiering these are the outcome nodes — no single tier-4 sink.
    REC_TARGETS: set[str] = set(run.get("rec_targets") or ([TARGET] if TARGET else []))

    # rec_target flag column may or may not exist in older selected_kpis.csv files
    if "rec_target" in sel_df.columns:
        _rec_from_csv = set(sel_df.loc[sel_df["rec_target"].fillna(False).astype(bool), "kpi_id"])
        REC_TARGETS = REC_TARGETS | _rec_from_csv

    kpi_cols = [
        k for k in sel_df.loc[sel_df["primary_candidate"], "kpi_id"].tolist()
        if k in panel.columns
    ]
    print("discovery candidate KPIs (weakest_pillar's primary_candidate list):", kpi_cols)
    print("rec_targets (tier-3 near-outcome nodes):", sorted(REC_TARGETS))

    def class_coverage_ok(df: pd.DataFrame, target: str, col: str, thresh: float = 0.5) -> bool:
        y = pd.to_numeric(df[target], errors="coerce")
        s = pd.to_numeric(df[col], errors="coerce")
        y_ok = y.dropna()
        if y_ok.nunique() > 2:
            return True
        for cls in y_ok.unique():
            mask = y == cls
            if mask.any() and float(s.loc[mask].isna().mean()) > thresh:
                return False
        return True


    # --- KG-depth context-variable map ---
    # Derived context variables (e.g. ORDER_COMPLEXITY = N_ITEMS * N_SELLERS)
    # get Tier 1 (one level deeper than root context vars at Tier 0).
    # run.json carries "context_var_parents" — a dict mapping each derived
    # context var to the list of context vars it depends on.  When absent,
    # fall back to an empty dict (all context vars become Tier 0).
    _ctx_parents: dict[str, list[str]] = run.get("context_var_parents") or {}

    def build_matrix(cand_kpis: list[str], verbose: bool = True) -> tuple[list[str], list[str], pd.DataFrame, dict[str, int]]:
        """Turn a candidate KPI list into the (core_kpis, cols, disc matrix, tiers) that
        GLINT discovery runs on.  Reused as-is for the weak-pillar fallback below.

        KG-based tier scheme (derived from kpis.jsonld depth):
          Tier 0: root context variables (no context-var parents)
          Tier 1: derived context variables (depend on other context vars)
          Tier 2: root KPIs (no parent KPI in the KG)
          Tier 3: child KPIs (have a parent KPI in the KG, e.g. IS_ON_TIME)
        No single Tier-4 TARGET sink — all rec_targets are outcome nodes.
        """
        core = [k for k in cand_kpis if k != TARGET and k not in DROP]
        dropped_recode = [k for k in cand_kpis if k != TARGET and k in DROP]
        if dropped_recode and verbose:
            print("dropped recodes", dropped_recode)

        # Class-coverage check: skip if no single TARGET (multi-target mode)
        if TARGET:
            dropped_cov = [k for k in core if not class_coverage_ok(panel, TARGET, k)]
        else:
            dropped_cov = []
        if dropped_cov and verbose:
            print("dropped (missing on a target class):", dropped_cov)
        core = [k for k in core if k not in dropped_cov]

        c_cols: list[str] = []
        extra = [TARGET] if TARGET else []
        for c in core + inbound + extra:
            if c in panel.columns and c not in c_cols and c not in DROP:
                c_cols.append(c)

        w = panel[c_cols].copy()
        for c in w.columns:
            if w[c].dtype == object or str(w[c].dtype) in ("string", "category"):
                w[c] = w[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
            else:
                w[c] = pd.to_numeric(w[c], errors="coerce")

        bins = [c for c in w.columns if w[c].nunique(dropna=True) <= 2]
        rider_drop = []
        for c in list(w.columns):
            if c == TARGET:
                continue
            if any(not class_coverage_ok(w, b, c) for b in bins):
                rider_drop.append(c)
        if rider_drop:
            if verbose:
                print("dropped (binary-rider class coverage):", rider_drop)
            w = w.drop(columns=rider_drop)
            c_cols = [c for c in c_cols if c not in rider_drop]
            core = [k for k in core if k not in rider_drop]

        d = w.dropna().reset_index(drop=True)
        if verbose:
            print("discovery matrix", d.shape, "columns", list(d.columns))
        if TARGET and TARGET in d.columns and verbose:
            panel_pos = int(pd.to_numeric(panel[TARGET], errors="coerce").eq(1).sum()) if TARGET in panel.columns else 0
            disc_pos = int(pd.to_numeric(d[TARGET], errors="coerce").eq(1).sum())
            print(
                f"TARGET {TARGET}: n={len(d)} n_pos={disc_pos} "
                f"prevalence={disc_pos / max(len(d), 1):.4f} "
                f"panel_pos={panel_pos} lost_to_dropna={panel_pos - disc_pos}"
            )
            print("target value counts:\n", d[TARGET].value_counts(dropna=False).head().to_string())

        # --- KG-based tier assignment ---
        t: dict[str, int] = {}
        in_set = set(inbound)
        graph_set = set(d.columns)
        for c in d.columns:
            if c in in_set:
                # Context variable: Tier 0 (root) or Tier 1 (derived from other ctx vars)
                if _ctx_parents.get(c):
                    t[c] = 1  # Tier 1: derived context var (e.g. ORDER_COMPLEXITY)
                else:
                    t[c] = 0  # Tier 0: root context var (e.g. TOTAL_PRICE, N_ITEMS)
            elif parent_of.get(c) and parent_of[c] in graph_set and parent_of[c] != c:
                # Tier 3: child KPI — has a parent KPI in the KG
                # (e.g. IS_ON_TIME child of DELAY_DAYS, IS_NEGATIVE_REVIEW child of REVIEW_SCORE)
                t[c] = 3
            else:
                # Tier 2: root KPI — no parent KPI in the KG
                # (e.g. DELAY_DAYS, IS_CANCELED, REVIEW_SCORE)
                t[c] = 2

        pd.DataFrame({"variable": list(d.columns), "tier": [t[c] for c in d.columns]}).to_csv(
            ARTIFACTS / "schema.csv", index=False
        )
        d.to_csv(ARTIFACTS / "discovery_panel.csv", index=False)
        if verbose:
            print(pd.Series(t).sort_values())
        return core, c_cols, d, t


    # The full selected set (pre weak-pillar filter) — kept around as the fallback
    # candidate pool below: if the weak-pillar pool turns out to have no parents
    # for TARGET at all, we retry discovery against this instead.
    kpi_cols_fallback = list(SELECTED)

    core_kpis, cols, disc, tiers = build_matrix(kpi_cols)


    # Bind this run's tiers/knobs (rebinds the empty engine-module placeholders).
    # Wrapped as a function so cell 12 can rebind against the fallback candidate
    # set (kpi_cols_fallback) if the weak-pillar pool turns out to have no
    # parents for TARGET. build_natural_strata / meinhausen_buhlmann_threshold
    # below read these as globals, so re-running this function is enough — the
    # functions themselves don't need to be redefined.
    def bind_dataset(d: pd.DataFrame, t: dict[str, int]) -> None:
        global TIERS, KNOWN_EDGES, COLLINEAR_PAIRS, POST_TREATMENT_FORBIDDEN, FORBIDDEN_EDGES
        global B, TARGET, TARGET_BINARY, MIN_POSITIVES, M_DOWNSAMPLE, F_KEEP, MIN_CELL, SEED
        global columns, n_vars, REC_TARGETS_SET

        TIERS = dict(t)
        KNOWN_EDGES = []
        REC_TARGETS_SET = set(REC_TARGETS)

        # Read collinear pairs from run.json (set by weakest_pillar from kpis.jsonld).
        # run["collinear"] is a list of [a, b] pairs written by weakest_pillar when
        # it serialises spec.collinear; fall back to an empty set if not present.
        COLLINEAR_PAIRS = {tuple(pair) for pair in (run.get("collinear") or [])}

        # Post-outcome pillars: any pillar whose stage == "post_outcome" in kpis.jsonld.
        # Under KG-based tiering, post_outcome KPIs are forbidden from being causes
        # of ANY non-post-outcome KPI (no tier threshold — the old tier >= 3 check is
        # removed because the KG-based scheme has no single tier-4 sink).
        post_outcome_pillars: set[str] = set(run.get("post_outcome_pillars") or [])
        post_outcome_kpis = {k for k, p in pillar_of.items() if p in post_outcome_pillars}
        POST_TREATMENT_FORBIDDEN = set()
        if post_outcome_pillars:
            # All non-post-outcome KPIs (tier >= 2) are operational nodes that
            # post_outcome KPIs must not point at.
            operational_kpis = {
                c for c in d.columns
                if TIERS.get(c, 2) >= 2
                and pillar_of.get(c) not in post_outcome_pillars
            }
            POST_TREATMENT_FORBIDDEN = {
                (r, op)
                for r in post_outcome_kpis if r in d.columns
                for op in operational_kpis
            }

        FORBIDDEN_EDGES = [(c, p) for p, t_p in TIERS.items() for c, t_c in TIERS.items() if t_c > t_p]
        B = int(run.get("bootstrap_B") or 8)
        # In multi-target mode TARGET is None; _outcome_n_pos() must handle this.
        TARGET = run.get("target")
        TARGET_BINARY = bool(
            TARGET
            and TARGET in d.columns
            and int(pd.to_numeric(d[TARGET], errors="coerce").nunique(dropna=True)) == 2
        )
        MIN_POSITIVES = 20 if TARGET_BINARY else 0
        M_DOWNSAMPLE = 10
        F_KEEP = 0.70
        MIN_CELL = max(30, min(120, len(d) // 100))
        SEED = 0
        columns = list(d.columns)
        n_vars = len(columns)


    bind_dataset(disc, tiers)


    # build_natural_strata / meinhausen_buhlmann_threshold read TIERS, columns,
    # MIN_CELL, MIN_POSITIVES, n_vars as globals (set just above) — no need to
    # redefine them here; the single module-level versions pick up the new
    # values automatically on the next call.
    print("tiers bound", {k: TIERS[k] for k in columns})
    print("B", B, "min_cell", MIN_CELL, "min_positives", MIN_POSITIVES, "target_binary", TARGET_BINARY)
    print("POST_TREATMENT_FORBIDDEN", sorted(POST_TREATMENT_FORBIDDEN))


    # GLINT graph plots — GT overlay when truth exists, else plain discovered structure.
    # (Path, Any already imported at module level above.)

    from matplotlib.lines import Line2D

    COLORS = {
        "true": "#2CA02C",
        "spur": "#D62728",
        "rev": "#FF7F0E",
        "miss": "#7F7F7F",
        "edge": "#2B6CB0",
        "node": "#E8F1FB",
        "node_iso": "#F0F0F0",
        "node_edge": "#2C3E50",
    }


    def _label(v: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return v
        return labels.get(v, v)


    def _pairs_to_edges(pairs: list) -> list[tuple[str, str]]:
        out = []
        for p in pairs:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((str(p[0]), str(p[1])))
        return out


    def draw_overlay(
        ax,
        score: dict[str, Any],
        title: str,
        all_vars: list[str],
        labels: dict[str, str] | None = None,
        node_pos: dict[str, tuple[float, float]] | None = None,
    ):
        G = nx.DiGraph()
        G.add_nodes_from(all_vars)
        if node_pos is None:
            pos = nx.circular_layout(G)
        else:
            pos = {v: node_pos[v] for v in all_vars if v in node_pos}
            missing = [v for v in all_vars if v not in pos]
            if missing:
                extra = nx.circular_layout(missing)
                pos.update(extra)

        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color="#E8F1FB", node_size=900, edgecolors="#333333", linewidths=1.0
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels={v: _label(v, labels) for v in all_vars},
            ax=ax,
            font_size=8,
            font_weight="bold",
        )

        def _draw(pairs, color, style="solid", alpha=0.9, width=2.0):
            edges = _pairs_to_edges(pairs)
            if not edges:
                return
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                ax=ax,
                edge_color=color,
                style=style,
                width=width,
                alpha=alpha,
                arrows=True,
                arrowsize=14,
                connectionstyle="arc3,rad=0.08",
                min_source_margin=12,
                min_target_margin=12,
            )

        _draw(score["etrue"], COLORS["true"])
        _draw(score["espur"], COLORS["spur"])
        _draw(score["efals"], COLORS["rev"])
        _draw(score["emiss"], COLORS["miss"], style="dashed", alpha=0.7, width=1.5)

        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        ax.margins(0.15)


    def save_colored_graph(
        metrics: dict,
        columns: list[str],
        labels: dict[str, str],
        out_path: Path,
        title: str,
    ):
        """Green=true, red=spurious, orange=reversed, dashed gray=missing (bnlearn / GT runs)."""
        d = len(columns)
        all_vars = [f"X{i + 1}" for i in range(d)]
        xlabels = {f"X{i + 1}": labels.get(columns[i], columns[i]) for i in range(d)}
        score = {
            "etrue": metrics.get("true_edges") or [],
            "espur": metrics.get("spur_edges") or [],
            "efals": metrics.get("fals_edges") or [],
            "emiss": metrics.get("miss_edges") or [],
        }
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        draw_overlay(ax, score, title, all_vars, labels=xlabels)
        legend = [
            Line2D([0], [0], color=COLORS["true"], lw=2, label=f"true ({metrics['etrue']})"),
            Line2D([0], [0], color=COLORS["spur"], lw=2, label=f"spurious ({metrics['espur']})"),
            Line2D([0], [0], color=COLORS["rev"], lw=2, label=f"reversed ({metrics['efals']})"),
            Line2D([0], [0], color=COLORS["miss"], lw=2, ls="--", label=f"missing ({metrics['emiss']})"),
        ]
        ax.legend(
            handles=legend,
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.08),
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


    def _tier_tree_layout(
        columns: list[str],
        tiers: dict[str, int],
        edges: list[tuple[str, str]],
    ) -> dict[str, tuple[float, float]]:
        """Left→right parent→child columns by tier; isolates sit in a bottom band.

        Within each tier, order nodes with a barycenter pass to cut crossings.
        """
        G = nx.DiGraph()
        G.add_nodes_from(columns)
        G.add_edges_from(edges)
        active = {n for n in columns if G.degree(n) > 0}
        isolates = [n for n in columns if n not in active]

        by_tier: dict[int, list[str]] = {}
        for n in active:
            by_tier.setdefault(int(tiers.get(n, 1)), []).append(n)
        tier_ids = sorted(by_tier.keys())

        # Seed order: out-degree then name
        order: dict[int, list[str]] = {
            t: sorted(by_tier[t], key=lambda n: (-G.out_degree(n), -G.in_degree(n), n))
            for t in tier_ids
        }

        def _bary(nodes: list[str], nbrs_fn) -> list[str]:
            rank = {n: i for i, n in enumerate(nodes)}

            def key(n: str):
                nbrs = [rank[u] for u in nbrs_fn(n) if u in rank]
                return (sum(nbrs) / len(nbrs) if nbrs else len(nodes) / 2.0, n)

            return sorted(nodes, key=key)

        # Forward + backward barycenter sweeps
        for _ in range(4):
            for i in range(1, len(tier_ids)):
                prev = {n: j for j, n in enumerate(order[tier_ids[i - 1]])}
                order[tier_ids[i]] = _bary(
                    order[tier_ids[i]],
                    lambda n, p=prev: [u for u in G.predecessors(n) if u in p],
                )
            for i in range(len(tier_ids) - 2, -1, -1):
                nxt = {n: j for j, n in enumerate(order[tier_ids[i + 1]])}
                order[tier_ids[i]] = _bary(
                    order[tier_ids[i]],
                    lambda n, p=nxt: [v for v in G.successors(n) if v in p],
                )

        x_gap = 3.6
        y_gap = 1.55
        pos: dict[str, tuple[float, float]] = {}
        for xi, t in enumerate(tier_ids):
            nodes = order[t]
            n = len(nodes)
            ys = np.linspace((n - 1) / 2.0, -(n - 1) / 2.0, n) if n > 1 else [0.0]
            for y, node in zip(ys, nodes):
                pos[node] = (xi * x_gap, float(y) * y_gap)

        if isolates:
            iso_y = min((p[1] for p in pos.values()), default=0.0) - 2.4
            n = len(isolates)
            xs = np.linspace(0.0, max(len(tier_ids) - 1, 1) * x_gap, n) if n > 1 else [0.0]
            for x, node in zip(xs, sorted(isolates)):
                pos[node] = (float(x), iso_y)

        return pos


    def save_structure_graph(
        out_path: Path,
        columns: list[str],
        edges: list[tuple[str, str]],
        *,
        tiers: dict[str, int] | None = None,
        labels: dict[str, str] | None = None,
        strengths: dict[tuple[str, str], float] | None = None,
        truth: np.ndarray | None = None,
        title: str = "",
        subtitle: str | None = None,
    ):
        """Hierarchical parent→child plot with strength labels.

        - No / empty truth → single-color discovered edges (no true/spur/rev legend).
        - Non-empty truth → GT overlay colors + missing dashed.
        """
        labels = labels or {c: c for c in columns}
        strengths = strengths or {}
        tiers = tiers or {c: 1 for c in columns}
        idx = {c: i for i, c in enumerate(columns)}

        has_gt = truth is not None and int(np.sum(truth != 0)) > 0

        # Ensure all edge endpoints are included as nodes (sink audit may
        # inject edges referencing KPIs outside the discovery matrix).
        col_set = set(columns)
        extra_nodes = [n for e in edges for n in e if n not in col_set]
        if extra_nodes:
            extra_nodes = list(dict.fromkeys(extra_nodes))  # dedupe, keep order
            columns = list(columns) + extra_nodes
            for n in extra_nodes:
                tiers.setdefault(n, 2)
                labels.setdefault(n, n)

        G = nx.DiGraph()
        G.add_nodes_from(columns)
        G.add_edges_from(edges)
        pos = _tier_tree_layout(columns, tiers, edges)

        active = [n for n in columns if G.degree(n) > 0]
        isolates = [n for n in columns if n not in active]

        n_tiers = max((tiers.get(c, 1) for c in active), default=1) - min(
            (tiers.get(c, 1) for c in active), default=1
        ) + 1
        fig_w = max(10.0, 2.8 * max(n_tiers, 2) + 2)
        fig_h = max(7.0, 0.55 * max(len(active), 4) + (2.5 if isolates else 0.8))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if active:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=active,
                ax=ax,
                node_color=COLORS["node"],
                node_size=1600,
                edgecolors=COLORS["node_edge"],
                linewidths=1.2,
            )
        if isolates:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=isolates,
                ax=ax,
                node_color=COLORS["node_iso"],
                node_size=1100,
                edgecolors="#AAAAAA",
                linewidths=1.0,
                alpha=0.85,
            )
        nx.draw_networkx_labels(
            G,
            pos,
            labels={n: labels.get(n, n) for n in columns},
            ax=ax,
            font_size=7.5,
            font_weight="bold",
        )

        edge_labels: dict[tuple[str, str], str] = {}
        for a, b in edges:
            s = strengths.get((a, b), strengths.get((str(a), str(b)), None))
            if s is None:
                continue
            edge_labels[(a, b)] = f"{float(s):.2f}"

        def _edge_meta(a: str, b: str) -> tuple[str, float]:
            """Return (color, width) for an edge."""
            w = 1.2 + 3.5 * float(strengths.get((a, b), 0.35))
            if not has_gt:
                return COLORS["edge"], w
            ia, ib = idx[a], idx[b]
            if truth[ia, ib] != 0:
                return COLORS["true"], w
            if truth[ib, ia] != 0:
                return COLORS["rev"], w
            return COLORS["spur"], w

        # Slight fan of curvatures so parallel parents don't sit on one line
        for i, (a, b) in enumerate(edges):
            color, width = _edge_meta(a, b)
            rad = 0.06 + 0.04 * ((i % 5) - 2)
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(a, b)],
                ax=ax,
                edge_color=color,
                width=width,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=14,
                connectionstyle=f"arc3,rad={rad}",
                min_source_margin=18,
                min_target_margin=18,
                alpha=0.9,
            )

        if has_gt:
            miss = []
            for i, u in enumerate(columns):
                for j, v in enumerate(columns):
                    if truth[i, j] != 0 and (u, v) not in set(edges):
                        miss.append((u, v))
            if miss:
                # add missing endpoints already in G
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=miss,
                    ax=ax,
                    edge_color=COLORS["miss"],
                    style="dashed",
                    width=1.3,
                    arrows=True,
                    arrowsize=12,
                    connectionstyle="arc3,rad=0.05",
                    alpha=0.65,
                    min_source_margin=14,
                    min_target_margin=14,
                )

        if edge_labels:
            # Manual mid-edge labels with staggered offsets (nx edge_labels collide heavily)
            for i, ((a, b), txt) in enumerate(edge_labels.items()):
                if a not in pos or b not in pos:
                    continue
                x0, y0 = pos[a]
                x1, y1 = pos[b]
                t = 0.45 + 0.08 * ((i % 3) - 1)
                mx = x0 + t * (x1 - x0)
                my = y0 + t * (y1 - y0)
                # perpendicular nudge
                dx, dy = x1 - x0, y1 - y0
                norm = (dx * dx + dy * dy) ** 0.5 + 1e-9
                ox, oy = -dy / norm, dx / norm
                nudge = 0.18 * (1 if i % 2 == 0 else -1)
                ax.text(
                    mx + nudge * ox,
                    my + nudge * oy,
                    txt,
                    fontsize=7,
                    color="#1A365D",
                    ha="center",
                    va="center",
                    zorder=5,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#CBD5E0", "alpha": 0.92},
                )

        full_title = title
        if subtitle:
            full_title = f"{title}\n{subtitle}"
        ax.set_title(full_title, fontsize=11, pad=12)
        ax.set_axis_off()
        ax.margins(0.08)

        if has_gt:
            legend = [
                Line2D([0], [0], color=COLORS["true"], lw=2, label="true"),
                Line2D([0], [0], color=COLORS["spur"], lw=2, label="spurious"),
                Line2D([0], [0], color=COLORS["rev"], lw=2, label="reversed"),
                Line2D([0], [0], color=COLORS["miss"], lw=2, ls="--", label="missing"),
                Line2D([0], [0], color="none", label="labels = edge strength"),
            ]
        else:
            legend = [
                Line2D([0], [0], color=COLORS["edge"], lw=2, label="discovered edge"),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=COLORS["node"],
                    markeredgecolor=COLORS["node_edge"],
                    markersize=10,
                    label="in graph",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=COLORS["node_iso"],
                    markeredgecolor="#AAAAAA",
                    markersize=10,
                    label="no edges",
                ),
                Line2D([0], [0], color="none", label="edge labels = strength"),
            ]
        ax.legend(
            handles=legend,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.06),
        )

        fig.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)


    def run_discovery(d: pd.DataFrame):
        env_pool_ = build_env_pool(d, m=M_DOWNSAMPLE, f_keep=F_KEEP, seed=SEED)
        env_pos_ = pd.DataFrame(
            [
                {
                    "name": e.name,
                    "tag": e.tag,
                    "n": len(e.data),
                    "n_pos": _outcome_n_pos(e.data),
                }
                for e in env_pool_
            ]
        )
        print(f"envs={len(env_pool_)} natural={sum(e.tag=='natural' for e in env_pool_)}")
        if len(env_pos_):
            print(env_pos_.to_string(index=False))
            print("env n_pos min/median", int(env_pos_.n_pos.min()), float(env_pos_.n_pos.median()))
        env_pos_.to_csv(ARTIFACTS / "env_positive_counts.csv", index=False)
        graph_ = discover_once(d, env_pool=env_pool_, seed=SEED, light=False, use_gcm=use_gcm)
        print("full |E|", graph_.number_of_edges())
        boot_dir_ = ARTIFACTS / "boot"
        freqs_, boot_graphs_ = bootstrap_stability(d, B=B, seed=SEED, cache_dir=boot_dir_, n_jobs=n_jobs)
        avg_e_ = float(np.mean([gg.number_of_edges() for gg in boot_graphs_])) if boot_graphs_ else 0.0
        thr_ = meinhausen_buhlmann_threshold(freqs_, avg_e_)
        stable_ = [(a, b) for (a, b), f in freqs_.items() if f >= thr_]
        if use_gcm:
            gcm_fit = _import_gcm_fit()
            gcm_meta_ = {}
            print("GCM scoring assembled graph (arrow_strength + falsify)", flush=True)
            try:
                strengths_, gcm_ll = gcm_fit.arrow_strengths_normalized(d, graph_, seed=SEED)
                gcm_meta_["gcm_ll"] = gcm_ll
            except Exception as exc:
                print(f"GCM arrow_strength failed ({exc}); using sklearn |coef| strengths", flush=True)
                strengths_ = edge_strengths(d, graph_)
                gcm_meta_["gcm_ll"] = float("nan")
                gcm_meta_["gcm_strength_error"] = str(exc)
            try:
                falsify_info = gcm_fit.falsify_dag(d, graph_, seed=SEED)
                gcm_meta_["falsify_p"] = falsify_info.get("p_value")
                gcm_meta_["falsified"] = falsify_info.get("falsified")
                gcm_meta_["falsifiable"] = falsify_info.get("falsifiable")
            except Exception as exc:
                print(f"GCM falsify_graph failed ({exc})", flush=True)
                gcm_meta_["falsify_p"] = None
                gcm_meta_["falsify_error"] = str(exc)
        else:
            print("GCM scoring skipped (use_gcm=False) — using sklearn |coef| strengths", flush=True)
            strengths_ = edge_strengths(d, graph_)
            gcm_meta_ = {
                "gcm_ll": None,
                "falsify_p": None,
                "falsified": None,
                "falsifiable": None,
                "gcm_skipped": True,
            }
        graph_.graph["gcm"] = gcm_meta_
        print(f"keep π={thr_:.3f} avg|E|={avg_e_:.1f} stable={len(stable_)}")
        return env_pool_, graph_, freqs_, boot_graphs_, thr_, stable_, strengths_


    t0 = time.time()
    env_pool, graph, freqs, boot_graphs, thr, stable, strengths = run_discovery(disc)

    # --- fallback: if the weak-pillar pool found no parents at all for ANY
    # outcome node (rec_targets in KG-based tiering), retry discovery against the
    # full selected-KPI set from Step 0/02 instead of quietly shipping a
    # disconnected graph.
    fallback_used = False
    incoming_to_target = [a for (a, b) in stable if b == TARGET] if TARGET else []
    # Outcome nodes: all rec_targets present in the discovery matrix.
    outcome_nodes = {c for c in columns if c in REC_TARGETS}
    stable_set = set(stable)
    any_outcome_has_parent = any(
        (a, b) in stable_set
        for (a, b) in stable_set
        if b in outcome_nodes
    )
    if not any_outcome_has_parent and pillars and set(kpi_cols) != set(kpi_cols_fallback):
        print(
            f"no parents found for any outcome node {sorted(outcome_nodes)} "
            f"in the weak-pillar pool {pillars} — "
            "falling back to the full selected KPI set from Step 0 (pre weak-pillar filter)."
        )
        fallback_used = True
        kpi_cols = list(kpi_cols_fallback)
        core_kpis, cols, disc, tiers = build_matrix(kpi_cols)
        bind_dataset(disc, tiers)
        env_pool, graph, freqs, boot_graphs, thr, stable, strengths = run_discovery(disc)
        incoming_to_target = [a for (a, b) in stable if b == TARGET] if TARGET else []
        outcome_nodes = {c for c in columns if c in REC_TARGETS}
        stable_set = set(stable)
        any_outcome_has_parent = any(
            (a, b) in stable_set for (a, b) in stable_set if b in outcome_nodes
        )
        if not any_outcome_has_parent:
            print(f"still no parents found for any outcome node even against the full selected set.")

    print("=== sink audit ===")
    s2_map = globals().get("LAST_STAGE2") or {}
    s3_map = globals().get("STAGE3_AUDIT") or {}
    gcm_map = globals().get("LAST_GCM_AUDIT") or {}
    n_boot = len(boot_graphs) or B
    if TARGET:
        incoming_tgt = [a for (a, b) in freqs if b == TARGET and a in disc.columns]
        rec_tgt = s2_map.get(TARGET)
        stage2_tgt = list(rec_tgt.parents) if rec_tgt is not None else []
        corr_cols = [c for c in dict.fromkeys(incoming_tgt + stage2_tgt) if c in disc.columns]
        if len(corr_cols) >= 2:
            print(f"candidate corr into {TARGET} (vote-split check):")
            print(disc[corr_cols].corr().round(4).to_string())
    audit_children = [c for c in columns if TIERS.get(c, 0) > 1]
    if TARGET and TARGET in audit_children:
        audit_children = [TARGET] + [c for c in audit_children if c != TARGET]
    for child in audit_children:
        rec = s2_map.get(child)
        parents2 = list(rec.parents) if rec is not None else []
        info = s3_map.get(child) or {}
        gap = info.get("stage2_gap_stat", float("nan"))
        xfer = info.get("stage2_transfer", "n/a")
        chosen = info.get("chosen", [])
        chosen_xfer = info.get("chosen_transfer", "n/a")
        chosen_gap = info.get("chosen_gap_stat", float("nan"))
        incoming = [(a, float(f), int(round(float(f) * n_boot))) for (a, b), f in freqs.items() if b == child]
        incoming.sort(key=lambda r: -r[1])
        gcm_info = gcm_map.get(child) or {}
        gcm_xfer = gcm_info.get("transfer", "n/a" if gcm_info.get("skipped") else gcm_info.get("error", "n/a"))
        gcm_out = gcm_info.get("parents_out", gcm_info.get("parents", []))
        print(
            f"{child} Stage-2 parents={parents2} transfer={xfer} gap_stat={gap} "
            f"Stage-3 chosen={chosen} transfer={chosen_xfer} gap_stat={chosen_gap} "
            f"GCM transfer={gcm_xfer} parents={gcm_out}"
        )
        print(f"{child} bootstrap into child: {incoming if incoming else 'none'} / {n_boot}")

    edge_rows = [
        {
            "source": a,
            "target": b,
            "confidence": float(freqs.get((a, b), 1.0)),
            "strength": float(strengths.get((a, b), 0.0)),
        }
        for a, b in stable
    ]
    edges = pd.DataFrame(edge_rows)
    edges.to_csv(OUTPUT_MAIN / "edges.csv", index=False)  # <-- the one main output
    graph_json = {
        "directed": True,
        "nodes": [{"id": c, "tier": int(tiers.get(c, 2))} for c in columns],
        "edges": edge_rows,
    }
    (ARTIFACTS / "edges.json").write_text(json.dumps(graph_json, indent=2), encoding="utf-8")
    gcm_meta = dict(graph.graph.get("gcm") or {})

    def _json_num(v):
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    metrics = {
        "pillars": pillars,
        "fallback_used": fallback_used,
        "candidate_kpis": kpi_cols,
        "n_rows": int(len(disc)),
        "n_vars": n_vars,
        "n_envs": len(env_pool),
        "B": B,
        "keep_threshold": thr,
        "n_stable_edges": len(stable),
        "n_full_edges": int(graph.number_of_edges()),
        "seconds": time.time() - t0,
        "gcm_ll": _json_num(gcm_meta.get("gcm_ll")),
        "falsify_p": _json_num(gcm_meta.get("falsify_p")),
        "falsified": gcm_meta.get("falsified"),
        "falsifiable": gcm_meta.get("falsifiable"),
    }
    if gcm_meta.get("gcm_strength_error"):
        metrics["gcm_strength_error"] = gcm_meta["gcm_strength_error"]
    if gcm_meta.get("falsify_error"):
        metrics["falsify_error"] = gcm_meta["falsify_error"]
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(ARTIFACTS / "glint_summary.csv", index=False)
    print(edges)

    tag = "FALLBACK:full_selected" if fallback_used else "+".join(pillars)
    subtitle = (
        "no parents found for target in the weak-pillar pool — fell back to the full selected set"
        if fallback_used
        else "joint parent search on weak-pillar KPIs + context + target (labels = strength)"
    )
    out_png = FIGURES / "03_glint_graph.png"
    save_structure_graph(
        out_png,
        columns,
        list(stable),
        tiers=tiers,
        labels={c: c for c in columns},
        strengths={(r["source"], r["target"]): r["strength"] for r in edge_rows},
        truth=None,
        title=f"GLINT merged [{tag}]  |E|={len(stable)}",
        subtitle=subtitle,
    )
    # also keep a copy next to edges
    save_structure_graph(
        ARTIFACTS / "graph.png",
        columns,
        list(stable),
        tiers=tiers,
        labels={c: c for c in columns},
        strengths={(r["source"], r["target"]): r["strength"] for r in edge_rows},
        truth=None,
        title=f"GLINT merged [{tag}]  |E|={len(stable)}",
        subtitle=subtitle,
    )
    print(metrics)
    return {"output": OUTPUT_MAIN, "artifacts": ARTIFACTS, "edges": edges, "metrics": metrics}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run GLINT discovery on weakest_pillar's output.")
    ap.add_argument("weakest_pillar_output_path", help="Folder with selected_kpis.csv, panel.csv, run.json")
    ap.add_argument("output_path", help="Folder to write edges.csv")
    ap.add_argument("--artifacts-path", default=None)
    ap.add_argument("--profile", action="store_true", help="Write a per-function cProfile timing report to artifacts/")
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Worker processes for the bootstrap loop. -1 (default) = all cores. 1 = sequential.",
    )
    ap.add_argument(
        "--gcm",
        dest="use_gcm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use DoWhy GCM for parent-set refinement and arrow-strength scoring (default: true). "
             "Pass --no-gcm to skip gcm_fit/DoWhy entirely and assemble the graph straight from "
             "the cheap sklearn screen, using linear-regression |coef| edge strengths instead.",
    )
    args = ap.parse_args()
    causal_graph(
        args.weakest_pillar_output_path,
        args.output_path,
        artifacts_path=args.artifacts_path,
        profile=args.profile,
        n_jobs=args.n_jobs,
        use_gcm=args.use_gcm,
    )