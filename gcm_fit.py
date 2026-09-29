"""DoWhy GCM helpers used after the cheap sklearn screen.

Imported only on the full-data path (not bootstrap workers). Missing DoWhy
raises — no silent R² fallback.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

MAX_GCM_ROWS = 8000
SCORE_ROWS = 2000
FALSIFY_PERMUTATIONS = 8
TRANSFER_ENV_CAP = 8
FALSIFY_ROWS = 800


def require_dowhy():
    try:
        import dowhy.gcm as gcm  # noqa: F401
        from dowhy.gcm.falsify import falsify_graph  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "DoWhy GCM is required for the post-screen evaluation/scoring step. "
            "Install it with: python3 -m pip install dowhy"
        ) from exc


def _sample(df: pd.DataFrame, n: int = MAX_GCM_ROWS, seed: int = 0) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()
    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna().reset_index(drop=True)


def _make_scm(graph: nx.DiGraph, data: pd.DataFrame):
    require_dowhy()
    import dowhy.gcm as gcm

    try:
        model = gcm.InvertibleStructuralCausalModel(graph)
        gcm.auto.assign_causal_mechanisms(model, data)
        return model
    except Exception:
        model = gcm.StructuralCausalModel(graph)
        gcm.auto.assign_causal_mechanisms(model, data)
        return model


def fit_local_gcm(df: pd.DataFrame, child: str, parents: list[str], seed: int = 0):
    """Fit a tiny SCM parents → child with auto-assigned mechanisms."""
    require_dowhy()
    import dowhy.gcm as gcm

    cols = list(dict.fromkeys(list(parents) + [child]))
    data = _numeric(_sample(df, seed=seed), cols)
    if child not in data.columns or len(data) < 50:
        raise ValueError(f"not enough rows to fit GCM for {child}")
    g = nx.DiGraph()
    g.add_nodes_from(cols)
    g.add_edges_from((p, child) for p in parents if p in data.columns and p != child)
    model = _make_scm(g, data)
    gcm.fit(model, data)
    return model, data


def _predict_child(model, data: pd.DataFrame, child: str, parents: list[str]) -> np.ndarray | None:
    try:
        mech = model.causal_mechanism(child)
    except Exception:
        return None
    if not parents:
        y = data[child].to_numpy(dtype=float)
        return np.full(len(y), float(np.nanmean(y)))
    X = data[parents].to_numpy(dtype=float)
    pred_model = getattr(mech, "prediction_model", None)
    if pred_model is not None:
        try:
            pred = np.asarray(pred_model.predict(X)).reshape(-1)
            if pred.shape[0] == len(data):
                return pred.astype(float)
        except Exception:
            pass
    if hasattr(mech, "estimate_probabilities"):
        try:
            proba = np.asarray(mech.estimate_probabilities(X))
            if proba.ndim == 2:
                pred = proba[:, -1]
            else:
                pred = proba.reshape(-1)
            if pred.shape[0] == len(data):
                return pred.astype(float)
        except Exception:
            pass
    return None


def mechanism_r2(model, data: pd.DataFrame, child: str, parents: list[str]) -> float:
    y = data[child].to_numpy(dtype=float)
    if y.std() < 1e-12:
        return 0.0
    pred = _predict_child(model, data, child, parents)
    if pred is None:
        return float("nan")
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _pnl_ok(pnl) -> tuple[bool, float | None]:
    if pnl is None:
        return True, None
    pnl_p = None
    try:
        if pnl[0] is not None:
            pnl_p = float(pnl[0])
    except Exception:
        pnl_p = None
    verdict = ""
    try:
        if len(pnl) > 1 and pnl[1] is not None:
            verdict = str(pnl[1]).lower()
    except Exception:
        verdict = ""
    if pnl_p is not None:
        return pnl_p >= 0.05, pnl_p
    if "not reject" in verdict or "cannot reject" in verdict or "accepted" in verdict:
        return True, pnl_p
    if "reject" in verdict:
        return False, pnl_p
    return True, pnl_p


def evaluate_local(
    df: pd.DataFrame, child: str, parents: list[str], seed: int = 0
) -> dict[str, Any]:
    """Fit + evaluate_causal_model on a local parent set. Returns scores."""
    require_dowhy()
    import dowhy.gcm as gcm

    model, data = fit_local_gcm(df, child, parents, seed=seed)
    result = gcm.evaluate_causal_model(
        model,
        data,
        max_num_samples=MAX_GCM_ROWS,
        evaluate_causal_mechanisms=True,
        compare_mechanism_baselines=False,
        evaluate_invertibility_assumptions=True,
        evaluate_overall_kl_divergence=False,
        evaluate_causal_structure=False,
    )
    perf = (result.mechanism_performances or {}).get(child)
    pnl = (result.pnl_assumptions or {}).get(child)
    pnl_ok, pnl_p = _pnl_ok(pnl)
    r2 = mechanism_r2(model, data, child, parents)
    if perf is not None and getattr(perf, "r2", None) is not None and np.isfinite(perf.r2):
        r2 = float(perf.r2)
    crps = float(perf.crps) if perf is not None and perf.crps is not None else float("nan")
    return {
        "model": model,
        "data": data,
        "r2": r2,
        "crps": crps,
        "mse": float(perf.mse) if perf is not None and perf.mse is not None else float("nan"),
        "pnl_p": pnl_p,
        "pnl_ok": pnl_ok,
        "kl": float(perf.kl_divergence) if perf is not None and perf.kl_divergence is not None else float("nan"),
    }


def gcm_transfers(
    env_pool: list,
    child: str,
    parents: list[str],
    pooled: pd.DataFrame,
    seed: int = 0,
) -> tuple[bool, float, dict[str, Any]]:
    """Pooled GCM must still fit in environments (held-out R² gap + invertibility)."""
    pooled_eval = evaluate_local(pooled, child, parents, seed=seed)
    gaps: list[float] = []
    env_n = 0
    worlds = [e for e in env_pool if getattr(e, "tag", "") == "natural"]
    worlds = worlds[:TRANSFER_ENV_CAP]
    if not worlds:
        worlds = list(env_pool)[:TRANSFER_ENV_CAP]
    cols = list(dict.fromkeys(list(parents) + [child]))
    for e in worlds:
        data = getattr(e, "data", e)
        env_num = _numeric(data, cols)
        if len(env_num) < 80:
            continue
        env_n += 1
        pooled_r2 = mechanism_r2(pooled_eval["model"], env_num, child, parents)
        cut = len(env_num) // 2
        if cut < 40:
            continue
        try:
            local_model, _ = fit_local_gcm(env_num.iloc[:cut], child, parents, seed=seed + env_n)
            local_r2 = mechanism_r2(local_model, env_num.iloc[cut:].reset_index(drop=True), child, parents)
        except Exception:
            continue
        if np.isfinite(pooled_r2) and np.isfinite(local_r2):
            gaps.append(float(pooled_r2 - local_r2))
    gap_stat = float(np.median(gaps)) if gaps else float("nan")
    transfers = bool(pooled_eval["pnl_ok"] and (not gaps or gap_stat >= -0.25))
    return transfers, float(pooled_eval["r2"]), {
        "pooled_crps": pooled_eval["crps"],
        "gap_stat": gap_stat,
        "pnl_p": pooled_eval["pnl_p"],
        "pnl_ok": pooled_eval["pnl_ok"],
        "n_envs": env_n,
        "n_gap": len(gaps),
    }


def prune_parents_gcm_1se(
    df: pd.DataFrame, child: str, parents: list[str], seed: int = 0
) -> tuple[list[str], dict[str, Any]]:
    """Drop a parent if removing it does not worsen GCM R² by more than 0.01."""
    if len(parents) <= 1:
        return list(parents), {"dropped": []}
    try:
        full_model, full_data = fit_local_gcm(df, child, parents, seed=seed)
        full_r2 = mechanism_r2(full_model, full_data, child, parents)
    except Exception as exc:
        return list(parents), {"dropped": [], "error": str(exc)}
    dropped: list[str] = []
    kept = list(parents)
    for p in list(parents):
        reduced = [q for q in kept if q != p]
        try:
            red_model, red_data = fit_local_gcm(df, child, reduced, seed=seed)
            red_r2 = mechanism_r2(red_model, red_data, child, reduced)
        except Exception:
            continue
        if np.isfinite(full_r2) and np.isfinite(red_r2) and (full_r2 - red_r2) <= 0.01:
            dropped.append(p)
            kept = reduced
            full_r2 = red_r2
    return kept, {"dropped": dropped, "r2": full_r2}


def refine_parent_sets(
    df: pd.DataFrame,
    env_pool: list,
    parents: dict[str, tuple[str, ...]],
    columns: list[str],
    root_tier: dict[str, int] | None = None,
    seed: int = 0,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """GCM transfer + likelihood prune on cheap-screen survivors only."""
    require_dowhy()
    out: dict[str, tuple[str, ...]] = {}
    audit: dict[str, Any] = {}
    n_children = sum(1 for c, pas in parents.items() if (root_tier or {}).get(c, 2) != 1 and pas)
    done = 0
    for child, pas in parents.items():
        tier = (root_tier or {}).get(child, 2)
        if tier == 1 or not pas:
            out[child] = tuple(pas)
            audit[child] = {"skipped": True, "parents": list(pas)}
            continue
        done += 1
        z = [p for p in pas if p in columns and p != child]
        print(f"  GCM refine {done}/{n_children}: {child} ← {z}", flush=True)
        try:
            ok, r2, xfer = gcm_transfers(env_pool, child, z, df, seed=seed)
        except Exception as exc:
            out[child] = tuple(z)
            audit[child] = {"error": str(exc), "parents": z, "transfer": "error"}
            print(f"    error: {exc}", flush=True)
            continue
        if not ok:
            out[child] = tuple(z)
            audit[child] = {"transfer": "skip", "parents": z, "r2": r2, **xfer}
            print(f"    transfer=skip r2={r2:.3f} gap={xfer.get('gap_stat')}", flush=True)
            continue
        kept, prune_info = prune_parents_gcm_1se(df, child, z, seed=seed)
        out[child] = tuple(kept)
        audit[child] = {
            "transfer": "keep",
            "parents_in": z,
            "parents_out": kept,
            "r2": r2,
            **xfer,
            **prune_info,
        }
        print(f"    transfer=keep r2={r2:.3f} parents {z} → {kept}", flush=True)
    return out, audit


def fit_graph_gcm(df: pd.DataFrame, graph: nx.DiGraph, seed: int = 0, n_rows: int = SCORE_ROWS):
    require_dowhy()
    import dowhy.gcm as gcm

    cols = [c for c in graph.nodes() if c in df.columns]
    data = _numeric(_sample(df, n=n_rows, seed=seed), cols)
    g = nx.DiGraph()
    g.add_nodes_from(cols)
    g.add_edges_from((u, v) for u, v in graph.edges() if u in cols and v in cols)
    model = _make_scm(g, data)
    gcm.fit(model, data)
    return model, data


def arrow_strengths_normalized(
    df: pd.DataFrame, graph: nx.DiGraph, seed: int = 0
) -> tuple[dict[tuple[str, str], float], float]:
    """DoWhy arrow_strength, incoming edges per child normalized to 1. Also mean GCM R²."""
    require_dowhy()
    import dowhy.gcm as gcm

    print(f"  GCM fit assembled graph on {SCORE_ROWS} rows", flush=True)
    model, data = fit_graph_gcm(df, graph, seed=seed, n_rows=SCORE_ROWS)
    strengths: dict[tuple[str, str], float] = {}
    r2s: list[float] = []
    children = [n for n in graph.nodes() if n in data.columns and list(graph.predecessors(n))]
    for i, node in enumerate(children, 1):
        pas = [p for p in graph.predecessors(node) if p in data.columns]
        print(f"  arrow_strength {i}/{len(children)}: {node} ← {pas}", flush=True)
        try:
            raw = gcm.arrow_strength(
                model,
                target_node=node,
                parent_samples=data[[c for c in ([node] + pas) if c in data.columns]],
                n_jobs=1,
                num_samples_conditional=150,
                max_num_runs=25,
            )
            mag = {k: abs(float(v)) for k, v in raw.items()}
            tot = sum(mag.values()) + 1e-12
            for key, w in mag.items():
                if isinstance(key, tuple) and len(key) == 2:
                    a, b = key
                else:
                    a, b = key, node
                strengths[(str(a), str(b))] = float(w / tot)
        except Exception as exc:
            print(f"    arrow_strength failed for {node}: {exc}", flush=True)
        r2s.append(mechanism_r2(model, data, node, pas))
    roots = [n for n in graph.nodes() if n in data.columns and not list(graph.predecessors(n))]
    for node in roots:
        r2s.append(mechanism_r2(model, data, node, []))
    mean_r2 = float(np.nanmean(r2s)) if r2s else float("nan")
    return strengths, mean_r2


def falsify_dag(df: pd.DataFrame, graph: nx.DiGraph, seed: int = 0) -> dict[str, Any]:
    require_dowhy()
    from dowhy.gcm.falsify import FalsifyConst, falsify_graph
    from dowhy.gcm.independence_test import regression_based

    cols = [c for c in graph.nodes() if c in df.columns]
    data = _numeric(_sample(df, n=min(FALSIFY_ROWS, MAX_GCM_ROWS), seed=seed), cols)
    g = nx.DiGraph()
    g.add_nodes_from(cols)
    g.add_edges_from((u, v) for u, v in graph.edges() if u in cols and v in cols)
    print(
        f"  falsify_graph n={len(data)} permutations={FALSIFY_PERMUTATIONS} (regression CI)",
        flush=True,
    )
    result = falsify_graph(
        g,
        data,
        n_permutations=FALSIFY_PERMUTATIONS,
        independence_test=regression_based,
        conditional_independence_test=regression_based,
        show_progress_bar=True,
        plot_histogram=False,
        n_jobs=1,
        allow_data_subset=True,
    )
    p_lmc = None
    try:
        p_lmc = float(result.summary[FalsifyConst.VALIDATE_LMC][FalsifyConst.P_VALUE])
    except Exception:
        try:
            p_lmc = float(result.summary.get("validate_lmc", {}).get("p_value"))
        except Exception:
            p_lmc = None
    return {
        "falsified": bool(result.falsified) if getattr(result, "falsified", None) is not None else None,
        "falsifiable": bool(result.falsifiable) if getattr(result, "falsifiable", None) is not None else None,
        "p_value": p_lmc,
    }
