"""weakest_pillar — build the KPI panel and grade weak pillars (generic version).

All dataset-specific logic (Olist filenames, hardcoded KPI formulas, hardcoded
context features) has been removed.  Everything is driven by three inputs:

  kpis.jsonld           — grain, tables mapping, pillars (with stage), collinear
                          pairs, contextVariables (with role), and KPI expressions
  <croissant>.jsonld    — column dtypes (used for datetime parsing)
  <dataset>/*.csv       — raw data files named in kpis.jsonld `tables`

No kpi_list.py, no olist.jsonld required.

Writes (same as the original):
  <output_path>/selected_kpis.csv
  <output_path>/panel.csv
  <output_path>/run.json
  <artifacts_path>/coverage.csv, kpi_meta.csv, kpi_influence.csv,
                   pillar_influence.csv, pillar_summary.csv,
                   segment_priority.csv, bootstrap_priority.csv,
                   figures/01_coverage.png, 02_influence.png, ...
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from spec_loader import DatasetSpec, CroissantSpec, discover_metadata
from kpi_catalog import build_panel as _build_kpi_panel, load_tables, infer_date_columns


# ---------------------------------------------------------------------------
# Helpers — path helpers unchanged from original
# ---------------------------------------------------------------------------

def _default_artifacts_path(output_path: Path) -> Path:
    parts = list(output_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "output":
            parts[i] = "artifacts"
            return Path(*parts)
    return output_path.parent / "artifacts" / output_path.name


def _write_profile_log(profiler, log_path: Path, stage_name: str, elapsed: float) -> None:
    import io, pstats
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pstats_path = log_path.with_suffix(".pstats")
    profiler.dump_stats(str(pstats_path))
    buf = io.StringIO()
    buf.write(f"=== {stage_name} profile ===\nwall time: {elapsed:.2f}s\n\n")
    buf.write("--- top 40 by cumulative time ---\n")
    pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("cumulative").print_stats(40)
    buf.write("\n--- top 40 by self time ---\n")
    pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("tottime").print_stats(40)
    log_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"{stage_name} profile -> {log_path}  (raw stats -> {pstats_path})")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def weakest_pillar(
    dataset_path,
    output_path,
    artifacts_path=None,
    recommend_list: str = "",
    target_kpi: str = "",
    profile: bool = False,
    profile_log: str | None = None,
) -> dict:
    """Thin entry point; wraps _impl with optional cProfile."""
    if not profile:
        return _weakest_pillar_impl(dataset_path, output_path, artifacts_path, recommend_list, target_kpi)
    import cProfile
    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    try:
        result = _weakest_pillar_impl(dataset_path, output_path, artifacts_path, recommend_list, target_kpi)
    finally:
        profiler.disable()
    elapsed = time.time() - t0
    log_path = Path(profile_log) if profile_log else result["artifacts"] / "profile_weakest_pillar.log"
    _write_profile_log(profiler, log_path, "weakest_pillar", elapsed)
    return result


# ---------------------------------------------------------------------------
# Recommend-list parsing helpers (unchanged logic, generic KPI ID handling)
# ---------------------------------------------------------------------------

def _kpi_id(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("@id") or value.get("id") or ""
    if not value:
        return ""
    return str(value).split(":")[-1]


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _ids_in_text(text: str, catalog: set[str]) -> set[str]:
    found: set[str] = set()
    if not text:
        return found
    for kid in catalog:
        if re.search(rf"(?<![A-Z0-9_]){re.escape(kid)}(?![A-Z0-9_])", text):
            found.add(kid)
    return found


def _load_recommend(name: str, dataset_path: Path) -> tuple[dict | None, list, str | None, Path | None]:
    raw = Path(str(name).strip())
    base = raw if raw.is_absolute() else dataset_path / "causal" / raw.name
    stem = base.with_suffix("")
    jsonld_path = stem.with_suffix(".jsonld")
    json_path = stem.with_suffix(".json")
    payload = json.loads(jsonld_path.read_text(encoding="utf-8")) if jsonld_path.exists() else None
    fallback, json_target = [], None
    if json_path.exists():
        raw_json = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(raw_json, list):
            fallback = raw_json
        else:
            fallback = raw_json.get("kpis") or raw_json.get("recommends") or []
            json_target = raw_json.get("target")
    used = jsonld_path if jsonld_path.exists() else (json_path if json_path.exists() else None)
    return payload, fallback, json_target, used


def _recommended(payload: dict | None, fallback: list, catalog: set[str]) -> list[str]:
    rec: list[str] = []
    for node in (payload or {}).get("@graph") or []:
        if node.get("@type") == "rec:KPIAnalysis" or "rec:recommends" in node:
            for item in _as_list(node.get("rec:recommends")):
                kid = _kpi_id(item)
                if kid in catalog and kid not in rec:
                    rec.append(kid)
    if rec:
        return rec
    for item in fallback:
        kid = _kpi_id(item)
        if kid in catalog and kid not in rec:
            rec.append(kid)
    return rec


def _declared_target(payload: dict | None, json_target) -> str:
    for node in (payload or {}).get("@graph") or []:
        if node.get("@type") == "rec:KPIAnalysis":
            kid = _kpi_id(node.get("rec:target"))
            if kid:
                return kid
    return _kpi_id(json_target) if json_target else ""


def _causal_index(payload: dict | None) -> tuple[dict[str, dict], dict[str, str]]:
    nodes: dict[str, dict] = {}
    sector_name: dict[str, str] = {}
    for node in (payload or {}).get("@graph") or []:
        types = _as_list(node.get("@type"))
        nid = node.get("@id")
        if "kpi:Sector" in types and nid and node.get("name"):
            sector_name[str(nid)] = str(node["name"])
        kid = _kpi_id(nid) if nid else ""
        if kid and ("kpi:KPI" in types or "rec:RecommendedKPI" in types or node.get("sector") is not None):
            nodes[kid] = node
    return nodes, sector_name


def _pillar_of_recommended(kid: str, node: dict | None, sector_name: dict[str, str], catalog_pillar: dict[str, str]) -> str | None:
    if node:
        sector = node.get("sector")
        if isinstance(sector, dict):
            name = sector.get("name")
            sid = sector.get("@id") or sector.get("id")
            if not name and sid:
                name = sector_name.get(str(sid))
            if name:
                return str(name)
        elif isinstance(sector, str):
            return sector_name.get(sector) or catalog_pillar.get(kid)
    return catalog_pillar.get(kid)


def _formula_refs(kid: str, catalog: set[str], catalog_parents: dict[str, list[str]],
                  list_parent: dict[str, str], catalog_formula: dict[str, str],
                  causal_nodes: dict[str, dict]) -> set[str]:
    found: set[str] = set()
    found.update(p for p in catalog_parents.get(kid, []) if p in catalog)
    parent = list_parent.get(kid)
    if parent in catalog:
        found.add(parent)
    found.update(_ids_in_text(catalog_formula.get(kid) or "", catalog))
    node = causal_nodes.get(kid)
    if node:
        for item in _as_list(node.get("parentKPI")):
            ref = _kpi_id(item)
            if ref in catalog:
                found.add(ref)
        derived = node.get("derivedFrom") or {}
        if isinstance(derived, dict):
            for item in _as_list(derived.get("kpi")):
                ref = _kpi_id(item)
                if ref in catalog:
                    found.add(ref)
        formula = node.get("formula")
        expr = str(formula.get("expression") or "") if isinstance(formula, dict) else (str(formula) if formula else "")
        found.update(_ids_in_text(expr, catalog))
    found.discard(kid)
    return found


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def _weakest_pillar_impl(
    dataset_path,
    output_path,
    artifacts_path=None,
    recommend_list: str = "",
    target_kpi: str = "",
) -> dict:
    DATASET = Path(dataset_path)
    OUTPUT_MAIN = Path(output_path)
    ARTIFACTS = Path(artifacts_path) if artifacts_path else _default_artifacts_path(OUTPUT_MAIN)
    OUTPUT = ARTIFACTS  # artifact writes go here
    FIGURES = ARTIFACTS / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MAIN.mkdir(parents=True, exist_ok=True)

    RECOMMEND_LIST = recommend_list
    TARGET_KPI = target_kpi

    print("DATASET", DATASET)
    print("RECOMMEND_LIST", RECOMMEND_LIST or "(all catalog KPIs)")
    print("TARGET_KPI", TARGET_KPI or "(from rec:target on the recommend list)")

    # ------------------------------------------------------------------
    # Step 0a — Load metadata from JSON-LD (no hardcoding)
    # ------------------------------------------------------------------
    spec, croissant = discover_metadata(DATASET)
    GRAIN = spec.grain[0] if len(spec.grain) == 1 else spec.grain[0]  # single-grain assumption for now

    catalog_ids = [k.kpi_id for k in spec.kpis]
    catalog_set = set(catalog_ids)
    catalog_pillar = {k.kpi_id: k.pillar for k in spec.kpis}
    catalog_parents = {k.kpi_id: ([k.parent] if k.parent else []) for k in spec.kpis}
    catalog_formula = {k.kpi_id: k.formula for k in spec.kpis}
    list_parent = {k.kpi_id: k.parent for k in spec.kpis if k.parent}

    print(f"Catalog: {len(catalog_ids)} KPIs, grain={GRAIN!r}")
    print("Pillars:", [(p.name, p.stage) for p in spec.pillars])
    print("Context vars:", [cv.var_id for cv in spec.context_vars])

    # ------------------------------------------------------------------
    # Step 0b — Load CSV tables
    # ------------------------------------------------------------------
    date_cols_by_table = infer_date_columns(spec, croissant)
    tables = load_tables(DATASET, spec, date_cols_by_table=date_cols_by_table)

    # For any table without explicit date-col annotations, apply heuristic
    for alias, df in tables.items():
        for c in df.columns:
            n = c.lower()
            if any(tok in n for tok in ("timestamp", "_date", "_at", "datetime")):
                if not pd.api.types.is_datetime64_any_dtype(df[c]):
                    tables[alias][c] = pd.to_datetime(df[c], errors="coerce")

    for alias, df in tables.items():
        print(f"  {alias:20s} {df.shape}")

    # ------------------------------------------------------------------
    # Step 0c — Resolve recommend list & target KPI
    # ------------------------------------------------------------------
    recommend_name = str(RECOMMEND_LIST or "").strip()
    payload, fallback, json_target, used_path = (
        _load_recommend(recommend_name, DATASET) if recommend_name else (None, [], None, None)
    )
    use_all = (not recommend_name) or used_path is None

    def _require_catalog_kpi(kid: str, source: str) -> str:
        if kid not in catalog_set:
            raise ValueError(
                f"{source} {kid!r} is not in the KPI catalog. "
                "Add it to dataset/kpis.jsonld."
            )
        return kid

    declared = _declared_target(payload, json_target)
    override = _kpi_id(TARGET_KPI) if str(TARGET_KPI or "").strip() else ""
    if override:
        target_id = _require_catalog_kpi(override, "target_kpi")
    elif declared:
        target_id = _require_catalog_kpi(declared, "rec:target")
    else:
        # No single target — multi-target mode uses rec_targets only.
        target_id = None
        print("INFO: no single target declared — using multi-target mode (rec_targets).")
    target_pillar_id = catalog_pillar.get(target_id) if target_id else None

    # ------------------------------------------------------------------
    # Step 0d — Expand selection from recommend list
    # ------------------------------------------------------------------
    if use_all:
        recommended: list[str] = list(catalog_ids)
        pillars: list[str] = list(dict.fromkeys(k.pillar for k in spec.kpis))
        from_pillars = list(catalog_ids)
        from_formulas: list[str] = []
        selected = list(catalog_ids)
        print("Step 0: no recommend list — using the full catalog.")
    else:
        print("recommend list", used_path)
        recommended = _recommended(payload, fallback, catalog_set)
        causal_nodes, sector_name = _causal_index(payload)
        pillars = []
        for kid in recommended:
            pillar = _pillar_of_recommended(kid, causal_nodes.get(kid), sector_name, catalog_pillar)
            if pillar and pillar not in pillars:
                pillars.append(pillar)
        if target_pillar_id and target_pillar_id not in pillars:
            pillars.append(target_pillar_id)
        from_pillars = [k.kpi_id for k in spec.kpis if k.pillar in pillars]
        frontier = set(from_pillars)
        extras: set[str] = set()
        changed = True
        while changed:
            changed = False
            for kid in list(frontier):
                for ref in _formula_refs(kid, catalog_set, catalog_parents, list_parent, catalog_formula, causal_nodes):
                    if ref not in frontier:
                        extras.add(ref)
                        frontier.add(ref)
                        changed = True
        from_formulas = [k for k in catalog_ids if k in extras]
        selected = [k for k in catalog_ids if k in set(from_pillars) or k in extras]

    # The full list of recommended KPIs that are in the selected set —
    # used as multi-target list for influence modelling and by causal_graph.py.
    rec_targets: list[str] = [k for k in recommended if k in set(selected)]
    # If an explicit target was declared and isn't already in rec_targets, add it
    if target_id and target_id not in rec_targets:
        rec_targets = list(rec_targets) + [target_id]
    if target_id and target_id not in selected:
        selected = list(selected) + [target_id]

    selection = {
        "recommended": recommended,
        "pillars": pillars,
        "from_pillars": from_pillars,
        "from_formulas": from_formulas,
        "selected": selected,
        "rec_targets": rec_targets,
    }
    (OUTPUT / "selected_kpis.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("recommended:", recommended)
    print("pillars:", pillars)
    print("selected:", selected)
    print("rec_targets:", rec_targets)

    # ------------------------------------------------------------------
    # Step 1 — Build the panel using expression compiler (generic)
    # ------------------------------------------------------------------
    panel = _build_kpi_panel(tables, spec, GRAIN)
    print("panel", panel.shape)
    print(panel.head(3).to_string())

    # Coverage report
    cov_rows = []
    for kpi in spec.kpis:
        ok = kpi.kpi_id in panel.columns and panel[kpi.kpi_id].notna().any()
        cov_rows.append({
            "kpi_id": kpi.kpi_id,
            "pillar": kpi.pillar,
            "direction": kpi.direction,
            "status": "ok" if ok else ("missing" if not kpi.expression else "failed"),
            "note": kpi.formula,
        })
    coverage = pd.DataFrame(cov_rows)
    coverage.to_csv(OUTPUT / "coverage.csv", index=False)
    print(coverage[["kpi_id", "status"]])

    # KPI metadata table (replaces kpi_metamodel())
    meta = pd.DataFrame([
        {
            "kpi_id": k.kpi_id,
            "name": k.name,
            "pillar": k.pillar,
            "direction": k.direction,
            "unit": k.unit,
            "formula": k.formula,
            "expression": k.expression,
            "parents": [k.parent] if k.parent else [],
        }
        for k in spec.kpis
    ])
    meta.to_csv(OUTPUT / "kpi_meta.csv", index=False)

    # Coverage figure
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    order = coverage.iloc[::-1]
    colmap = {"ok": "#2B6CB0", "missing": "#C53030", "failed": "#DD6B20"}
    ax.barh(order["kpi_id"], np.ones(len(order)),
            color=[colmap.get(s, "#A0AEC0") for s in order["status"]], height=0.7)
    ax.set_title("KPI coverage")
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("ok", int((coverage.status == "ok").sum()), "/", len(coverage))

    # ------------------------------------------------------------------
    # Validate rec_targets
    # ------------------------------------------------------------------
    rec_on_panel = [t for t in rec_targets if t in panel.columns]
    if not rec_on_panel:
        raise ValueError(
            "No recommended target KPI was compiled onto the panel. "
            "Check expressions in kpis.jsonld and coverage.csv."
        )

    # ------------------------------------------------------------------
    # Step 2 — Build run.json metadata
    # ------------------------------------------------------------------
    pillar_of = dict(zip(meta["kpi_id"], meta["pillar"]))
    directions = dict(zip(meta["kpi_id"], meta["direction"]))

    # Context vars from spec
    segment_var = next((cv.var_id for cv in spec.context_vars if cv.role == "segment"), None)
    segment_alt_var = next((cv.var_id for cv in spec.context_vars if cv.role == "segment_alt"), None)
    inbound_vars = [cv.var_id for cv in spec.context_vars if cv.role == "context" and cv.var_id in panel.columns]

    # Recode (child) KPIs to drop from discovery
    recode_ids = [k.kpi_id for k in spec.kpis if k.parent]

    # Post-outcome pillar names (for causal_graph.py's POST_TREATMENT_FORBIDDEN logic)
    post_outcome_pillars = [p.name for p in spec.pillars if p.stage == "post_outcome"]

    # KG-based tiering: build a map of which context vars are derived from
    # other context vars (e.g. ORDER_COMPLEXITY = N_ITEMS * N_SELLERS).
    # causal_graph.py uses this to assign Tier 1 (derived) vs Tier 0 (root).
    _ctx_ids = {cv.var_id for cv in spec.context_vars}
    context_var_parents: dict[str, list[str]] = {}
    for cv in spec.context_vars:
        expr = getattr(cv, "expression", "") or ""
        # Find references to other context var IDs inside the expression
        parents = [vid for vid in _ctx_ids if vid != cv.var_id and re.search(r'\b' + re.escape(vid) + r'\b', expr)]
        if parents:
            context_var_parents[cv.var_id] = sorted(parents)

    run = {
        "grain": GRAIN,
        "segment": segment_var or (segment_alt_var or ""),
        "segment_alt": segment_alt_var if segment_alt_var != segment_var else None,
        "min_segment_n": 30,
        # rec_targets: every recommended KPI — used for multi-target influence
        # modelling and by causal_graph.py for KG-based tier assignment.
        "rec_targets": rec_targets,
        "drop_from_influence": recode_ids,
        "drop_from_discovery": recode_ids,
        "inbound": inbound_vars,
        "weak_n": 2,
        "bootstrap_B": 8,
        "n_rows": int(len(panel)),
        "selected": selected,
        "recommend_list": RECOMMEND_LIST,
        # Generic metadata passed through to causal_graph.py
        "collinear": [list(pair) for pair in spec.collinear],
        "post_outcome_pillars": post_outcome_pillars,
        # KG-based tiering: context vars derived from other context vars
        "context_var_parents": context_var_parents,
    }
    (OUTPUT / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2))

    # ------------------------------------------------------------------
    # Notebook 02 — grade weak pillars
    # ------------------------------------------------------------------
    SELECTED = [k for k in (selection.get("selected") or catalog_ids) if k in set(meta["kpi_id"])]
    REC_TARGETS = run["rec_targets"]
    DROP = set(run.get("drop_from_influence") or [])
    SEGMENT = run["segment"]
    MIN_N = int(run.get("min_segment_n") or 30)
    B = int(run.get("bootstrap_B") or 8)
    WEAK_N = int(run.get("weak_n") or 2)

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

    # All candidate driver KPIs — union of every KPI that could act as a driver
    # in any of the per-target models (i.e. all selected, non-dropped, non-diagnostic
    # KPIs that are on the panel).
    all_candidate_ids = [
        k for k in SELECTED
        if k in panel.columns
        and k not in DROP
        and directions.get(k) != "diagnostic"
    ]

    # For the NO_DRIVERS check: if none of the recommended KPIs have any potential
    # drivers across all the per-target models, skip grading entirely.
    def _has_cross_pillar_drivers(t_kpi: str) -> bool:
        t_pillar = pillar_of.get(t_kpi)
        return any(
            k for k in all_candidate_ids
            if k != t_kpi and pillar_of.get(k) != t_pillar
            and class_coverage_ok(panel, t_kpi, k)
        )

    NO_DRIVERS = not any(_has_cross_pillar_drivers(t) for t in REC_TARGETS if t in panel.columns)
    if NO_DRIVERS:
        print("WARNING: no cross-pillar driver KPIs found for any rec target. Skipping weak-pillar grading.")

    def build_goodness(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
        out = df.copy()
        for kid in ids:
            s = pd.to_numeric(out[kid], errors="coerce")
            p = s.rank(method="average", pct=True)
            d = directions.get(kid, "higher_better")
            out[f"g_{kid}"] = (1.0 - p) if d == "lower_better" else p
        return out

    def fit_influence(
        good: pd.DataFrame,
        ids: list[str],
        target_kpi: str,
        n_repeats: int = 8,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Fit one influence model: `ids` as drivers → `target_kpi` as target.

        Returns a DataFrame with columns [kpi_id, phi, r2, pillar, target_kpi].
        """
        gcols = [f"g_{k}" for k in ids if f"g_{k}" in good.columns]
        ids_ok = [c[2:] for c in gcols]
        if not ids_ok or target_kpi not in good.columns:
            return pd.DataFrame(columns=["kpi_id", "phi", "r2", "pillar", "target_kpi"])
        work = good[gcols + [target_kpi]].dropna()
        if len(work) < 10:
            return pd.DataFrame(columns=["kpi_id", "phi", "r2", "pillar", "target_kpi"])
        X = work[gcols].to_numpy(dtype=float)
        y = work[target_kpi].to_numpy(dtype=float)
        Xs = StandardScaler().fit_transform(X)
        binary = pd.Series(y).nunique(dropna=True) == 2
        if binary:
            pos = float(np.max(y))
            model = LogisticRegression(solver="liblinear", max_iter=200).fit(Xs, y)
            pred = model.predict_proba(Xs)[:, list(model.classes_).index(pos)]
            r2 = float(average_precision_score(y == pos, pred))
            perm = permutation_importance(model, Xs, y, n_repeats=n_repeats, random_state=seed, scoring="average_precision")
            print(f"  [{target_kpi}] PR-AUC={r2:.4f} (prevalence={float(np.mean(y == pos)):.4f})")
        else:
            model = LinearRegression().fit(Xs, y)
            r2 = float(model.score(Xs, y))
            perm = permutation_importance(model, Xs, y, n_repeats=n_repeats, random_state=seed, scoring="r2")
            print(f"  [{target_kpi}] R²={r2:.4f}")
        raw = np.maximum(perm.importances_mean, 0.0)
        if raw.sum() <= 1e-12:
            raw = np.abs(model.coef_).flatten()
        shares = raw / (raw.sum() + 1e-12) * max(r2, 0.0)
        return pd.DataFrame({
            "kpi_id": ids_ok,
            "phi": shares,
            "r2": r2,
            "pillar": [pillar_of.get(k, "") for k in ids_ok],
            "target_kpi": target_kpi,
        }).sort_values("phi", ascending=False)

    def _aggregate_phi(phi_parts: list[pd.DataFrame]) -> pd.DataFrame:
        """Average φ per (kpi_id, pillar) across all per-target models.

        The returned DataFrame has the same schema as the old single-model phi
        (columns: kpi_id, phi, r2, pillar) so all downstream code is unchanged.
        The r2 column holds the mean R² across models where each driver appeared.
        """
        if not phi_parts:
            return pd.DataFrame(columns=["kpi_id", "phi", "r2", "pillar"])
        combined = pd.concat(phi_parts, ignore_index=True)
        agg = (
            combined.groupby(["kpi_id", "pillar"], as_index=False)
            .agg(phi=("phi", "mean"), r2=("r2", "mean"))
        )
        return agg.sort_values("phi", ascending=False).reset_index(drop=True)

    def _run_influence_loop(
        df: pd.DataFrame,
        rec_targets: list[str],
        candidate_ids: list[str],
        n_repeats: int = 8,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Run one fit_influence model per rec target; aggregate into one phi DataFrame."""
        good = build_goodness(df, candidate_ids)
        # Copy raw target columns onto goodness frame so fit_influence can find them
        for t in rec_targets:
            if t in df.columns and t not in good.columns:
                good[t] = df[t].values

        phi_parts: list[pd.DataFrame] = []
        for i, t_kpi in enumerate(rec_targets):
            if t_kpi not in df.columns:
                print(f"  skip [{t_kpi}]: not on panel")
                continue
            t_pillar = pillar_of.get(t_kpi)
            # Drivers = candidate KPIs from OTHER pillars only (never self, never same pillar)
            drivers_for_t = [
                k for k in candidate_ids
                if k != t_kpi
                and pillar_of.get(k) != t_pillar
                and class_coverage_ok(df, t_kpi, k)
            ]
            if not drivers_for_t:
                print(f"  skip [{t_kpi}]: no cross-pillar drivers")
                continue
            print(f"  influence model {i+1}/{len(rec_targets)}: target={t_kpi} drivers={drivers_for_t}")
            phi_t = fit_influence(good, drivers_for_t, target_kpi=t_kpi,
                                  n_repeats=n_repeats, seed=seed)
            if not phi_t.empty:
                phi_parts.append(phi_t)

        return _aggregate_phi(phi_parts)

    def within_pillar_weights(phi_df: pd.DataFrame) -> pd.DataFrame:
        out = phi_df.copy()
        out["w"] = out.groupby("pillar")["phi"].transform(lambda s: s / (s.sum() + 1e-12))
        return out

    def pillar_infl(phi_df: pd.DataFrame) -> pd.DataFrame:
        r2 = float(phi_df["r2"].iloc[0])
        g = phi_df.groupby("pillar", as_index=False)["phi"].sum().rename(columns={"phi": "Infl"})
        g["Infl_share"] = g["Infl"] / (r2 + 1e-12)
        g["r2"] = r2
        return g.sort_values("Infl", ascending=False)

    def pillar_perf_by_unit(good: pd.DataFrame, weights: pd.DataFrame, unit: str, units: list) -> pd.DataFrame:
        pillar_order = [p for p, _ in weights.groupby("pillar")]
        g_cols = sorted({f"g_{kid}" for kid in weights["kpi_id"].unique() if f"g_{kid}" in good.columns})
        n_orders = good.groupby(unit).size().reindex(units)
        if not g_cols:
            wide = pd.DataFrame(np.nan, index=units, columns=pillar_order)
        else:
            gm = good.groupby(unit)[g_cols].mean().reindex(units)
            perf_cols = {}
            for pillar, wsub in weights.groupby("pillar"):
                col_w = {f"g_{r.kpi_id}": float(r.w) for r in wsub.itertuples() if f"g_{r.kpi_id}" in g_cols}
                if col_w:
                    cols = list(col_w.keys())
                    vals = gm[cols].to_numpy(dtype=float)
                    finite = np.isfinite(vals)
                    w_arr = np.array([col_w[c] for c in cols], dtype=float)
                    num = (np.where(finite, vals, 0.0) * w_arr).sum(axis=1)
                    den = (finite * w_arr).sum(axis=1)
                    perf_cols[pillar] = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), np.nan)
                else:
                    perf_cols[pillar] = np.full(len(units), np.nan)
            wide = pd.DataFrame(perf_cols, index=units)[pillar_order]
        long = wide.stack().reset_index()
        long.columns = [unit, "pillar", "Perf"]
        long["n_orders"] = long[unit].map(n_orders)
        return long

    def grade_segments(good: pd.DataFrame, weights: pd.DataFrame, infl: pd.DataFrame, unit: str, min_n: int) -> pd.DataFrame:
        counts = good[unit].value_counts()
        units = counts[counts >= min_n].index.tolist()
        if not units:
            return pd.DataFrame()
        perf = pillar_perf_by_unit(good, weights, unit, units)
        pri = perf.merge(infl[["pillar", "Infl", "Infl_share", "r2"]], on="pillar", how="left")
        pri["Priority"] = pri["Infl"].fillna(0) * (1.0 - pri["Perf"].fillna(0.5))
        return pri[[unit, "pillar", "Perf", "n_orders", "Infl", "Infl_share", "r2", "Priority"]].reset_index(drop=True)

    def summarize(seg: pd.DataFrame, unit: str) -> pd.DataFrame:
        return (
            seg.groupby("pillar", as_index=False)
            .agg(Priority_mean=("Priority", "mean"), Perf_mean=("Perf", "mean"),
                 Infl=("Infl", "first"), n_segments=(unit, "nunique"))
            .sort_values("Priority_mean", ascending=False)
            .reset_index(drop=True)
        )

    if NO_DRIVERS:
        goodness = panel.copy()
        phi = pd.DataFrame(columns=["kpi_id", "phi", "r2", "pillar"])
    else:
        print(f"Running influence loop over {len(REC_TARGETS)} rec targets …")
        phi = _run_influence_loop(panel, REC_TARGETS, all_candidate_ids)
        # Build a goodness frame for segment grading (uses all candidate ids)
        goodness = build_goodness(panel, all_candidate_ids)
        # Ensure all rec target columns are present on goodness for grade_segments
        for t in REC_TARGETS:
            if t in panel.columns and t not in goodness.columns:
                goodness[t] = panel[t].values
        print(phi)

    weights = within_pillar_weights(phi)
    infl = pillar_infl(phi) if len(phi) else pd.DataFrame(columns=["pillar", "Infl", "Infl_share", "r2"])
    seg = (
        grade_segments(goodness, weights, infl, SEGMENT, MIN_N)
        if (not NO_DRIVERS and len(phi) and SEGMENT and SEGMENT in goodness.columns)
        else pd.DataFrame()
    )
    summary = (
        summarize(seg, SEGMENT) if len(seg)
        else pd.DataFrame(columns=["pillar", "Priority_mean", "Perf_mean", "Infl", "n_segments"])
    )
    print(infl)
    print(summary)

    state_summary = None
    seg_alt = run.get("segment_alt")
    if (not NO_DRIVERS) and seg_alt and seg_alt in goodness.columns:
        state_seg = grade_segments(goodness, weights, infl, seg_alt, min(MIN_N, 200))
        state_summary = summarize(state_seg, seg_alt)
        print(state_summary)

    # Bootstrap stability — each replicate runs the full per-target loop
    rng = np.random.default_rng(0)
    n = len(panel)
    records: dict[str, list[float]] = {}
    top_hits: dict[str, int] = {}
    if not NO_DRIVERS:
        for b in range(B):
            boot = panel.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
            phib = _run_influence_loop(boot, REC_TARGETS, all_candidate_ids,
                                       n_repeats=4, seed=b)
            if phib.empty:
                continue
            # Goodness frame for this bootstrap replicate
            gboot = build_goodness(boot, all_candidate_ids)
            for t in REC_TARGETS:
                if t in boot.columns and t not in gboot.columns:
                    gboot[t] = boot[t].values
            wb = within_pillar_weights(phib)
            ib = pillar_infl(phib)
            sb = grade_segments(gboot, wb, ib, SEGMENT, MIN_N) if SEGMENT and SEGMENT in gboot.columns else pd.DataFrame()
            sm = summarize(sb, SEGMENT) if len(sb) else pd.DataFrame()
            if sm.empty:
                continue
            weak_b = sm.head(WEAK_N)["pillar"].tolist()
            for r in sm.itertuples():
                records.setdefault(r.pillar, []).append(float(r.Priority_mean))
            for p in weak_b:
                top_hits[p] = top_hits.get(p, 0) + 1

    boot_df = pd.DataFrame([
        {
            "pillar": p,
            "Priority_mean": float(np.mean(v)),
            "p05": float(np.quantile(v, 0.05)),
            "p95": float(np.quantile(v, 0.95)),
            f"P_in_top_{WEAK_N}": top_hits.get(p, 0) / B,
            "n_boot": len(v),
        }
        for p, v in records.items()
    ], columns=["pillar", "Priority_mean", "p05", "p95", f"P_in_top_{WEAK_N}", "n_boot"])
    if len(boot_df):
        boot_df = boot_df.sort_values("Priority_mean", ascending=False).reset_index(drop=True)

    weak: list[str]
    if NO_DRIVERS:
        weak = []
        print("weak pillars → GLINT: []  (no driver KPIs)")
    else:
        weak = boot_df.head(WEAK_N)["pillar"].tolist() if len(boot_df) else summary.head(WEAK_N)["pillar"].tolist()
        print("weak pillars → GLINT:", weak)

    # ------------------------------------------------------------------
    # Write artifacts
    # ------------------------------------------------------------------
    phi.to_csv(OUTPUT / "kpi_influence.csv", index=False)
    infl.to_csv(OUTPUT / "pillar_influence.csv", index=False)
    summary.to_csv(OUTPUT / "pillar_summary.csv", index=False)
    seg.to_csv(OUTPUT / "segment_priority.csv", index=False)
    boot_df.to_csv(OUTPUT / "bootstrap_priority.csv", index=False)
    if state_summary is not None:
        state_summary.to_csv(OUTPUT / "state_priority.csv", index=False)
    (OUTPUT / "weak_pillars.json").write_text(json.dumps({
        "pillars": weak, "segment": SEGMENT,
        "selected": SELECTED, "from_formulas": selection.get("from_formulas") or [],
    }, indent=2), encoding="utf-8")

    # Figures
    if len(phi):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(phi["kpi_id"].iloc[::-1], phi["phi"].iloc[::-1], color="#2B6CB0")
        ax.set_xlabel("φ (mean share of R², aggregated over rec targets)")
        n_models = len(REC_TARGETS)
        mean_r2 = float(phi["r2"].mean()) if len(phi) else float("nan")
        ax.set_title(f"KPI influence (aggregated over {n_models} targets,  mean R²={mean_r2:.3f})")
        fig.tight_layout()
        fig.savefig(FIGURES / "02_influence.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    if len(summary):
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.bar(summary["pillar"], summary["Priority_mean"], color="#C05621")
        ax.set_ylabel("mean Priority")
        ax.set_title("Pillar priority (higher = weaker / more urgent)")
        fig.tight_layout()
        fig.savefig(FIGURES / "02_priority.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    if len(boot_df):
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        yerr = np.vstack([boot_df["Priority_mean"] - boot_df["p05"], boot_df["p95"] - boot_df["Priority_mean"]])
        ax.bar(boot_df["pillar"], boot_df["Priority_mean"], yerr=yerr, color="#2B6CB0", capsize=4)
        ax.set_title(f"Bootstrap Priority (B={B})")
        fig.tight_layout()
        fig.savefig(FIGURES / "02_bootstrap.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print("saved figures →", FIGURES)

    # ------------------------------------------------------------------
    # Primary candidates (same logic as original, now from spec)
    # ------------------------------------------------------------------
    from_formulas_set = set(selection.get("from_formulas") or [])
    rec_targets_set = set(REC_TARGETS)
    if weak:
        weak_pillar_kpis = {k for k in SELECTED if pillar_of.get(k) in set(weak)}
        # Always keep: weak-pillar KPIs + formula dependencies + all rec targets
        # (every rec target is a potential causal endpoint in causal_graph.py)
        keep = weak_pillar_kpis | from_formulas_set | rec_targets_set
        primary_candidates = {k for k in SELECTED if k in keep}
    else:
        primary_candidates = set(SELECTED)
    print("primary candidate KPIs:", [k for k in SELECTED if k in primary_candidates])

    # ------------------------------------------------------------------
    # Main outputs
    # ------------------------------------------------------------------
    sel_rows = [
        {
            "kpi_id": kid,
            "pillar": pillar_of.get(kid),
            "parent": list_parent.get(kid),
            "rec_target": kid in rec_targets_set,  # True for all rec KPIs (tier-3 in causal graph)
            "primary_candidate": kid in primary_candidates,
        }
        for kid in SELECTED
    ]
    pd.DataFrame(sel_rows).to_csv(OUTPUT_MAIN / "selected_kpis.csv", index=False)

    # Panel: keep selected KPIs + inbound context vars + all rec_targets
    keep_cols: list[str] = []
    for c in [GRAIN] + SELECTED + inbound_vars + REC_TARGETS:
        if c in panel.columns and c not in keep_cols:
            keep_cols.append(c)
    panel[keep_cols].to_csv(OUTPUT_MAIN / "panel.csv", index=False)

    main_run = dict(run)
    main_run["pillars"] = weak
    # Ensure rec_targets is always present in main run.json consumed by causal_graph.py
    main_run.setdefault("rec_targets", rec_targets)
    (OUTPUT_MAIN / "run.json").write_text(json.dumps(main_run, indent=2), encoding="utf-8")

    print("weakest_pillar main outputs ->", OUTPUT_MAIN)
    print("weakest_pillar artifacts     ->", ARTIFACTS)
    return {"output": OUTPUT_MAIN, "artifacts": ARTIFACTS}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the panel and grade weak KPI pillars.")
    ap.add_argument("dataset_path", help="Folder containing kpis.jsonld, <croissant>.jsonld, and CSVs")
    ap.add_argument("output_path", help="Folder to write selected_kpis.csv, panel.csv, run.json")
    ap.add_argument("--artifacts-path", default=None)
    ap.add_argument("--recommend-list", default="", help="Path or name of recommend list JSON-LD")
    ap.add_argument("--target-kpi", default="", help="Override target KPI ID")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    weakest_pillar(
        args.dataset_path,
        args.output_path,
        artifacts_path=args.artifacts_path,
        recommend_list=args.recommend_list,
        target_kpi=args.target_kpi,
        profile=args.profile,
    )