"""spec_loader.py — discovers and normalises all JSON-LD metadata inputs.

Reads kpis.jsonld (KPI catalog + grain + tables + pillars + collinear + contextVariables)
and the Croissant file (distribution + recordSet) from the dataset directory.
Returns typed dataclasses; 
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KPISpec:
    kpi_id: str
    name: str
    pillar: str
    unit: str
    direction: str          # higher_better | lower_better | diagnostic
    expression: str         # machine-readable expression (required in kpis.jsonld)
    formula: str            # prose formula (kept for display)
    parent: str | None      # bare kpi_id of the single parent (if isChild == yes)
    children: list[str]     # bare kpi_ids of children


@dataclass
class ContextVar:
    var_id: str
    expression: str
    role: str | None        # context | segment | segment_alt | null


@dataclass
class PillarSpec:
    name: str
    stage: str              # operational | post_outcome


@dataclass
class DatasetSpec:
    """Everything the pipeline needs, read from kpis.jsonld."""
    grain: list[str]                        # e.g. ["order_id"]
    tables: dict[str, str]                  # alias -> csv filename
    pillars: list[PillarSpec]
    collinear: list[tuple[str, str]]        # pairs that must never be parent-child
    context_vars: list[ContextVar]
    kpis: list[KPISpec]

    # Derived helpers (populated by __post_init__)
    kpi_by_id: dict[str, KPISpec] = field(default_factory=dict, init=False)
    pillar_stage: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.kpi_by_id = {k.kpi_id: k for k in self.kpis}
        self.pillar_stage = {p.name: p.stage for p in self.pillars}


@dataclass
class CroissantSpec:
    """Everything the pipeline needs from the Croissant file."""
    csv_files: list[str]                    # contentUrl names, in order
    columns: dict[str, list[str]]           # csv_filename -> [col_name, ...]
    col_dtype: dict[tuple[str, str], str]   # (csv, col) -> sc:Date | sc:Text | sc:Float | sc:Integer


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _strip_prefix(value: Any) -> str:
    """'kpi:DELAY_DAYS' -> 'DELAY_DAYS', 'rec:target' -> 'target'."""
    s = str(value or "")
    return s.split(":")[-1] if ":" in s else s


def load_kpis_jsonld(path: Path) -> DatasetSpec:
    """Parse kpis.jsonld (with the new top-level fields) into a DatasetSpec."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    # ---- grain ----
    grain_raw = raw.get("grain", [])
    if isinstance(grain_raw, str):
        grain_raw = [grain_raw]
    grain: list[str] = list(grain_raw)

    # ---- tables ----
    tables: dict[str, str] = dict(raw.get("tables") or {})

    # ---- pillars ----
    pillars: list[PillarSpec] = []
    for p in raw.get("pillars") or []:
        pillars.append(PillarSpec(name=str(p["name"]), stage=str(p.get("stage", "operational"))))

    # ---- collinear ----
    collinear: list[tuple[str, str]] = []
    for pair in raw.get("collinear") or []:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            collinear.append((_strip_prefix(pair[0]), _strip_prefix(pair[1])))

    # ---- contextVariables ----
    context_vars: list[ContextVar] = []
    for cv in raw.get("contextVariables") or []:
        role = cv.get("role")
        context_vars.append(ContextVar(
            var_id=str(cv["id"]),
            expression=str(cv.get("expression") or ""),
            role=str(role) if role is not None else None,
        ))

    # ---- KPIs ----
    kpis: list[KPISpec] = []
    for part in raw.get("hasPart") or []:
        kpi_id = _strip_prefix(part.get("@id", ""))
        if not kpi_id:
            continue

        # parent: first entry in `parents` list
        parents_raw = [_strip_prefix(p) for p in (part.get("parents") or [])]
        parent = parents_raw[0] if parents_raw else None

        # children
        children = [_strip_prefix(c) for c in (part.get("children") or [])]

        kpis.append(KPISpec(
            kpi_id=kpi_id,
            name=str(part.get("name") or kpi_id),
            pillar=str(part.get("pillar") or ""),
            unit=str(part.get("unit") or ""),
            direction=str(part.get("direction") or "higher_better"),
            expression=str(part.get("expression") or ""),
            formula=str(part.get("formula") or ""),
            parent=parent,
            children=children,
        ))

    return DatasetSpec(
        grain=grain,
        tables=tables,
        pillars=pillars,
        collinear=collinear,
        context_vars=context_vars,
        kpis=kpis,
    )


def load_croissant(path: Path) -> CroissantSpec:
    """Parse a Croissant-format JSON-LD into a CroissantSpec."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    csv_files: list[str] = []
    for dist in raw.get("distribution") or []:
        name = dist.get("contentUrl") or dist.get("name")
        if name:
            csv_files.append(str(name))

    columns: dict[str, list[str]] = {}
    col_dtype: dict[tuple[str, str], str] = {}

    for rs in raw.get("recordSet") or []:
        source = rs.get("source") or {}
        file_obj = source.get("fileObject") or {}
        csv_name = file_obj.get("@id") or file_obj.get("name") or ""
        # resolve to a real filename: match against csv_files by id/name
        resolved = _resolve_csv_name(csv_name, csv_files)
        if not resolved:
            continue
        for fld in rs.get("field") or []:
            col = fld.get("name") or fld.get("@id") or ""
            if not col:
                continue
            dtype = fld.get("dataType") or fld.get("sc:dataType") or ""
            if isinstance(dtype, list):
                dtype = dtype[0] if dtype else ""
            columns.setdefault(resolved, []).append(str(col))
            col_dtype[(resolved, str(col))] = str(dtype)

    return CroissantSpec(csv_files=csv_files, columns=columns, col_dtype=col_dtype)


def _resolve_csv_name(id_or_name: str, csv_files: list[str]) -> str:
    """Match a recordSet source id/name back to the distribution contentUrl.

    Croissant often uses the pattern  "<csvname>_fileobject"  as the @id.
    We strip that suffix (and any similar _fileObject variants) before matching.
    """
    if not id_or_name:
        return ""
    # exact match first
    if id_or_name in csv_files:
        return id_or_name

    # Strip common Croissant @id suffixes: _fileobject, _fileObject, -fileobject …
    import re
    cleaned = re.sub(r"[_\-]fileobject$", "", id_or_name, flags=re.IGNORECASE)

    # exact match after stripping suffix
    if cleaned in csv_files:
        return cleaned

    # basename match
    base = Path(cleaned).name
    for f in csv_files:
        if Path(f).name == base:
            return f

    # stem match (handles path prefixes)
    stem = Path(cleaned).stem
    for f in csv_files:
        if Path(f).stem == stem:
            return f

    # substring match (last resort)
    for f in csv_files:
        if stem and stem in Path(f).stem:
            return f

    return ""


def discover_metadata(dataset_path: Path) -> tuple[DatasetSpec, CroissantSpec]:
    """Find and load kpis.jsonld and the Croissant file from dataset_path.

    Rules:
      - kpis.jsonld: must be named exactly "kpis.jsonld" in dataset_path.
      - Croissant file: any *.jsonld that has both "distribution" and
        "recordSet" top-level keys. If multiple exist, the first found wins.
        kpis.jsonld is excluded from this search.
    """
    dataset_path = Path(dataset_path)
    kpis_path = dataset_path / "kpis.jsonld"
    if not kpis_path.exists():
        raise FileNotFoundError(f"kpis.jsonld not found in {dataset_path}")
    spec = load_kpis_jsonld(kpis_path)

    croissant_path: Path | None = None
    for p in sorted(dataset_path.glob("*.jsonld")):
        if p.name == "kpis.jsonld":
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if raw.get("distribution") and raw.get("recordSet"):
            croissant_path = p
            break

    if croissant_path is None:
        raise FileNotFoundError(
            f"No Croissant file (*.jsonld with 'distribution' + 'recordSet') found in {dataset_path}"
        )
    croissant = load_croissant(croissant_path)
    return spec, croissant