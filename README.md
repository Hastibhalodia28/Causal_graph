# GLINT KPI Causal Discovery

A metadata-driven pipeline that builds a grain-level KPI panel, identifies weak KPI pillars, and discovers a stable causal graph for a target KPI.

## Pipeline Flow

```text
                 Dataset
        ┌────────────────────────┐
        │ kpis.jsonld            │
        │ Croissant JSON-LD      │
        │ CSV tables             │
        └───────────┬────────────┘
                    │
                    ▼
          ┌───────────────────┐
          │  spec_loader.py   │
          │ Metadata loading  │
          │ DatasetSpec       │
          │ CroissantSpec     │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │  kpi_catalog.py   │
          │ Load tables       │
          │ Build KPI panel   │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ expr_compiler.py  │
          │ Compile KPI       │
          │ expressions       │
          └─────────┬─────────┘
                    │
                    ▼
          ┌──────────────────────┐
          │  weakest_pillar.py   │
          │ KPI grading          │
          │ Pillar analysis      │
          │ KPI selection        │
          └──────────┬───────────┘
                     │
          selected_kpis.csv
          panel.csv
          run.json
                     │
                     ▼
          ┌──────────────────────┐
          │   causal_graph.py    │
          │ Causal discovery     │
          │ Stability selection  │
          │ Graph construction    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │      gcm_fit.py      │
          │ DoWhy GCM refinement │
          │ Arrow strength       │
          │ DAG falsification    │
          └──────────┬───────────┘
                     │
                     ▼
                Causal Graph
```

## Modules

### `spec_loader.py`

Loads and normalizes the JSON-LD metadata used by the pipeline.

It reads:

* `kpis.jsonld`
* the Croissant JSON-LD file

`kpis.jsonld` provides the grain, table mappings, pillars, collinear pairs, context variables, and KPI definitions. The loader converts these into typed dataclasses such as `DatasetSpec`, `KPISpec`, `ContextVar`, and `PillarSpec`.

The Croissant loader extracts CSV files, columns, and declared data types. `discover_metadata()` locates `kpis.jsonld` and finds the Croissant file containing `distribution` and `recordSet`.

**Main responsibility:**
`JSON-LD metadata → DatasetSpec + CroissantSpec`

---

### `expr_compiler.py`

Evaluates the machine-readable KPI expressions defined in `kpis.jsonld`.

The expressions are processed using an AST-based compiler.

Supported operations include:

* table references such as `table.column`
* KPI references
* arithmetic operations
* comparisons
* boolean operations
* `days()` and `hours()`
* `date()`, `month()`, `year()`
* `lower()`
* `notblank()`
* `isnull()`
* `if_else()`
* `coalesce()`
* `sum()`, `mean()`, `min()`, `max()`
* `count()`, `nunique()`, `any()`, `all()`, `mode()`

The result of `compile_expression()` is a Series or scalar that can be merged into the KPI panel.

**Main responsibility:**
`KPI expression → calculated Series/value`

---

### `kpi_catalog.py`

Connects the metadata, source tables, and expression compiler to build the KPI panel.

It provides three main operations:

#### Context variables

`compile_context_vars()` evaluates context-variable expressions and adds them to the working panel.

#### KPIs

`compile_all_kpis()` evaluates KPI expressions in dependency order. KPIs whose parent is not yet available are deferred until the dependency becomes available.

#### Panel

`build_panel()`:

1. Finds the grain column.
2. Creates one row per unique grain value.
3. Compiles context variables.
4. Compiles KPIs.
5. Merges the calculated values into the panel.

The resulting panel contains the grain, context variables, and KPI columns.

The module also contains `load_tables()`, which loads the CSVs specified by `DatasetSpec.tables`, and `infer_date_columns()` for date-column handling.

**Main responsibility:**
`Source tables + metadata + expressions → KPI panel`

---

### `weakest_pillar.py`

Performs the KPI and weak-pillar analysis before causal discovery.

The module:

1. Loads metadata and tables.
2. Resolves the target KPI.
3. Determines the KPI set to analyze.
4. Builds the grain-level panel.
5. Calculates KPI coverage and metadata.
6. Calculates KPI goodness.
7. Estimates KPI influence on the target.
8. Calculates pillar influence and performance.
9. Calculates pillar priority.
10. Repeats the priority calculation using bootstrap resamples.
11. Identifies weak pillars.
12. Marks the primary KPI candidates for causal discovery.

The target can come from the `--target-kpi` argument, the recommend-list, or the first non-diagnostic KPI in the catalog.

For a KPI, goodness is based on its percentile rank, with the direction reversed for lower-is-better KPIs.

Pillar priority is calculated as:

```text
Priority = Influence × (1 − Performance)
```

The weak-pillar result is then used to determine the KPI candidates passed to `causal_graph.py`.

### Outputs

Main outputs:

```text
selected_kpis.csv
panel.csv
run.json
```

Additional artifacts include:

```text
coverage.csv
kpi_meta.csv
selected_kpis.json
kpi_influence.csv
pillar_influence.csv
pillar_summary.csv
segment_priority.csv
bootstrap_priority.csv
state_priority.csv
weak_pillars.json
figures/
```

These outputs are defined directly by the module.

**Main responsibility:**
`KPI panel → weak pillars → selected causal-discovery KPIs`

---

### `causal_graph.py`

This is the main GLINT causal discovery engine.

The module is organized into discovery stages.

#### Stage 0 — Schema

Builds the variable schema using the assigned tiers.

Variables are classified into roles such as context, mediator, and outcome.

Tier constraints are also used to determine forbidden directions.

#### Stage 1 — Environments

Creates the environments used for causal discovery.

It builds:

* natural strata from tier-1/context variables
* downsampled environments

The environment pool combines the natural strata and downsampled environments.

#### Stage 2 — Candidate Parents

Generates candidate parents for each target variable.

Candidate filtering uses:

* tier ordering
* forbidden edges
* collinear pairs
* post-treatment restrictions
* known edges

The candidates are then pruned using the 1-SE rule.

The full Stage 2 path also applies the FCI-based confounder screen.

#### Stage 3 — Mechanism Transfer and Parent Selection

Tests candidate parent sets using predictive performance and environment transfer.

Parent sets are evaluated using the available environments, and the selected parent sets are passed to the next stage.

#### Stage 4 — Direction and Graph Assembly

Candidate relationships are oriented using:

1. forbidden-edge rules
2. tier ordering
3. cross-validation comparison when necessary

The graph is then assembled as a directed graph.

Known edges can be forced into the graph, and cycles are removed by removing an edge from detected cycles.

#### Stage 5 — Residual and Refutation Checks

The graph is checked using:

* residual-parent correlations
* placebo checks
* 30% subset stability checks

The residual score is based on the mean absolute correlation between residuals and parents.

#### Stage 6 — Stability Selection

The discovery process is repeated over bootstrap samples.

The bootstrap path uses a lighter Stage 2 without FCI to reduce the cost of repeated discovery.

Bootstrap replicates are independent and can be executed across worker processes. The worker initialization explicitly copies the dataset-specific discovery settings into each worker.

#### Stage 7 — Experiment Ranking

The module also contains experiment-ranking functionality for relationships that remain undecided.

### Final graph

The main output is:

```text
edges.csv
```

with:

```text
source
target
confidence
strength
```

Additional artifacts include:

```text
edges.json
metrics.json
glint_summary.csv
schema.csv
discovery_panel.csv
graph.png
bootstrap cache
figures/
```

The causal graph entry point reads:

```text
selected_kpis.csv
panel.csv
run.json
```

from `weakest_pillar.py`.

**Main responsibility:**
`Selected KPI panel → causal graph`

---

### `gcm_fit.py`

Provides the DoWhy GCM layer used after the sklearn-based causal discovery screen.

It is imported on the full-data path and is not used inside the bootstrap workers.

It provides:

* local structural causal model fitting
* mechanism R² calculation
* causal-model evaluation
* environment transfer evaluation
* parent-set pruning
* parent-set refinement
* graph-level GCM fitting
* normalized arrow strengths
* DAG falsification

`fit_local_gcm()` creates a local parent → child SCM and assigns causal mechanisms automatically.

`gcm_transfers()` evaluates whether the pooled mechanism continues to fit across environments and checks the invertibility assumption.

`causal_graph.py` can run with GCM enabled or disabled. When disabled, the graph is assembled from the sklearn discovery path without importing `gcm_fit.py`.

**Main responsibility:**
`Discovered graph → GCM refinement, scoring and falsification`

---

## Repository Structure

```text
.
├── spec_loader.py
├── expr_compiler.py
├── kpi_catalog.py
├── weakest_pillar.py
├── causal_graph.py
└── gcm_fit.py
```

## Execution Flow

### Step 1 — Build KPI panel

```bash
python weakest_pillar.py <dataset_path> <output_path>
```

Optional arguments:

```bash
--artifacts-path
--recommend-list
--target-kpi
--profile
```

The script expects the dataset directory to contain `kpis.jsonld`, a Croissant JSON-LD file, and the CSV files referenced by the metadata.

### Step 2 — Run causal discovery

```bash
python causal_graph.py <weakest_pillar_output_path> <output_path>
```

Optional arguments:

```bash
--artifacts-path
--profile
--n-jobs
--gcm
--no-gcm
```

`--n-jobs` controls the bootstrap worker processes, while `--gcm` controls the DoWhy GCM refinement/scoring path.

## Complete Flow

```text
kpis.jsonld
     │
     ├───────────────┐
     │               │
     ▼               ▼
spec_loader.py   Croissant JSON-LD
     │
     ▼
DatasetSpec
     │
     ▼
kpi_catalog.py
     │
     ├── load tables
     ├── context variables
     └── KPI compilation
              │
              ▼
       expr_compiler.py
              │
              ▼
          panel.csv
              │
              ▼
     weakest_pillar.py
              │
              ├── KPI goodness
              ├── KPI influence
              ├── pillar performance
              ├── pillar priority
              └── bootstrap priority
              │
              ▼
      selected_kpis.csv
          run.json
              │
              ▼
       causal_graph.py
              │
              ├── schema / tiers
              ├── environments
              ├── candidate parents
              ├── parent selection
              ├── graph orientation
              ├── residual/refutation checks
              ├── bootstrap stability
              └── experiment ranking
              │
              ▼
          gcm_fit.py
              │
              ├── GCM refinement
              ├── mechanism evaluation
              ├── arrow strength
              └── DAG falsification
              │
              ▼
          edges.csv
```
