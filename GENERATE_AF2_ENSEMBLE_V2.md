# ensemblr-generate

Template-directed adaptive AF2 ensemble generator. Runs ColabFold iteratively, using PCA of the growing structure pool to identify under-sampled regions of conformational space and target them with custom structural templates.

The `ensemblr-generate` command is installed with the package (see the README).
`generate_af2_ensemble_v2.py` in the repo root runs the same thing.

## Quickstart

ColabFold is not installed with ensemblr. Install
[localcolabfold](https://github.com/YoshitakaMo/localcolabfold), then either put `colabfold_batch`
on your `PATH` or tell ensemblr where it is:

```bash
export COLABFOLD_DIR=/path/to/localcolabfold    # or pass --colabfold-dir
```

`mkdssp` should also be on `PATH` (it is in the conda environment). Without it the PCA uses all Cα
atoms and ignores `--pca-exclude`/`--pca-include`. Check the dependency report at the top of the log.

From a target directory containing `input.fasta`:

```bash
ensemblr-generate --dry-run   # check the probe schedule
ensemblr-generate             # writes synthetic_ensemble/
```

The defaults are the production settings: pLDDT > 80 gated on the secondary-structure core
(`--selection-quality`), 25 probe seeds per depth, 2000 round-0 predictions, then 50 adaptive
rounds of 200 at a fixed depth (the probe optimum), PCA refit every 5 rounds, no coverage early stop.
Rerun the same command to resume, or with a larger `--max-rounds` to extend.

Common variations:

```bash
ensemblr-generate --score-metric iptm               # multimer
ensemblr-generate --quality-thresh-pct 0.9          # never reaches pLDDT 80
ensemblr-generate --pca-exclude "resid 1:120"       # ignore a domain
ensemblr-generate --fill --fill-interpolate --dry-run   # fill PC1 gaps afterwards
```

---

## Pipeline overview

### 1. MSA pre-fetch

The full MSA is fetched once from the ColabFold server (`--msa-only`) and cached in `output_dir/msa/`. All subsequent ColabFold runs read from this cache, so no repeated server queries.

### 2. Two-phase probe

Identifies the MSA depth that maximises structural diversity while maintaining quality.

**Utility function:**  `U(d) = yield(d) × intra_RMSD(d)`

Rewards depths that produce both high-pLDDT structures (yield) and structurally diverse predictions (intra_RMSD).

- **Phase 1 — coarse sweep:** powers-of-2 depths from `2^msa_start` to the natural MSA ceiling, `n_probe` seeds × 1 model each.
- **Phase 2 — fine sweep:** geometrically-spaced intermediate depths around the coarse utility peak, at `2^(1/fine_steps)` spacing.
- **Output:** `optimal_depth` (argmax U), `viable_window` (all depths with U ≥ `viable_fraction` × U_peak).

The natural MSA ceiling (all available sequences) is always included in the coarse sweep as an anchor for context but is excluded from the viable window.

### 3. Round 0

Initial seeding with no templates, pure MSA predictions.

**Default (no annealing):** budget distributed evenly across all depths in `viable_window`.

**With `--anneal-round0`:** schedule is `viable_window` depths followed by a geometric extension from `viable_window[-1]` up to the MSA ceiling. Seeds the ensemble with both diverse low-depth structures and high-coevolution-constrained end-state conformations. The number of extension steps is determined by `--round0-step-density` and the total budget.

### 4. Adaptive rounds

Iterative template-directed sampling targeting under-represented regions of conformational space.

Each round:
1. **PCA** on secondary-structure-element Cα atoms (DSSP selection from highest-pLDDT probe structure; falls back to all Cα if DSSP unavailable). Re-run every `--pca-interval` rounds; between re-runs, structures are projected onto cached eigenvectors.
2. **Grid:** partition PC1–PC2 into an N×N grid (N auto or `--n-grid`). Identify sparse cells (count < `--min-count-per-cell`).
3. **Representative selection:** for each sparse cell, select the highest-pLDDT member as the template.
4. **ColabFold:** one job per targeted cell, passing the representative as `--custom-template-path`. Templates provide a weak geometric prior; `--use-dropout` keeps predictions stochastic.
5. **Depth control (dynamic temperature):** MSA depth adjusts each round to hit a target template hit-rate (fraction of new structures landing in the target cell). If hit rate is too low, depth increases (stronger template influence); if too high, depth decreases (more stochasticity).

Stops when coverage ≥ `--coverage-threshold` or `--max-rounds` is reached.

### Output

```
output_dir/
├── msa/                  # cached MSA (.a3m)
├── rounds/
│   ├── probe/            # probe ColabFold outputs
│   ├── round_0/          # round-0 ColabFold outputs
│   └── round_N/          # adaptive round outputs
│       └── cell_I_J/
│           ├── template/ # representative PDB
│           └── msa_*/    # ColabFold output
├── structures/           # symlinks to all accepted PDBs
├── pca_state.npz         # PCA model + grid edges (for restarts)
├── stats.json            # per-round statistics
└── plddt_dict.json       # {filename: pLDDT} for all accepted structures
```

### Restart behaviour

Runs are fully restartable. On restart:
- Existing structures are reloaded from `plddt_dict.json`
- PCA state (eigenvectors, grid edges, round counters) is reloaded from `pca_state.npz`
- ColabFold directories with a `log.txt` are skipped automatically
- The `rounds_since_pca` counter is persisted, so `--pca-interval` scheduling is preserved across restarts

### Handoff to analysis

```python
from ensemblr import Ensemble
e = Ensemble('output_dir/structures/')
# plddt_dict is returned by run_adaptive_ensemble_v2(), or reload from file:
import json
plddt_dict = json.load(open('output_dir/plddt_dict.json'))
```

---

## All flags

### Positional

| Argument | Description |
|---|---|
| `fasta` | Input FASTA file (default `input.fasta`) |
| `output_dir` | Root output directory (default `synthetic_ensemble`) |

### MSA depth range

| Flag | Default | Description |
|---|---|---|
| `--msa-start EXP` | `1` | log₂ of smallest probe depth (1 → 2:4) |
| `--msa-ceiling N` | auto | Cap on MSA depth; defaults to natural MSA size |

### Probe

| Flag | Default | Description |
|---|---|---|
| `--n-probe N` | `25` | Seeds per depth in probe rounds |
| `--fine-steps N` | `4` | Geometric steps per octave in fine probe (step ≈ 1.19×, giving 3 new depths per octave) |
| `--viable-fraction F` | `0.5` | Depths with U ≥ F × U_peak enter the viable window |
| `--min-mean-quality F` | 50 (pLDDT) / 30 (pTM, ipTM) | Exclude probe depths with mean quality below this |

### Sampling budget

| Flag | Default | Description |
|---|---|---|
| `--budget-initial N` | `2000` | Total predictions for round 0 |
| `--budget-per-round N` | `200` | Predictions per adaptive round |
| `--n-models N` | `5` | AF2 models per seed |
| `--seeds-per-cell N` | `3` | Seeds per targeted sparse cell; `n_cells/round = budget_per_round // (seeds_per_cell × n_models)` |
| `--num-recycle N` | `0` | AF2 recycling iterations (0 = single forward pass) |
| `--max-rounds N` | `50` | Maximum adaptive rounds after round 0 |

### Quality filters

| Flag | Default | Description |
|---|---|---|
| `--score-metric M` | `plddt` | `plddt`, `ptm`, `iptm` or `iptm_composite` (0.8 ipTM + 0.2 pTM); all on a 0–100 scale |
| `--quality-thresh F` | 80 (pLDDT) / 50 (pTM, ipTM) | Minimum quality to accept a structure |
| `--quality-thresh-pct PCT` | off | Set the threshold to PCT × the best coarse-probe depth's mean quality instead (e.g. `0.9`) |
| `--selection-quality` / `--no-selection-quality` | on | Gate pLDDT on the PCA selection (DSSP secondary-structure core) rather than the whole chain. pLDDT only |
| `--pca-exclude SEL` | none | MDAnalysis selection removed from the PCA selection and quality gating (e.g. `"resid 1:120"`) |
| `--pca-include SEL` | none | MDAnalysis selection always added to the PCA selection (e.g. a functional loop) |
| `--relaxed` | off | Collect Amber-relaxed PDBs (requires `--amber` in `--extra-flags`) |

### Coverage analysis

| Flag | Default | Description |
|---|---|---|
| `--min-count-per-cell N` | `3` | Grid cells with fewer structures are sparse |
| `--coverage-threshold F` | `1.0` | Stop when this fraction of occupied cells are well-sampled. 1.0 disables the early stop, leaving `--max-rounds` as the limit |
| `--n-grid N` | auto | Override grid resolution (N×N); auto = clamp(√(structures/min_count), 5, 30) |
| `--n-pca-components N` | `3` | Number of PCA components computed (grid uses first 2) |

### Round-0 annealing

| Flag | Default | Description |
|---|---|---|
| `--anneal-round0` | off | Extend round-0 beyond the viable window up to the MSA ceiling |
| `--round0-step-density N` | `10` | Controls extension granularity: `total_steps = budget // (N × n_models)`, `extension_steps = total_steps − len(viable_window)`. Seeds per step is recalculated from actual steps so the full budget is used. |

### PCA schedule

| Flag | Default | Description |
|---|---|---|
| `--pca-interval N` | `5` | Re-run PCA every N adaptive rounds (1 = every round) |
| `--no-stable-grid` | off | Recompute grid edges every round; default holds edges fixed for the duration of each PCA interval for stable cell indices |

### Dynamic temperature

Off by default: adaptive rounds run at the probe-optimal depth. When turned on, it controls MSA depth in adaptive rounds to maintain a target template hit-rate (fraction of new structures landing in the template's target cell). In practice this did not help.

| Flag | Default | Description |
|---|---|---|
| `--hit-rate-target F` | `0` | Target hit rate (e.g. `0.5`). If below target, depth increases (stronger template influence); if above 2×target, depth decreases. `0` disables dynamic temperature and uses `--anneal-rate` instead. |
| `--temp-step F` | `1.5` | Multiplicative depth adjustment factor per round |
| `--anneal-rate F` | `1.0` | Fallback depth multiplier per round when `--hit-rate-target 0` (1.0 = fixed depth) |

### Fill mode (post-hoc gap filling along one PC)

Runs against a finished run instead of the main pipeline. Builds synthetic templates by blending real
structures so their projection lands in a gap (or beyond an edge) along one PC, then runs ColabFold
with them. The blends are never added to the ensemble.

| Flag | Default | Description |
|---|---|---|
| `--fill` | off | Run fill mode. Needs `--fill-interpolate` and/or `--fill-extrapolate` |
| `--fill-axis N` | `1` | PC (1-indexed) to fill along |
| `--fill-interpolate` | off | Target interior gaps |
| `--fill-extrapolate` | off | Target one step beyond each sampled extreme |
| `--fill-extrap-direction` | `both` | `low`, `high` or `both` |
| `--fill-min-gap X` | auto | Minimum gap (PC units) worth filling; auto = `--fill-min-gap-percentile` (99) of nearest-neighbour spacing. `--dry-run` prints the percentiles |
| `--fill-max-points-per-gap N` | `1` | Fill targets per large gap |
| `--fill-extrap-step F` | `1.0` | Extrapolation distance in units of the last gap |
| `--fill-max-targets N` | all | Cap on targets per round (largest first) |
| `--fill-rounds N` | `1` | Fill rounds; each re-derives targets, stops early if nothing new is accepted |
| `--fill-depth P:U` | probe optimum | MSA depth for fill runs |
| `--fill-seeds N` | `5` | Seeds per target |

### Miscellaneous

| Flag | Default | Description |
|---|---|---|
| `--pdb-templates` | off | Allow ColabFold to query PDB templates in probe and round-0. Adaptive rounds always use `--custom-template-path` regardless. |
| `--extra-flags STR` | `''` | Extra flags passed verbatim to `colabfold_batch` (e.g. `"--amber --relax-max-iterations 200"`) |
| `--dry-run` | off | Print settings and commands without running ColabFold. Probe uses synthetic values so the full logic is exercised. |
| `--verbose` / `--no-verbose` | on | Print an ASCII N×N coverage grid after each adaptive round (`·` empty, `░` sparse, `█` well-sampled, `*` targeted this round) |

---

## Understanding MSA depth

Depths are expressed as `paired:unpaired` (e.g. `8:16`). For monomers, paired sequences = 0, so `8:16` means 8 unpaired sequences. The probe sweeps these in powers of 2.

- **Low depth** → high stochasticity → broad exploration of conformational space
- **High depth** → low stochasticity → refinement near high-confidence basins
- **Ceiling** (`n_total:n_total×2`) → uses the full available MSA → structures close to solved/predicted reference states

The utility function U = yield × intra_RMSD identifies the sweet spot where predictions are both high-quality and diverse.

---

## Tips

- Start with `--dry-run` to verify the probe depth schedule looks sensible before committing compute.
- Lower `--n-probe` (e.g. 5) and `--max-rounds` for quick tests.
- If the probe marks every depth noisy, or the optimum sits at the ceiling, the threshold is too strict for this protein: try `--quality-thresh-pct 0.9`.
- If adaptive rounds sit at a 0% hit rate with coverage flat, the remaining sparse cells are edge cells AF2 won't populate. Stop, or try `--fill`.
- Relaxed structures require `--relaxed --extra-flags "--amber"`.

---

## Future / known improvements

### Multi-template targeting

ColabFold's `--custom-template-path` reads all PDB files in the directory, up to AF2's internal limit of 4 templates. Currently `_prepare_template_dir` copies a single representative (highest-pLDDT member of the target cell) as `tmpl.pdb`. A natural extension would be to pass the top-N members from the cell, providing multiple geometric priors per prediction.

Practical notes for implementation:
- ColabFold uses the first 4 characters of each PDB filename as the template PDB code, so files must have distinct 4-character names (`tmp1.pdb`, `tmp2.pdb`, etc.).
- `_prepare_template_dir` currently uses `shutil.copy2` — switching to `os.symlink` (pointing at `os.path.realpath(src)`) would avoid the disk overhead of copying a PDB per cell per round.
- Whether multiple templates improve targeting over a single high-pLDDT representative is untested.
