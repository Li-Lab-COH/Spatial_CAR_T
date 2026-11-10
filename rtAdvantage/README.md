# What does radiation buy the CAR-T? — direct and abscopal

The `spp1Analysis` project asked one cell type's question and answered it thoroughly. This one
asks the treatment-level question that motivated the experiment:

> **Adding RT on top of Cy + CAR-T — what does it change in the tumour microenvironment, and
> does the change also appear in the *unirradiated* (abscopal) tumour?**

Same data, same HPC kernel (`spatial_gpu_py311`), same rules: helpers **defined inline** (the
kernel cannot see these local files), every notebook ends in a printed
`SUMMARY — report these back` block, and at n = 2–4 a side the evidence is **effect size +
sign-agreement across the two CAR-T backgrounds**, never a p-value.

---

## Start here: what `spp1Analysis` already ruled out

Four RT questions are closed. Do not reopen them without a new reason.

| Closed question | Answer | Where |
|---|---|---|
| Does RT shift SPP1-TAMs toward fatty-acid processing? | **No** — flat under 4 independent population definitions, in a design powered to catch 36% of the Spp1-high-vs-low state effect. | `09` |
| Does RT leave *any* reproducible cell-intrinsic signature? | **No, in any compartment** — macs, monocytes, tumour, neutrophils, cDC, all cells. Cross-background logFC r = −0.48..+0.01, permutation p = 0.30..0.98. Cy+CAR-T through the identical pipeline: r = +0.76..+0.89. | `09 §D` |
| Does RT change SPP1-TAM abundance, clustering, or coupling *strength*? | **No** — 3 of 4 spatial readouts null, two of them fail their own contrast-tier control. | `08 §A/§B/§D` |
| Does RT change the spatial **range** of SPP1-TAM↔niche coupling? | **Yes** — decays to ~37% of near-field by 100–150 µm without RT, persists at ~68–98% with it. 3 of 4 controls pass. **Systemic** — abscopal ≥ irradiated (0.98 vs 0.68). | `08 §C` + `08b` |

**The shape of that result is the thesis of this project.** The one RT effect that survived is
*geometric and systemic*: nothing about what a cell expresses, something about how the tissue is
arranged, and it is at least as strong in the tumour that was never in the beam. So this project
looks where that kind of effect lives — **composition and geometry**, not expression — and it
treats the abscopal tumour as the primary arm rather than the afterthought.

---

## Two modalities, two different jobs

This is the single most important design fact here, and it is measured, not assumed
(`00` reproduces every number):

| | single cell (`annotation_v4.h5ad`) | spatial (`1_C2l_annotations_400`) |
|---|---|---|
| cells | 93,548 × 22,949 genes | 2,088,557 × 19,059 genes |
| samples | 29 (5 arms × 2 sites × 3 mice; `CyTAG72_Tu1` has 2) | 32 tissues |
| myeloid | ~87,000 | ~460,000 |
| **all lymphocytes** | **1,844 (2.0%)** — CD8_T 513, NK 1,008, CD4_T 85, Treg 79, NKT 106, B 53 | **131,520** — CD8_T 45,824, NK 22,924, CD4_T 10,085, Treg 5,155, B 40,422 |
| Cancer_cell | 6,614 | 363,019 |

**The effector arm is only readable spatially.** 513 CD8 T cells across 29 samples is ~18 per
sample — enough for a composition fraction with wide error bars, not enough for a state, a
subtype, or an exhaustion axis. The single-cell object is a myeloid-and-tumour object. Plan
accordingly: **single cell answers composition and tumour-cell state; spatial answers the
lymphoid compartment and all geometry.**

The reverse constraint is just as sharp. Spatial **tissue-level fractions are not usable** as an
RT readout — per-tissue counts span two to three orders of magnitude (CD8_T 12 → 9,440;
M1_like_Mac 0 → 4,755) because sections differ in how much tumour vs stroma vs necrosis they
caught. Spatial contributes **within-tissue** quantities (distances, gradients, neighbourhoods,
range statistics), which are internally normalised and therefore immune to section content.
This is the same reason `08` worked and `06`'s burden contrast did not.

## CAR-T cells cannot be found by their transgene

`var_names` carries two spiked-in construct features, `PSCA` and `TAG72` (both have no
`chromosome` — they are added references, distinct from mouse `Psca`). Their **specificity is
perfect and their sensitivity is nil**:

- 8 transgene-positive cells in the whole object — 7 `CD8_T`, 1 `CD4_T`, 0 in every other type.
- `PSCA` reads appear only in PSCA arms, `TAG72` only in TAG72 arms, **zero in NoTx**.

So the construct is real and correctly assigned, and it identifies 8 cells. **Do not build a
CAR-T-vs-endogenous-T analysis on it.** Transferred T cells have to be treated as part of the
CD8 compartment. Mouse `Psca` is separately near-undetectable (313 cells, 315 UMI total, only 4
of them `Cancer_cell`), so *"does RT raise the CAR target antigen"* is also unanswerable here —
it needs protein, or a probe panel that actually covers it.

## The hemisphere split (`docs/B_subsample_barcodes.xlsx`) — what it is and is not

All 60,026 barcodes join cleanly to spatial `cell_id` (100% match). Verified geometry:

| | `RTCyPSCA_2_3` | `RTCyPSCA_2_4` |
|---|---|---|
| cells in file / in tissue | 35,264 / 112,486 (**31%**) | 24,762 / 65,157 (**38%**) |
| split axis | clean cut in **y** at 5,092 (zero overlap) | clean cut in **y** at 12,490 (zero overlap) |
| `hi` / `lo` | 17,632 / 17,632 | 12,381 / 12,381 |

So it is a **median cut in y through a sub-region** of each section, exactly balanced by
construction. Two consequences that have to be carried into every use of it:

1. **`hi + lo` is not the tissue.** 62–69% of each section's cells are outside the selected
   region, and the selected region is compositionally unlike the whole (in `2_3`, Cancer_cell is
   1.3% inside vs 11.0% across the tissue; M1_like_Mac 13.0% vs 4.2%). Any statistic compared
   between a split tissue and an unsplit one is comparing different tissue content.
2. **`hi` and `lo` are two different regions, not two replicates.** They diverge sharply —
   `2_3` M1_like_Mac 6.4% (hi) vs 19.5% (lo); `2_4` Cancer_cell 4.0% (hi) vs 12.9% (lo).

**Therefore it does not raise n.** Two halves of one section from one mouse carry one mouse's
worth of biological information; counting them as n = 4 inflates the apparent replicate count
while *adding* between-region variance that the RT contrast does not want. A p-value computed
over halves is wrong in a way that always looks favourable.

**What it is genuinely good for, and how this project uses it:** as the **within-tissue
reproducibility control** on geometric readouts. Compute `08`-style range/gradient statistics
separately in `hi` and `lo`; if the abscopal effect is real it should appear in both halves of
both sections. At n = 2 tissues that is the strongest internal control available, and it is
worth more than a fake n = 4.

**If the goal is abscopal n, the lever already in the data is the other background.** Abscopal
tissues: `RTCyPSCA_2_*` (2) **plus `RTCyT72_2_*` (3)** = 5, against a non-RT Tu2 baseline of
`CyPSCA_2_*` (4) + `CyT72_2_*` (2) = 6. Two backgrounds, so the claim rests on sign-agreement.

---

## Design, and the contrasts it supports

Mouse `Tu1`/`loc 1` = **irradiated**; `Tu2`/`loc 2` = **abscopal**, same animal. There is **no
RT-alone arm** — every irradiated mouse also received Cy + CAR-T — so every contrast is worded
*"adding RT on top of Cy + CAR-T"*, never *"RT vs untreated"*.

| # | contrast | isolates | caveat |
|---|---|---|---|
| 1 | `RTCy* Tu1` vs `Cy* Tu1` | RT effect at the irradiated site | between-animal |
| 2 | **`RTCy* Tu2` vs `Cy* Tu2`** | **abscopal / systemic RT effect** | between-animal; **the primary contrast** |
| 3 | `(Tu1−Tu2)_RT` − `(Tu1−Tu2)_nonRT` | local beam effect, difference-in-differences | within-animal; **mandatory** — the non-RT arms show the same Tu1/Tu2 gap, so it is a *site* effect unless differenced out |
| 4 | treated vs `NoTx` | **pipeline check only, never a claim** | see the rule below |

> **`NoTx` is not a comparator.** Untreated tumours differ from treated ones in nearly
> everything — tumour is 25-42% of cells vs 4-6%, macrophages 24-29% vs 46-60% — so a difference
> against `NoTx` says the animal got treated, not what radiation did. It is used here for exactly
> one purpose: to confirm a measurement can detect a perturbation it is known to contain. Every
> substantive claim rests on **treated-vs-treated** comparisons. (`NoTx` is also unusable in
> spatial at all — 7 of 8 tissues have <50 macrophages.)

**Arm-name trap:** spatial spells the non-specific arm `CyT72`/`RTCyT72`; single cell spells it
`CyTAG72`/`RTCyTAG72`. Mismatching silently drops every CAR-T-background contrast. `00` writes a
single mapping table that the later notebooks read rather than re-typing.

---

## Run order

| Notebook | Answers | Status |
|---|---|---|
| `00_design_and_detectability.ipynb` | *What can this dataset say about RT at all?* Rebuilds both designs, audits the transgene, characterises the hemisphere split, and prints a per-cell-type **minimum-detectable-effect** table. | **Run first.** Everything downstream reads its CSVs. |
| `01_composition_sc.ipynb` | *Does RT change what the TME is made of — and does it change it in the abscopal tumour?* CLR composition, DiD, cross-background concordance against the treated-vs-NoTx benchmark. | **Written and dry-run locally — see below.** Re-run on the kernel to confirm. |
| `02_pairwise_cell_counts_sc.ipynb` | *Every group against every other group, every cell type separately, nothing collapsed.* 45 pairs x 18 cell types = 810 comparisons, unadjusted p on two scales, plus the Tu1/Tu2 site control and a ranked DE-follow-up shortlist. | **Written and dry-run — see below.** This is the exhaustive pass `01`'s summary statistic could not substitute for. |
| `02b_composition_deep_dive.ipynb` | *Why does the pairwise table look the way it does?* Variance decomposition, Aitchison PCA + PERMANOVA, the `CAR x RT` interaction and its four controls, 10x10 fold-change grids, volcanoes, per-mouse small multiples for all 18 cell types, cell-type co-variation, effect size by contrast type, equal-depth downsampling. **10 figures, 9 tables.** | **Written and dry-run.** |
| `03_tumour_state_sc.ipynb` | *Does RT change the tumour cell — antigen presentation, IFN response, death-receptor sensitivity, senescence/SASP?* Tumour pseudobulk (summed raw counts), 27 programs, exact-relabelling nulls, random-program control, within-mouse DiD, ambient-RNA check. | **Written and run locally (2026-10-22), ~30 s.** See below. |
| `04_spatial_infiltration.ipynb` | *Does RT change where CD8/NK sit relative to tumour — infiltration vs exclusion?* Within-tissue geometry in **true µm**, internally normalised, lymphoid-matched control (vs B cells), label-stringency and leave-one-out checks. | **Written and run locally (2026-10-22), ~20 s.** Reads only obs + coordinates. |
| `05_abscopal_geometry.ipynb` | *Does `08 §C`'s RT range effect survive true µm, and does any abscopal effect reproduce in both hemispheres of both `RTCyPSCA_2_*` sections?* Streams `X` once to score (cached to `05_cell_scores.npz`), correlograms with `08b`'s controls, hemisphere test against a half-to-half noise null. | **Written and run locally (2026-10-22), ~4 min first run, ~1 min cached.** |

### `03`-`05` (local runs, 2026-10-22) — provisional, one execution each

**Plain version: no RT advantage shows up in tumour-cell state, in effector geometry, or in a
reproducible abscopal geometry. And the one RT result that had survived (`08`'s longer coupling
range) does not survive true microns: where its direction reappears, the niche field itself got
coarser, which `08b`'s control was meant to exclude.**

**`03` — tumour-cell state** (6,614 Cancer_cells, 29 pseudobulks, 27 programs scored):
- Yardsticks: treated-vs-NoTx median |effect| 0.68 SD, p<0.05 in 35% (MHC-I/II, ISGs, IFN up
  4-6 SD); the CAR axis 0.72 SD. **RT: 0.24 SD, p<0.05 in 0% between animals, 0% in the
  within-mouse DiD**, sign agreement below chance (22%).
- **No RT main-effect lead** (sign-agree + p<0.10 + beats 90% of random programs): 0. cGAS-STING,
  TRAIL and the curated death-receptor set have too few genes detected in tumour cells to score.
- Five **interaction** leads — PSCA up, TAG72 down — but the top three (MHC-II both sites,
  MHC-I machinery at Tu1) are 92-100% **myeloid-dominant genes** (`B2m`, `H2-K1`, `Cd74`...).
  Ambient-suspect, not proven ambient. The other two (abscopal p53, apoptosis) are at
  p_inter 0.055/0.09 with 27 programs tested.

**`04` — effector geometry** (true µm: cell spacing 6.6-9.8 µm, so `px × 0.34` is right):
- **Design limit:** for CD8 the TAG72 irradiated contrast is **not estimable** (1 v 1 usable
  tissues), so the irradiated site is PSCA-only and the site x RT DiD is PSCA-only.
- **Face-validity warning:** every lymphoid label — B and Treg most of all — sits in more
  tumour-rich surroundings than a random non-tumour cell (reference is ~60% neutrophils), also
  under `c2l_strict`. "CD8 is tumour-proximal" is therefore not cytotoxic infiltration by
  itself; `prox30_vs_B` is the effector-specific readout.
- **RT main-effect leads: 0.** The only signal is again an **interaction** at the abscopal site:
  CD8 tumour proximity PSCA −0.50 vs TAG72 +0.91 log2 (p_inter 0.06), which vanishes against B
  cells (`prox30_vs_B` PSCA −0.00, TAG72 +0.43). In the figure RT moves the two backgrounds
  *toward each other* — the same convergence `02b` saw in composition.
- `core` infiltration needed a 20% (not 50%) tumour threshold: treated sections are 4-6% tumour.

**`05` — the `08` range claim in true µm, and the hemisphere test:**
- `08`'s bands were ~4.4x too short physically: its "0-25 µm" near field is 0-110 µm and its
  "100-150 µm" is ~440-660 µm. A true 0-25 µm band is not estimable (too few non-myeloid
  cells around a macrophage).
- At 440-660 µm, more far-band coupling with RT in **both** backgrounds shows up in **1 of 4**
  tier x site cells (damage tier, irradiated: PSCA +0.03, TAG72 +0.08 r, p_main 0.09) with flat
  amplitude — **but the niche field's own 440-660 µm autocorrelation also rises with RT in both
  backgrounds** (damage +0.07, metabolic +0.12, p 0.023). That is the field-control failure
  `08b` existed to rule out.
- **Abscopal TAG72 coupling is not estimable** (only 1 non-RT abscopal TAG72 section has
  macrophages; `CyT72_2_4` has 12). Abscopal PSCA: damage-tier coupling is higher with RT at
  **every** distance (near +0.14, far +0.21) — amplitude, not range — and its niche field moved
  by the same amount (+0.21).
- **Hemisphere test:** only one readout has all 4 halves on the RT side and exceeds
  half-to-half noise: metabolic far-band coupling, with r values of −0.02 to +0.03 in the halves.
  It is consistent, but too small to support an effect. CD8/NK geometry does not reproduce
  across halves (1/3, 2/3).

**What this means for the summary claim:** across composition (`01`/`02`/`02b`), tumour state
(`03`), effector geometry (`04`), and range (`05`), the reproducible structure in this dataset
is **which CAR was given**, and RT acts mainly as a `CAR x RT` interaction that pulls the two
backgrounds together, not as a main effect. Nothing here supports an RT advantage at the
abscopal site.

### First (local) runs of `01` and `02` — provisional, one execution each

**Plain version of the result: the biggest thing separating these tumours is which CAR they got,
not whether they were irradiated.**

**What `02` found** (810 comparisons, every cell type on its own, unadjusted p):

- Restricting to **treated-vs-treated** comparisons where every sample had at least 20 cells:
  187 tests, 13 at p<0.05 against ~9 expected by chance. There is not a lot here.
- Of the 45 comparisons that clear the DE-follow-up bar, **28 are PSCA-vs-TAG72 and only 9 are
  RT-vs-non-RT.** The PSCA arms carry roughly twice the `M1_like_Mac` of the TAG72 arms
  (e.g. abscopal 25.7% vs 11.3%) whether or not they were irradiated.
- **The one real RT lead:** `M1_like_Mac` at the irradiated site in the PSCA background —
  **20.2% -> 27.8%** (`CyPSCA_Tu1` -> `RTCyPSCA_Tu1`), p = 0.059 on percentage / 0.087 after the
  closure correction, **fully separated** (all three RT mice above all three controls), 343+ cells
  in every sample. It is not reproduced in the TAG72 background (13.8% -> 15.5%).
- **Nothing survives at the abscopal site.** Best `RTCyPSCA_Tu2` vs `CyPSCA_Tu2` is
  `Intermediate_Mac` at p = 0.20.
- **The site control kills the T-cell lead.** `CD8_T` is 2.5x higher in the irradiated than the
  abscopal tumour in `RTCyTAG72` (0.85% vs 0.35%, p = 0.023, fully separated) — but the *non-RT*
  CAR-T arms show the same asymmetry (`CyPSCA` 1.7x, `CyTAG72` 2.2x) while `NoTx` does not (0.94x).
  So T cells preferentially accumulate in Tu1 **because of CAR-T, not because of the beam.**
  That is a real finding, just not an RT one. Across all 18 cell types: **no Tu1/Tu2 gap is
  present with RT and absent without it.**

**What `01` found**, in the same plain terms: treating the composition as one 7- or 14-number
vector, the RT and non-RT arms do not shift in the same direction in the two CAR-T backgrounds
(irradiated r = +0.32, abscopal r = -0.47, both inside their own permutation null). The
within-mouse `(Tu1-Tu2)_RT - (Tu1-Tu2)_non-RT` contrast *does* reproduce across backgrounds
(r = +0.95 coarse / +0.66 fine, beating essentially all 200 relabellings the design permits) —
but dropping the single most influential compartment takes r from 0.95 to 0.76, and removing the
sparse compartments takes p from 0.004 to 0.055. And `02`'s per-cell-type pass shows why the
summary statistic should not be trusted alone: **it is flat while `M1_like_Mac` moves.**

**What `02b` added** — the structural answer, with numbers:

| how much of the variation each factor explains (23 treated samples, mean across cell types) | |
|---|---|
| animal-to-animal (residual) | **72.3%** |
| which CAR was given | **10.3%** |
| CAR x RT interaction | **8.7%** |
| tumour site | 5.3% |
| **radiation (main effect)** | **3.5%** |

- For `M1_like_Mac` the CAR target explains **51%** of the variance and radiation **3.6%**; for
  `M2_like_Mac`, 38% vs 2.1%. **No cell type has radiation above 25%.**
- **PERMANOVA on the whole composition, treated only:** CAR target p = 0.021, **radiation
  p = 0.597**, site p = 0.222. Samples separate by which CAR they got, not by whether they were
  irradiated.
- **Hit rate by contrast type** (>=20 cells everywhere, unadjusted p<0.05): CAR target 33% of
  comparisons (6.7x chance), radiation 11% (2.2x), tumour site 0%. Median |log2FC|: CAR target
  0.35, radiation 0.20.
- **The interaction is the interesting part.** `cart x RT` outweighs the RT main effect
  (8.7% vs 3.5%) and exceeds 15% for `M2_like_Mac`, `Cancer_cell`, `Intermediate_Mac`,
  `N1_like_Neu`, `N2_like_Neu`, `NKT`. Plotting the two backgrounds' RT effects against each
  other gives r = **-0.81** over the 7 well-counted cell types. **That is suggestive, not
  established** — four controls in §4b: permutation p = 0.051 against a null with sd 0.53,
  partial r after removing each background's baseline -0.62 (p = 0.13), and the baseline-gap
  control is consistent with the two arms simply converging. The defensible statement is the
  weak one: **do not assume the RT effect points the same way in both CAR backgrounds** — which
  means `01`'s sign-agreement rule may have been averaging a real interaction to zero.
- **Only ~5 of 18 cell types are independent** (5 PCs carry 80% of the variance; 10 pairs at
  |r| > 0.6). The 810 tests in `02` are nowhere near 810 independent looks.
- **`M1_like_Mac` survives the depth control.** Downsampling every sample to 483 cells, 300
  times: observed p = 0.059, median p across draws 0.088, p<0.10 in 60% of draws. Weakened but
  not an artifact. Its background-specificity is stark: PSCA Tu1 20.2% -> 27.8% (p = 0.059,
  separated), PSCA Tu2 25.7% -> 21.6% (p = 0.34), TAG72 Tu1 13.8% -> 15.5% (p = 0.77),
  TAG72 Tu2 11.3% -> 12.4% (p = 0.60).

**What is worth doing next, in order:**
1. **DE on `M1_like_Mac`, `RTCyPSCA_Tu1` vs `CyPSCA_Tu1`** — 343-1503 cells a sample, 3v3. The
   abundance moved; ask whether the extra cells are a different state or just more of the same.
2. **DE on `M1_like_Mac`/`M2_like_Mac`, PSCA vs TAG72** — the dominant axis in the data, and it
   has never been characterised.
3. **The effector arm, spatially** (`04`) — 45,824 CD8_T there vs 513 here.
4. **Test the interaction properly.** If RT really does different things on the two CAR
   backgrounds, that is a more interesting claim than either main effect — and the spatial
   object, with 20x the cells, is where it can be powered.

`01` is deliberately the composition notebook and not another expression notebook: **`09` closed
expression, and composition is orthogonal to it.** A compartment can double in abundance with
every cell in it transcribing identically — `09`'s null says nothing about that, and nothing in
`spp1Analysis` tested it outside the macrophage lineage.

## How a claim earns its way out of here

Inherited from `spp1Analysis`, because all three of these produce positive-looking results:

1. **Global offsets.** Anything that scales a whole sample (depth, complexity, dissociation,
   section content) moves every readout at once. → Contrasts are site-matched and, where
   possible, within-mouse; composition goes through **CLR**, so it is read against the sample's
   own geometric mean rather than a closed total.
2. **Closure.** Proportions sum to 1, so a myeloid influx mechanically deflates every other
   fraction. A "CD8 decrease" can be pure arithmetic. → Both raw proportions and CLR are
   reported; **where they disagree, the disagreement is the finding**, not a nuisance.
3. **The site effect masquerading as the beam.** `Tu1` and `Tu2` differ at baseline in the
   non-RT arms too. → Never report `Tu1 − Tu2` inside an RT arm without the non-RT arm's own
   `Tu1 − Tu2` subtracted from it.

And one inherited statistical rule: **permute.** Cross-background concordance across hundreds of
features is not hundreds of independent tests. `09` got r = −0.29 with a parametric
p = 1.7e-152 and a label-permutation p = 0.49.

## Paths

Resolved at runtime — HPC first, local copy as fallback (same pattern as `08`/`09`):

```
HPC    /coh_labs/yunroseli/Jona/CAR-T
local  /Users/janzules/Roselab/Spatial/CAR_T
  single cell   data/sc-reference/annotation_v4.h5ad          (HPC: data/zarr/fullDataset/annotating_references/)
  spatial       data/zarrFiles/1_C2l_annotations_400          (HPC: data/zarr/fullDataset/processing_Zarr/1_C2l_annotated_CC)
  hemispheres   docs/B_subsample_barcodes.xlsx
  out           results/rt_advantage/  (+ figures/)
```

Coordinates: **µm = `obsm['spatial']` × 0.34, with no hires scale.** `obsm['spatial']` is in
full-resolution pixels, and 0.34 µm is one full-resolution pixel. The `× hires_scale` used in
`spp1Analysis` 02-08 under-states distance 4.2-5x and varies by slide. `04` checks this against
cell spacing (6.6-9.8 µm nearest-neighbour distance).
