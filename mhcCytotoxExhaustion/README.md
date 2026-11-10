# MHC, cytotoxicity, exhaustion and chemokine programs — single cell

**Question.** What do antigen-specific CAR-T (PSCA vs the non-specific TAG72 CAR), direct
radiation (Tu1) and indirect / abscopal radiation (Tu2) do to MHC, cytotoxic, exhaustion and
chemokine programs, and in which cell types? Which of them should be taken into the spatial
SPP1–CD44 analysis?

**Why this folder exists.** `rtAdvantage/03` found that the only reproducible tumour-state signals
were PSCA-up / TAG72-down **interactions** on MHC-I/II, and that the genes behind them are
macrophage-dominant. That made it unclear whether the signal was tumour biology or macrophage RNA
in the tumour pseudobulk. This folder characterises those programs properly, adds T/NK exhaustion
and chemokine axes, and resolves the ambient-RNA question.

Single cell only (`annotation_v4.h5ad`). `06` reads the spatial object for **gene detection
only**; the spatial analysis itself is done elsewhere.

## Design and contrasts

| effect | definition | replication rule |
|---|---|---|
| **PSCA** | PSCA CAR − TAG72 CAR, site-matched | same sign in the non-RT **and** RT strata |
| **direct RT** | RT − no-RT in Tu1 | same sign in the PSCA **and** TAG72 backgrounds |
| **indirect RT** | RT − no-RT in Tu2 (abscopal) | same, both backgrounds |
| **CAR × RT** | (RT effect in PSCA) − (RT effect in TAG72) | single test |
| **DiD** | (Tu1 − Tu2)_RT − (Tu1 − Tu2)_non-RT, mice paired | per background |
| treated vs NoTx | pipeline check only | |

- The mouse / sample is the unit everywhere.
- Sample-level tests use **exact relabelling**.
- DE is pydeseq2 on summed raw counts, with **BH within each pre-specified panel** (chemokine ligands
  and receptors corrected separately).
- GSEA covers all GMT sets with 10–500 genes; chemokine axes (1–5 genes) are never GSEA'd.
- Population associations use **Freedman–Lane** permutation with car / rt / site as covariates.

## Run order

All notebooks run locally on `spatial_cpu_py311` (re-runs take about 10 minutes with the caches in
place). Outputs go to `results/mhc_cytotox/`.

| notebook | answers | runtime |
|---|---|---|
| `00_programs_and_pseudobulk` | design, program catalogue (65 programs), pseudobulk + per-cell panel cache, coverage, Cxcl11 check | 15 s |
| `01_which_cells_express` | **Q1** per-cell level and contribution per condition, mouse-bootstrap ranking, **ambient-RNA decontamination**, chemokine sources, expression-vs-abundance decomposition, receptor-positive fractions | 10 s |
| `02_differential_expression` | **Q2a** 128 DESeq2 fits (15 cell groups × 10 contrasts), panel BH, replication, leave-one-mouse-out | 5 min (then cached) |
| `03_pathways_and_programs` | **Q2b** GSEA (105 rankings × 5,844 sets) + sample-level program scores, random-program control, "stands out" = both methods agree, Ccl5 / Xcl1 sensitivity | 6 min (then cached) |
| `04_exhaustion_T_NK` | exhaustion in CD8 / NK / T: sample-level tests, binomial GLMMs, MDE, checkpoint ligands, the Cd44 caveat | 20 s |
| `04b_chemokine_axes_sc` | receptor map, ligand ↔ receiving-population association, chemokines vs IFN-γ / MHC / cytotoxicity, SPP1-high vs IFN-stimulated macrophages, axis table | 30 s |
| `05_claims_ledger` | **Q3** tiered program, exhaustion and chemokine ledgers; the decisive tumour-MHC-II ambient test | 15 s |
| `06_spatial_shortlist` | **Q4** spatial handoff: R01 cytotoxicity-score audit, per-sample LIANA SPP1→CD44 vs programs, spatial detection / spillover QC | 3 min (then cached) |

## Results (local run, 2026-10-22)

### What is definitive

**1. PSCA (antigen-specific CAR) drives an IFN-γ → MHC → CXCL9/10 program in myeloid cells.**
This is the dominant signal; 21 program × cell-type claims are Definitive.
- **Programs:** IFN-γ (Hallmark NES +2.9 to +3.3), ISG core, MHC-I machinery (+1.9 to +3.2 SD) and
  MHC-II core (+2.3 to +3.4 SD).
- **Where:** IFN-stimulated macrophages, M2-like, intermediate, SPP1-TAM and classical monocytes.
- **Replication:** in both RT strata, with every leave-one-mouse-out, beating random programs.
- **Chemokines (gene level, both strata):**
  - Cxcl9: +3.9 log2 in classical monocytes, +3.1 in IFN-stimulated macrophages, +2.5 in M2, +3.1 in cDC.
  - Cxcl10, Ccl5 and Cxcl16 also rise.
- **Checkpoint ligand:** Cd274 (PD-L1) +1.5 to +2.5 across every myeloid group.
- **Effector module in NK:** +1.2 SD. It holds without Ccl5.

**2. Tumour-cell MHC-II is macrophage RNA, not tumour biology.**
- Median intrinsic fraction of tumour MHC-II signal: H2-Ab1 0.04, H2-Eb1 0.22, Cd74 0.25, H2-Aa 0.30.
- Tumour MHC-I is mostly intrinsic (0.53–0.81).
- The CAR × RT "MHC-II up in tumour" result from `rtAdvantage/03` (H2-Ab1 / Eb1 / Cd74 ~ +2.1 to +2.8,
  p 0.025–0.055) **disappears in tumour cells with zero macrophage-marker reads** (p = 0.10–0.68).
- The contamination fraction itself trends the same way (+1.45 at Tu1, p 0.17).
- **Do not claim RT or CAR induces MHC-II on tumour cells from this object.**

**3. Indirect (abscopal) RT lowers the IFN / CXCL9 axis in myeloid cells, in both backgrounds.**
- ISG core −1.3 SD in IFN-stimulated macrophages (Definitive).
- Cxcl9 −0.8 / −1.2 log2 in IFN-stimulated macrophages and −0.7 / −1.3 in classical monocytes.
- No leave-one-out flips.
- **This is the only reproducible abscopal effect in the folder, and it points down, not up.**

**4. Direct RT has no Definitive effect on any program.**

### Suggestive (shown in the ledgers with caveats)

- **CD8 exhaustion shifts toward terminal with direct RT, in both backgrounds:**
  - IR ≥ 2 +17 pp, Havcr2+, Lag3+, Gzmb+ +30 pp;
  - progenitor (Tcf7, no Xcl1) down; terminal − progenitor +0.38.
  - Caveats: 24 "Supported" exhaustion tests against ~4 expected, but the measures share genes, and
    the MDE at ~14 CD8 cells per sample is 20–50 pp. Treat it as a lead, not a result.
- **CAR × RT in lymphocytes:** CD8 Nr4a1 / Nr4a3 down, Cxcr6 and Gzmk up, Pdcd1 down in pooled
  lymphocytes. These are single tests. The program-level CAR × RT count (54 at p < 0.05) is **at
  chance** for ~2,000 tests.
- **Per-sample SPP1→CD44 potential to CD8** (single-cell LIANA) is negatively associated with:
  - CD8 inhibitory receptors: partial r −0.67, p 0.011, q 0.10, n = 16;
  - CD8 effector score: r −0.51, p 0.08;
  - tumour MHC-I: r −0.54, p 0.015.

### Chemokines

- **Sources (Q1):**
  - Cxcl9 / Cxcl10 — classical monocytes and IFN-stimulated macrophages.
  - Ccl2 / 7 / 12 — M2 macrophages.
  - Ccl17 / Ccl22 — cDC.
  - Cxcl12 — fibroblasts / endothelium.
  - Xcl1 — NK (76%).
  - Cxcl1 / 2 / 3 — macrophages and neutrophils.
- **Receptors:**
  - Cxcr3: CD8 27%, NK 6%, Treg 29%.
  - Xcr1: ~0% everywhere, cDC included. **No cDC1 were captured, so XCL1 → XCR1 is ligand-only.**
  - Ccr4: NKT / Treg / CD4, all with too few cells to model.
- **Ligand ↔ receiving population (Q3).**
  - Only one axis survives BH: **Cxcl1 / Cxcl2 from M2 macrophages and classical monocytes ↔ N1
    neutrophil abundance** (partial r 0.63–0.84, q ≤ 0.05).
  - Cxcl2 also holds within mouse (Tu1 − Tu2: r 0.88–0.89, p 0.04); Cxcl1 is weaker (r 0.66–0.76,
    p 0.10–0.15).
  - CXCR3 ligands show **no** association with CD8 / NK abundance once car / rt / site are accounted
    for (MDE |r| ≈ 0.59 at n = 23).
  - This is association with composition, **not recruitment**.
- **IFN-γ ↔ CXCL9/10 (Q4).**
  - Cell by cell, within every macrophage / monocyte / DC state, IFN-γ (chemokines removed) tracks
    Cxcl9 / Cxcl10 in 96–100% of samples (mean ρ 0.17–0.59).
  - MHC-I tracks them too (ρ 0.16–0.28, ≥ 80% of samples).
  - MHC-II tracks them inconsistently (ρ −0.08 to 0.32; in some states < 60% of samples).
  - Across samples the link is carried by the CAR effect itself.
  - Cxcl9 / 10 does not add to CD8 / NK abundance or Gzmb / Prf1 / Nkg7 after adjustment.
- **SPP1-high vs IFN-stimulated macrophages** (paired, both SPP1 definitions agree; note that the 50-gene SPP1
  signature itself contains Ccl2 / Ccl7 / Ccl8 / Ccl12, so for those genes the signature definition is circular.
  Re-checked in `code/mhcCytotoxFigures` with a chemokine-free signature: CCL2/7/12 module +2.9 log2, 29/29
  tissues; Alice's definition +2.8, 28/28):
  - SPP1-TAMs are **low** in Cxcl9 / Cxcl10, Ccl5, Ccl17 / Ccl22 and Ccr2;
  - they are **high** in Ccl2 / Ccl7 / Ccl12, Ppbp, Cxcl3, Cxcl1 and Ackr3.
  - So SPP1-TAMs carry a monocyte / neutrophil-recruiting profile rather than the CXCR3 / CCR5
    effector-recruiting one.
- **Sensitivity:** removing Ccl5 / Xcl1 flips 2 of 51 group × effect cells.

### For the spatial SPP1–CD44 analysis

- **The R01 "cytotoxic-high" score** (GO granzyme-mediated set) **is not a lymphocyte readout.**
  - In single cell, its top 15% scorers are ~85% macrophages / monocytes (NK 7%, CD8 2%), driven by
    Srgn, Lamp1 and Bnip3.
  - In spatial, **neither it nor a lymphocyte-restricted alternative separates lymphocyte-labelled
    cells** (AUROC 0.55; single cell 0.82 / 0.95).
  - Gzmb is detected in 27% of spatial `M1_like_Mac` cells vs 14% of `CD8_T`; Spp1 in 94–98% of
    CD8 / NK / B / tumour labels.
  - The observed SPP1–CD44 vs cytotoxic segregation therefore cannot be read as segregation of
    killer cells until it is redone with a readout that survives this test.
- **Best-supported things to map** (`spatial_handoff_programs.csv`):
  - IFN-γ / ISG and MHC-II in classical monocytes and M1-labelled macrophages, which pass as
    cell-type readouts;
  - Cxcl9 / Cxcl10 / Ccl5 / Cxcl16 from those sources.
- **LIANA `mouseconsensus` lacks Cxcl9→Cxcr3**, as well as Ccl3→Ccr5, Ccl7→Ccr2, Cxcl3→Cxcr2 and
  Ppbp→Cxcr2. Add these pairs, or a default run will never score them.
- **Do not map** (`spatial_do_not_map.csv`):
  - tumour MHC-II;
  - the R01 score;
  - CD8 exhaustion / progenitor modules;
  - Cxcr3 / Ccr4 / Cxcr6 / Xcr1 as per-cell receptor readouts (≤ 0.6% detection, at the B-cell
    spillover floor).

## Key files (`results/mhc_cytotox/`)

| file | content |
|---|---|
| `ledger_programs.csv`, `ledger_chemokines.csv`, `ledger_exhaustion.csv` | the Q3 tiers |
| `chemokine_axis_table.csv` | axis, ligand source, receptor carrier, treatment effect, support, MHC / cytotoxicity link, spatial transferability, LIANA-resource flag |
| `spatial_handoff_programs.csv`, `spatial_do_not_map.csv` | the Q4 handoff |
| `de_all.parquet`, `de_panel_replication.csv`, `gsea_all.parquet`, `program_score_contrasts.csv` | the full statistics |
| `q1_*.csv` | sources, contributions, ambient estimates |

## Caveats

- n = 2–3 mice per arm × site. The exact-relabelling p-value floor is ~0.05 per stratum and
  ~0.003–0.01 for joint effects.
- CD8 / NKT / CD4 / Treg are too sparse for per-sample pseudobulks in most samples; exhaustion rests
  on cell-level GLMMs and sample-level fractions with large MDEs.
- GO "T-cell-mediated cytotoxicity" is 26 / 57 MHC genes. Cytotoxicity and exhaustion sets scored in
  myeloid or tumour cells are **reclassified**, not claimed.
- Nothing here measures migration. Chemokine conclusions are transcriptional, and "inferred
  chemotactic potential" is reported separately and never as recruitment.
