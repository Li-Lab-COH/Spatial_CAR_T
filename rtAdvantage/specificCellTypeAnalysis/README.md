# Specific cell-type analysis — what RT improves on, cell type by cell type

Single-cell follow-ups to `rtAdvantage/00–05` and `pscaRtFigures/`. There is no RT-only arm, so each
notebook asks **what RT improves on** in composition, gene expression and behavior. "Improves" means a change
that should favor killing or reduce immunosuppression, either in both CAR backgrounds or specifically with
PSCA (CAR × RT).

| notebook | cell types | outputs |
|---|---|---|
| `01_macrophage_rt.ipynb` | monocytes, macrophages and their sub-clusters; SPP1-TAMs | `results/rt_advantage/specific_cell_type/macrophages/` |
| `02_summary_figures.ipynb` | two summary figures from 01 + earlier tables: S1 irradiated tumor (PSCA macrophage shift, RT MHC trend, RT CD73 drop), S2 abscopal leads | `results/rt_advantage/specific_cell_type/summary/` |
| `03_nk_rt.ipynb` | NK cells: IFN-γ, cytotoxic / inhibitory phenotype, CD73 / adenosine, recruitment environment; spatial cross-check | `results/rt_advantage/specific_cell_type/nk/` |
| `04_story_figures.ipynb` | five presentation figures: PSCA arms NK, RT lifts the local brake, the two-cell adenosine relay, the abscopal brake, model + proposal | `results/rt_advantage/specific_cell_type/story/` |

Planned next: a spatial analysis of **cytotoxic cell-type enrichment around cancer cells** (see the notes at
the end).

## 01 — Macrophages

Runs locally on `spatial_cpu_py311` and reads `results/mhc_cytotox/` (run `mhcCytotoxExhaustion/00` first).
The first run takes ~30 minutes, mostly 24 per-site DESeq2 fits, 7 paired state fits and 79 GSEA rankings.
All of them are cached (`de_persite_carxrt.parquet`, `de_state_vs_rest.parquet`, `gsea_all.parquet`), so later
runs take a few minutes. Delete a cache file to refit.

**States**
- **M1-like:** the IFN-stimulated macrophage. It is the same cell set as the lvl3 label.
- **M2-like:** split into Apoe-high, Arg1-high and Fn1-high.
- **Intermediate:** the same cells as Mmp12-high.
- **Classical monocytes:** split into IFN-stimulated and phagocytic.
- **Non-classical (patrolling) monocytes.**
- **SPP1-TAMs, two definitions:**
  - the top quartile of the 50-gene signature (`spp1Analysis/09`);
  - Alice's lvl4 `Spp1_pos_mac`.

**Figures**

| figure | content |
|---|---|
| `FigureM1_composition` | per-tumor share of each state; effect heatmap (PSCA, RT per CAR, CAR × RT, both sites); SPP1+ share inside each sub-cluster |
| `FigureM1b_spp1_trem2_high` | Spp1-high, Trem2-high and double-high macrophages (one global cut); which sub-clusters host them; fewer cells vs lower expression; per-cell expression |
| `FigureM6_ieg_axis_check` | the tumor-wide immediate-early / NF-κB program: by arm × compartment, same-mouse coherence, gene effects adjusted for it, Hallmark GSEA without its genes |
| `FigureM2_state_identity` | what each macrophage state does vs the other macrophages in the same tumor: function modules and top GSEA pathways |
| `FigureM3_rt_behavior` | what RT changes in each state: module scores (RT in PSCA, RT in TAG72, CAR × RT, abscopal) and Hallmark GSEA |
| `FigureM4_spp1_tams` | SPP1-TAM share, what SPP1-TAMs do, and how RT changes the SPP1-TAM vs other-macrophage gap |
| `FigureM5_scorecard` | every RT test, hits vs chance, and the strongest advantage / against hits |

**Statistics**
- **Unit:** one tumor.
- **Composition and module scores:** exact relabelling p.
  - Within one CAR the floor is 0.10 at 3 v 3, so per-CAR rows also report a Welch t p (on log share for
    composition, as in `rtAdvantage/02`).
  - Pooled RT and CAR × RT use the combined exact null.
- **Genes:** pydeseq2 0.5.4.
  - Per site, `~ car + rt + car:rt`.
  - State identity: paired `~ donor + state`, where the comparison pseudobulk is all macrophages minus the
    state, by subtraction of summed counts.
- **GSEA:** prerank weighted KS on the Wald statistic over Hallmark, Reactome and GO BP (sizes 15–500).
- **Module orientation** (▲ anti-tumor, ▼ suppressive, · context) is the textbook role of each module. It is
  a reading aid, not a measured function.
  - Pvr (CD155) is left unoriented, because it is a ligand for both TIGIT (inhibitory) and CD226 (activating).
  - Cd80 and Cd86 are counted as co-stimulation, although CD80 also binds PD-L1 in cis.
- **Chance:** the scorecard reports, for every family and contrast, how many tests reach p < 0.05 against the
  5% expected by chance.

### What 01 found (first run, 2026-11-12)
**1. PSCA, not RT, reshapes the macrophage pool** (PSCA − TAG72, pooled over RT strata, exact p).
- **M1-like goes up:** +0.74 log2 of the mono/mac compartment in Tu1 and +1.0 in Tu2 (p 0.005). IFN-stimulated
  monocytes also go up.
- **M2 sub-clusters go down:** Apoe-high −0.9 log2 (p 0.01–0.015) and Fn1-high −1.2 to −1.3 (p 0.005).
- **M1:M2 ratio:** +1.5 to +1.7 log2 (p 0.005).
- **Spp1-high and Trem2-high macrophages shrink with it.**

  | readout | Tu1 | Tu2 |
  |---|---|---|
  | Spp1-high, pp of macrophages | −13 (p 0.005) | −17 (p 0.04) |
  | Trem2-high, pp | −10 (p 0.045) | −9 (p 0.18) |
  | Spp1+Trem2 double-high, pp | −9 (p 0.01) | −9 (p 0.075) |

  - **Where they sit:** the Apoe-high cluster holds 43–54% of these cells and 66–70% of the double-high ones,
    and 57% of Apoe-high cells are Spp1-high.
  - **Why they shrink:** about two thirds of the drop is fewer cells of the high-expressing sub-clusters
    (p 0.005 at both sites). The drop in Spp1/Trem2 expression within clusters is not significant (p 0.18–0.7).

**2. RT on top of PSCA adds little to composition.** In the irradiated tumor:
- M1-like trends up within PSCA (Welch p 0.094).
- Patrolling monocytes trend down (p 0.062).
- **Pooled over both CARs, fewer Apoe-high (gene-defined) macrophages:** −0.67 log2 (p 0.01). The composition
  part of the Trem2-high drop is −2.5 pp (p 0.04).

**3. What RT changes in behavior is almost entirely PSCA-specific (CAR × RT) and in the irradiated tumor.**
- Module scores, CAR × RT at Tu1: 61 of 376 tests at p < 0.05, against 19 expected by chance.
- Module scores, RT pooled over both CARs: 2 of 376, below chance.
- **Advantage direction, falling on the PSCA side:**
  - lower Nt5e/CD73 (5 groups; p 0.001–0.02), plus Adora2a and Entpd1/CD39 in monocytes;
  - lower Vegfa (6 groups);
  - lower Il10 (M2-like, SPP1-TAM, classical monocytes).
- **Against:** lower Cd80/Cd86 (co-stimulation).
- **Hallmark GSEA, CAR × RT at Tu1, 7–8 of 8 myeloid groups:** TNFα/NF-κB, inflammatory response, hypoxia,
  TGF-β, apoptosis and G2M down; OXPHOS up.

**4. Check — a tumor-wide immediate-early / NF-κB program moves with CAR × RT (Figure M6).**
- **Where it goes:** lower in RT + PSCA, higher in RT + TAG72, in every immune compartment. Tumor cells do not
  follow it, and a mouse's two tumors agree only weakly. It could be technical or systemic.
- **Which results it touches:**
  - Il1b, Il10, Vegfa and Ciita track it within arms.
  - The CD73/adenosine module moves against it.
  - Pvr, Cd80 and Cd86 barely track it.
  - Removing its 31 genes leaves the Hallmark result unchanged.
- **Treat the CAR × RT myeloid shifts as real but entangled with this program.** CD73 is the cleanest.

**5. SPP1-TAMs** (vs the other macrophages in the same tumor) are:
- **high in:** lysosome/vacuole, collagen turnover, iron uptake, mTORC1, scavenger, TREM2/lipid, complement
  and CCR2 chemokines;
- **low in:** MHC-II, Treg-recruiting chemokines and co-stimulation.
- **RT barely changes how they differ from their neighbors:** SPP1-gap tests are at chance.

## 03 — NK cells

Runs locally on `spatial_cpu_py311`. The first run takes about 10 minutes, mostly one pass over the spatial matrix
(1.77 billion stored values; cached in `spatial_gene_by_tissue_label.npz`). Later runs take about 2 minutes.

**Data**
- **Single cell:** NK pseudobulks, plus per-cell counts for NK / NKT / T cells read from `annotation_v4.h5ad`
  (`layers/counts`).
  - NK numbers are small (5–107 per tumor, 5–55 in Tu2), so NK readouts use tumors with ≥ 5 NK cells and
    show the cell counts.
- **Spatial:** 31 tumor sections (two sections of one tumor merged), c2l labels.
  - **NK expression readouts:** need ≥ 30 NK-labeled cells.
  - **Spillover floor:** the same gene's detection in tumor-labeled cells.

**Figures**

| figure | content |
|---|---|
| `FigureN1_ifng` | Ifng+ NK cells per tumor, per-cell model, four Ifng measures, who supplies Ifng, IFN-γ supply vs macrophage ISG, spatial check |
| `FigureN2_nk_phenotype` | NK modules, sub-states, per-cell fractions and Hallmark GSEA across RT and PSCA contrasts |
| `FigureN3_adenosine` | myeloid CD73 per tumor, RT effects on CD73 / CD39 by site, within-mouse Tu1 − Tu2, CD73 sources, A2A receptors on NK |
| `FigureN4_recruitment` | NK abundance (single cell, spatial), spatial lymphocyte census, NK homing receptors, cue vs NK association, environment heatmap |
| `FigureN5_scorecard` | hits vs chance and the strongest advantage / against hits |

### What 03 found (first run, 2026-11-12)
**1. Abscopal NK IFN-γ comes from PSCA, not from RT.**
- **Ifng+ NK cells in Tu2, pooled over tumors:**
  - PSCA: 51% without RT vs 38% with RT;
  - TAG72: 19% without RT vs 27% with RT;
  - untreated: 6%.
- **PSCA − TAG72 without RT:** +33 pp (Welch p 0.028; per-cell model p < 0.001). In the NK pseudobulk the
  difference is +2.8 log2 (p 0.007).
- **What RT does at Tu2:**
  - lowers Ifng in PSCA mice (per-cell model p 0.006; Ifng per Ifng+ cell −0.43, p 0.024);
  - raises it in TAG72 mice (+11 pp, p 0.049);
  - CAR × RT −25 pp (p 0.035).
- **NK activation programs follow the same pattern:**
  - PSCA without RT switches them on in Tu2 (Hallmark IFN-γ response NES +2.4, TNFα +3.1, FDR < 0.05);
  - RT turns them off there (IFN-γ −2.0, TNFα −2.3);
  - in the irradiated tumor, RT raises the NK interferon response in both CARs (+1.6 to +1.8).
- **Spatial agrees:** PSCA > TAG72 in Tu2 (+2.0 pp, p 0.047), with no RT effect.
- **Who supplies IFN-γ:** CD4 T cells supply 39–56% of tumor Ifng transcripts in PSCA tumors vs 0–14% in TAG72,
  likely including CD4 CAR-T cells, which single cell cannot separate.

**2. The abscopal tumor keeps its CD73.**
- **Where CD73 comes from:** about 90% is myeloid in treated tumors (M2-like ~40%, classical monocytes 20–31%).
- **Effect of RT in PSCA mice on myeloid Nt5e:**
  - irradiated tumor: −0.92 log2 (Welch p 0.006; CAR × RT −1.59, p 0.015);
  - abscopal tumor: +0.57 (p 0.40).
- **Within each RT + PSCA mouse:** Tu1 is 1.8 log2 below Tu2, vs 0.3 in no-RT mice (exact p 0.20; the floor is
  0.10).
- **CD39 (Entpd1):** unchanged at both sites.
- **NK receptors:** 23–41% of NK cells carry the A2A receptor (Adora2a), unchanged by RT. So the abscopal NK
  cells sit in adenosine-producing surroundings and can sense it.
- **Spatial:** Nt5e detection is dominated by tumor-cell signal and is not used.

**3. RT does not recruit NK cells; in situ it depletes them from the irradiated field.**
- **Single cell:** NK abundance does not change (p ≥ 0.55).
- **Spatial, irradiated tumor:**

  | label | RT in PSCA (log2) | p | RT pooled over both CARs (log2) | p |
  |---|---|---|---|---|
  | NK | −3.5 | Welch 0.002; exact 0.029, the smallest possible | −3.1 | 0.014 |
  | CD8 | −2.9 | — | −2.7 | 0.014 |

  - NKT cells fall less.
  - CD4, B cells and Tregs do not fall. This is the classic radiosensitivity pattern: Tregs are relatively
    radioresistant.
- **Spatial, abscopal tumor:** NK cells unchanged; Tregs rise (+1.3 log2, pooled p 0.007).
- **NK homing receptors:** unchanged by RT.
- **Recruiting cues:**
  - **Without RT:** PSCA itself raises CXCR3 / CCR5 / CXCR6 ligands in Tu2 (p 0.016–0.035), and IDO1.
  - **RT with PSCA in Tu1:**
    - **Better:** Cxcl16 (+2.4 SD, p 0.033) and less PGE2 synthesis (CAR × RT p 0.015).
    - **Worse:** less IL-12 (−1.8 SD, p 0.018).
- **Within arms, tumors with more of these hold fewer NK cells:** TGF-β1 (rho −0.52, p 0.042), galectin-9
  (−0.58, p 0.016) and tumor Pvr (−0.52, p 0.033).

**4. NK state in the irradiated tumor with PSCA + RT:**
- **Fewer NKG2A+ (Klrc1) NK cells:** −28 pp (p 0.03; CAR × RT p 0.02).
- **Sub-states and receptors:** fewer Klrg1+ cells and Granzyme-high NK cells; more Il18r1.
- **Abscopal tumor:** activating receptors fall (RT pooled −2.0 SD, p 0.01).
- **Overall:** the NK hit counts sit near chance. The clearest RT effects are the spatial lymphocyte loss and the
  irradiated-site CD73 drop.

## 04 — Story figures (presentation set)

Reads `01`, `03`, `mhcCytotoxExhaustion/07_adenosine_signaling_sc` and the pseudobulks; recomputes every plotted
number. Runs in about 2 minutes.

| figure | message |
|---|---|
| `Story1_PSCA_arms_NK` | PSCA arms NK cells in the **abscopal** tumor: granzyme A +31 pp, CCL5 +30 pp, CCL3 +21 pp, cytotoxic-NK subtype +26 pp, IFN-γ +20 pp (exact p 0.005–0.02). The same program at the irradiated site is much weaker (p 0.21–0.68). |
| `Story2_RT_lifts_local_brake` | Where the beam lands, RT + PSCA lowers myeloid CD73 (−0.92 log2, p 0.006; CAR × RT −1.59, p 0.015), lowers the adenosine module and PGE2 synthesis (CAR × RT p 0.005–0.015), raises CXCL16 (+2.45 SD, p 0.033) and the NK interferon response (FDR 0.007–0.023), and halves NKG2A+ NK cells (−28 pp, p 0.030). IL-12 moves the wrong way. |
| `Story3_adenosine_relay` | The adenosine machinery is a **two-cell macrophage relay**. |
| `Story4_abscopal_brake` | The abscopal tumor keeps the brake. |
| `Story5_model_and_proposal` | Model with each link labeled by its evidence, plus three experiments. |

### The two-cell adenosine relay (Figure 3)
From `07_adenosine_signaling_sc` pool shares and `01`'s state-vs-rest fits:

| enzyme | top source | share | state vs other macrophages |
|---|---|---|---|
| CD38 (NAD⁺ → ADPR) | **SPP1-TAM** | 54% | SPP1-TAM +2.3, Apoe-high +2.1 log2 |
| CD203a / ENPP1 (ADPR → AMP) | **SPP1-TAM** | 72% | SPP1-TAM +3.6, Apoe-high +3.2 |
| CD73 (AMP → adenosine) | **other TAM** | 43% | **Arg1-high +1.5**; SPP1-TAM −0.8, Apoe-high −1.6 |
| CD39 (ATP → AMP) | classical mono | 26% | flat across states |

The top source is bootstrap-stable in 100% of resamples for CD38, CD203a and CD73.
**So the cells that start the NAD⁺ route are not the cells that finish it** — this is the CD39-independent
CD38 / CD203a / CD73 route (Horenstein et al., OncoImmunology), split across two macrophage subsets.

**TREM2 link:** among 48,475 macrophages, 80% of TREM2+ cells are Spp1+ vs 57% of TREM2− cells (OR 2.9), and
40% of TREM2+ cells are classified SPP1-TAM vs 8% of TREM2− cells. TREM2 is +1.7 log2 in SPP1-TAM and
+1.65 in Apoe-high.

### What the abscopal tumor keeps (Figure 4)
- **Myeloid CD73 is unchanged by RT at Tu2** (RT in PSCA +0.57, p 0.40), while it falls at Tu1.
- **A2A receptor (Adora2a):** NK 32%, CD8 28%, cDC 65%; tumor cells 1%. So the receptor sits on the killers.
- **Across tumors of the same arm and site, more myeloid CD73 goes with a lower NK granzyme/perforin score**
  (rho −0.53, permutation p 0.025, n = 23).
- **Tregs rise in the abscopal tumor after RT** (spatial, pooled +1.26 log2, p 0.007).

### Caveats carried on the figures
- Adenosine itself is never measured; neither is protein. The relay is inferred from transcripts.
- Figure 4C is an association, not causation.
- In situ, RT removes ~11× of the NK cells and ~7× of the CD8 cells from the irradiated field while Tregs are
  spared, so Figure 2 describes the cells that remain, not how many there are.
- PSCA itself also reshapes this pathway: it lowers tissue Enpp1 (−0.98 SD, q 0.001) and monocyte Nt5e
  (−1.16, q 0.010) while raising Cd38 and Entpd1 (`07_adenosine/07_treatment_effects.csv`).
- The TREM2 arm of the proposal is a hypothesis. Nothing here tests whether TREM2+ macrophages specifically
  supply the adenosine, or whether blocking TREM2 changes NK function.

## Notes for the planned spatial analysis
- **Distances:** use true µm, obsm['spatial'] × 0.34, never × the hires scale.
- **Prior result:** `rtAdvantage/04` already measured effector proximity to tumor cells (30/50 µm, with a
  B-cell yardstick) and found no RT main effect. Start from it.
- **Do not use the R01 "cytotoxic-high" GO score** as a lymphocyte readout (AUROC 0.55 in spatial).
- **Reference baseline from `03`:** in situ, RT depletes NK (~10-fold) and CD8 (~7-fold) cells from the irradiated field and Tregs are spared. Any cytotoxic-around-tumor analysis has to normalize for this.
- **Single-cell leads to test:**
  - in the irradiated tumor, RT goes with more Gzmb+, Cxcr3+ and Ccr5+ CD8 cells;
  - with PSCA, macrophage Pvr, Cd80 and Cd86 go down and Ciita goes up.

## 05 — Adenosine production by Spp1 / Trem2 macrophage gates

`05_spp1_trem2_adenosine.ipynb` → `results/rt_advantage/specific_cell_type/adenosine/`
(`FigureA1_spp1_trem2_adenosine.png/.pdf`, `05_gate_contribution.csv`, `05_gate_detection.csv`,
`05_dp_state_breakdown.csv`, `05_sensitivity_spp1_tertile.csv`).

Re-cuts the macrophage compartment by **marker positivity** (what a PhenoCycler panel would gate on)
instead of transcriptome state, and asks how much of each step of both adenosine routes each gate carries.
Gates: Spp1⁺Trem2⁺ 20,413 cells (42% of macrophages), Spp1⁺Trem2⁻ 13,061 (27%), Spp1⁻Trem2⁺ 5,247 (11%),
Spp1⁻Trem2⁻ 9,754 (20%). Contribution = the gate's share of the whole tumor's transcripts for the enzyme
(a partition over all cells); the unit is the mouse (12 treated mice) and p is an exact sign-flip test.

| step | enzyme | Spp1⁺Trem2⁺ share | enrichment over cell mass |
|---|---|---|---|
| ATP/ADP → AMP | CD39 / Entpd1 | 37% (CI 28–46) | +0.11 log2, p 0.15 — **not enriched** |
| NAD⁺ → ADPR | CD38 | 59% (CI 52–66) | **+0.88 log2, p < 0.001** |
| ADPR → AMP | CD203a / Enpp1 | 71% (CI 64–78) | **+1.15 log2, p < 0.001** |
| AMP → adenosine | CD73 / Nt5e | 37% (CI 32–42) | +0.19 log2, p 0.12 — **not enriched** |

The gate is 33% of the tumor's transcriptional mass, which is the no-specialization baseline.

**What this adds to the relay story.**
- The double-positive gate owns the two NAD⁺ steps and is only at parity for CD39 and CD73 — the two ends
  of the cascade are held by the tissue at large, the middle is a TREM2⁺SPP1⁺ specialty.
- **CD39 is near-universal and uninformative:** 88% of double-positive cells, 72–86% of every other gate.
  21% of double-positive cells carry CD39 and CD73 together, which is the 22% expected from independence —
  there is no CD39/CD73 double-positive subset. CD38⁺CD203a⁺ (7.8%) is the only pair above chance (5.1%).
- **The division of labor survives inside the gate:** Apoe-high double-positive cells are 56% CD38⁺ and
  27% CD203a⁺ but only 13% CD73⁺; Arg1-high are 43% CD73⁺ and 7% CD203a⁺; Fn1-high 35% CD73⁺.
- Only 1.5% of double-positive cells carry all three NAD⁺-route enzymes — the relay is obligate.

**Caveats.** Transcript level only; adenosine is never measured. Spp1 detection in macrophages carries
ambient from the Spp1-high tumor cells, so the figure is accompanied by a sensitivity gate using the
top within-sample Spp1 tertile (Spp1⁺Trem2⁺ then holds 40% of Cd38 and 53% of Enpp1 — same ranking, smaller
gate). These are descriptive gates, not treatment contrasts.
