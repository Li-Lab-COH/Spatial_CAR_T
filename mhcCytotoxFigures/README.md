# Presentation figures — MHC, interferon, cytotoxic and chemokine programs

One notebook, `presentation_figures.ipynb`, builds five figures and a QC supplement from the outputs of
`code/mhcCytotoxExhaustion/` (run that folder first). It recomputes every plotted estimate, 95% CI and p-value,
so each panel is self-contained. Runs locally on `spatial_cpu_py311` in ~1 minute; the first run takes ~2
minutes, because it re-scores the chemokine-free SPP1 signature and caches `fig5_pseudobulk.parquet`.

Outputs: `results/mhc_cytotox_figures/`

| file | content |
|---|---|
| `Figure1_cytotoxic` | NK / CD8 cytotoxic machinery: cell-type landscape, PSCA effects by RT stratum, per-sample fractions, abundance |
| `Figure2_interferon` | PSCA → interferon programs across myeloid populations: program heatmap, gene-level DE, every sample |
| `Figure3_antigen_presentation` | MHC-I vs MHC-II machinery: coefficient plot, gene-level DE, MHC-II score per sample |
| `FigureS3_tumour_MHC_QC` | why tumour MHC-II is left out: intrinsic fraction, and the CAR × RT effect in all vs clean tumour cells |
| `Figure4_PSCA_vs_RT` | PSCA vs direct vs indirect RT: forest plot, replication matrix, the abscopal ISG / Cxcl9 decrease |
| `Figure5_SPP1_chemokines` | SPP1-high vs IFN-stimulated macrophage chemokines: paired DE, matched tissues, modules, PSCA induction |
| `captions.md` | draft captions with the numbers filled in |
| `panels/` | every panel as its own PNG, for building slides |
| `fig*_*.csv` | the numbers behind each panel |

Each figure is saved as a 300-dpi PNG and as a PDF with editable text (Type 42).

## How the statistics are done
- **Unit:** one tumour sample. Program scores are pseudobulk z-scores, and effects are in within-arm SD units,
  as in `mhcCytotoxExhaustion/03`.
- **PSCA effect:** PSCA − TAG72, matched by site within each RT stratum. "Pooled" averages the two strata, and
  CAR × RT is the difference between them.
- **p-values:** exact relabelling.
- **95% CIs:** pooled-variance t intervals on the same estimate.
- **Leave-one-mouse-out:** a claim is stable if dropping any single sample flips no stratum's sign.
- **Gene-level heatmaps:** pydeseq2 log2FC with genome-wide BH padj.
- **Floor:** with 3 tumours per side, the smallest exact p within one background is 0.10.

## Colour (dataviz skill; palette validated with its checker)
- **CAR background:** PSCA orange `#eb6834`, TAG72 blue `#2a78d6`, untreated muted grey. All-pairs CVD
  ΔE 24.7, normal-vision 33.6, PASS.
- **Tumour site:** marker shape (circle = Tu1, square = Tu2).
- **Effects:** diverging blue ↔ red with a grey midpoint; the red arm is lightness-matched to the documented
  blue steps.
- **Expression magnitude:** single-hue violet ramp, so it is never read as TAG72.

## Decisions made while building the figures (worth knowing before presenting)
1. **Figure 1 title is softer than the one suggested.**
   - With the validated core module (Gzmb, Prf1, Nkg7, Ctsw), the NK PSCA effect is +1.38 SD with RT
     (p = 0.065), +0.17 without RT (p = 0.80), and +0.78 pooled (95% CI −0.43 to +1.98; p = 0.15).
   - The broader Ccl5-free score gives +1.10 (p = 0.048).
   - The title therefore reads "associated with higher …, most clearly with RT". The original wording is one
     line away (`FIG1_TITLE`).
   - CD8: pooled +0.00 (p = 1.0). It is under-sampled, with 16 usable samples.
2. **Figure 5 re-derives the SPP1-TAM call.**
   - The 50-gene SPP1-TAM signature contains **Ccl2, Ccl7, Ccl8 and Ccl12**. Comparing CCL2/7/12 between cells
     selected on that score would be circular.
   - The figure uses the signature with those four removed (46 genes; Jaccard 0.91 with the original call)
     and Alice's independent `Spp1_pos_mac` annotation.
   - The result holds under both: the CCL2/7/12 module is +2.9 / +2.8 log2 above IFN-stimulated macrophages
     in 29/29 and 28/28 tissues; CXCL9/10 is −1.4 / −0.9.
   - Caveat: whether Cxcl9/10 helped define the `IFN_stimulated_Mac` label is not documented.
3. **Figure 2C omits untreated tumours.** They sit 10–20 SD lower and would flatten the PSCA-vs-TAG72
   contrast; untreated is a pipeline check, not a comparator.
4. **The Reactome "IFN-γ signaling" column in Figure 2A does not move.** Here it is 19 receptor and kinase
   components (Ifngr1/2, Jak2, Camk2, Mapk, Ptpn, Socs), not induced genes; the caption says so.
5. **Tumour-cell MHC-II stays out of Figure 3.**
   - Its CAR × RT effect (+2.2 to +3.5 SD, p ≤ 0.02) weakens and loses significance in tumour cells with no
     macrophage-marker reads (Figure S3).
