# What RT adds to PSCA CAR-T in the irradiated tumour

`psca_rt_figures.ipynb` builds two figures from the outputs of `code/mhcCytotoxExhaustion/` and
`code/rtAdvantage/01`; run those first. It runs locally on `spatial_cpu_py311`. The first run takes about
5 minutes, because it fits the per-site DESeq2 models and caches them in `persite_carxrt_de.parquet`.
Later runs take under a minute.

Outputs: `results/psca_rt_figures/`

| file | content |
|---|---|
| `Figure1_CD8_irradiated` | CD8 T cells in Tu1: per-tumour marker fractions, RT effect forest, gene-level RT effect in PSCA vs TAG72, cell counts per tumour |
| `Figure2_macrophages_irradiated` | M1-like / M2-like share, PSCA effect without → with RT for Ciita, Cd274, Pvr, Il1b, Cd86, Cd80, which arm moves, per-tumour expression |
| `captions.md` | draft captions with the numbers filled in |
| `fig*_*.csv` | the numbers behind each panel |
| `persite_carxrt_de.parquet` | cache of the Tu1-only and Tu2-only pydeseq2 fits |

## Statistics
- **Unit:** one tumour sample. CD8 fractions use only tumours with at least 8 CD8 cells.
- **Direct RT:** RT − no RT in Tu1, per CAR. **Pooled** is the mean of the two CARs. **CAR × RT** is the
  RT effect in PSCA minus the RT effect in TAG72.
- **p-values:**
  - Exact relabelling. Within one CAR at 3 v 3 the floor is 0.10.
  - Composition panels also show the Welch t p used in `rtAdvantage/02`.
- **Genes:** a new pydeseq2 fit on **Tu1 samples only** (`~ car + rt + car:rt`). The earlier
  `mhcCytotoxExhaustion/02` fit pooled both tumours, so it could not tie a CAR × RT gene effect to the
  irradiated tumour. Both p-values are in `fig*_genes_tu1.csv`.

## What held up in the irradiated tumour, and what did not
**Held up**
- **CD8 cells, RT pooled over both CARs:**
  - **Up:** Gzmb+ +30 pp (p 0.03), Cxcr3+ +13 pp (p 0.01), Ccr5+ +22 pp (p 0.03), ≥ 2 inhibitory
    receptors +17 pp (p 0.03).
  - **Down:** Tcf7+ −20 pp (p 0.05). Tox+ trends down, −18 pp (p 0.07).
  - **CAR dependence:** none of these differ significantly between PSCA and TAG72.
- **CD8 genes, CAR × RT at Tu1:** with RT, PSCA CD8 cells lose Nr4a1, Nr4a3 and Klf2 while TAG72 CD8 cells
  gain them (p 0.005–0.013). Cxcr6 also differs (p 0.035).
- **Macrophage genes, CAR × RT at Tu1:**
  - Pvr is lower in all 5 macrophage groups (p 0.006–0.02).
  - Cd86 is lower in all 5 (p 0.002–0.01).
  - Cd80 is lower in 4 (p ≤ 0.04).
  - Ciita is higher in 3 (p 0.02–0.04); in M1-like cells p = 0.09.

**Only trends**
- **M1-like share:** RT raises it within PSCA by +7.6 pp (Welch p 0.08, exact 0.10) and not within TAG72
  (+1.8 pp). RT also raises M2-like cells in PSCA, so the M1:M2 balance is not shifted beyond what PSCA
  does alone (CAR × RT p 0.96).
- **Il1b:** p 0.06–0.09 at Tu1. It reached p ≤ 0.04 only in the both-site fit.
- **PD-L1 (Cd274):** p ≥ 0.09 at Tu1.

**Who moves (panel E)**
- **PSCA arm:** RT lowers Pvr, Cd86 and Cd80.
- **TAG72 arm:** RT raises the same genes.
- An interaction is therefore partly "TAG72 gets worse with RT", not only "PSCA gets better".
