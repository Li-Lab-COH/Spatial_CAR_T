# What RT adds to PSCA CAR-T in the irradiated tumor

`psca_rt_figures.ipynb` builds two figures from the outputs of `code/mhcCytotoxExhaustion/` and
`code/rtAdvantage/01`; run those first. It runs locally on `spatial_cpu_py311`. The first run takes about
5 minutes, because it fits the per-site DESeq2 models and caches them in `persite_carxrt_de.parquet`.
Later runs take under a minute.

`rt_advantage_summary.ipynb` (run after `psca_rt_figures.ipynb`) builds **Figure 0**, a one-page
summary of every RT advantage found so far in genes and cell composition. It has a scorecard that marks
each finding Holds (p ≤ 0.05), Trend, Caveat (significant, but it moves with a tumor-wide immediate-early program), or Against.

Outputs: `results/psca_rt_figures/`

| file | content |
|---|---|
| `Figure0_RT_advantage_summary` | RT effect on composition per CAR (Tu1, Tu2), CD8 phenotype, CAR × RT gene heatmap (macrophages, CD8), scorecard |
| `Figure1_CD8_irradiated` | CD8 T cells in Tu1: per-tumor marker fractions, RT effect forest, gene-level RT effect in PSCA vs TAG72, cell counts per tumor |
| `Figure2_macrophages_irradiated` | M1-like / M2-like share, PSCA effect without → with RT for Ciita, Cd274, Pvr, Il1b, Cd86, Cd80, which arm moves, per-tumor expression |
| `captions.md` | draft captions with the numbers filled in |
| `fig*_*.csv` | the numbers behind each panel |
| `persite_carxrt_de.parquet` | cache of the Tu1-only and Tu2-only pydeseq2 fits |

## Statistics
- **Unit:** one tumor sample. CD8 fractions use only tumors with at least 8 CD8 cells.
- **Direct RT:** RT − no RT in Tu1, per CAR. **Pooled** is the mean of the two CARs. **CAR × RT** is the
  RT effect in PSCA minus the RT effect in TAG72.
- **p-values:**
  - Exact relabelling. Within one CAR at 3 v 3 the floor is 0.10.
  - Composition panels also show the Welch t p on log share used in `rtAdvantage/02` (`p_prop`).
- **Genes:** a new pydeseq2 fit on **Tu1 samples only** (`~ car + rt + car:rt`). The earlier
  `mhcCytotoxExhaustion/02` fit pooled both tumors, so it could not tie a CAR × RT gene effect to the
  irradiated tumor. Both p-values are in `fig*_genes_tu1.csv`.

## What held up in the irradiated tumor, and what did not
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
- **M1-like share:** RT raises it within PSCA by +7.6 pp (Welch t on log share p 0.059, exact 0.10) and not within TAG72
  (+1.8 pp). RT also raises M2-like cells in PSCA, so the M1:M2 balance is not shifted beyond what PSCA
  does alone (CAR × RT p 0.96).
- **Il1b:** p 0.06–0.09 at Tu1. It reached p ≤ 0.04 only in the both-site fit.
- **PD-L1 (Cd274):** p ≥ 0.09 at Tu1.

**Who moves (panel E)**
- **PSCA arm:** RT lowers Pvr, Cd86 and Cd80.
- **TAG72 arm:** RT raises the same genes.
- An interaction is therefore partly "TAG72 gets worse with RT", not only "PSCA gets better".

## Caveat added 2026-11-12: a tumor-wide immediate-early / NF-κB program
`rtAdvantage/specificCellTypeAnalysis/01` (Figure M6) found a shared program that every immune compartment of a
tumor follows, from macrophages and monocytes to CD8 cells. It is built from immediate-early genes (Fos, Jun,
Egr1, Dusp1, Hspa1a/b, Nr4a1, Klf2, Nfkbia).

**Where it goes**
- In the irradiated tumor it is lower with RT + PSCA (−1.1 to −1.4 SD) and higher with RT + TAG72 (+0.6 to
  +0.9 SD); CAR × RT p 0.07–0.14 per compartment.
- Tumor cells do not follow it.
- A mouse's two tumors agree only weakly (rho 0.35).
- These data cannot tell whether it is technical (handling) or systemic.

**Consequences for the gene claims above**
- **CD8 Nr4a1, Nr4a3, Klf2:** these are immediate-early genes, so the CD8 CAR × RT result is this program.
- **Macrophage Ciita:** tracks the program within arms (r −0.76 to −0.98). Its gain disappears after adjustment.
- **Pvr, Cd80, Cd86:** do not track it within arms. After adjustment they keep ~60–70% of their effect, but lose
  significance.
- **Il1b:** tracks it (r ≈ 0.7).
- **Not affected:** the RT-pooled CD8 results (Gzmb+, Cxcr3+, Ccr5+, Tcf7+), because the program moves in
  opposite directions in the two CARs.
- **A new PSCA-side result from the macrophage notebook:** with RT, myeloid cells lose Nt5e/CD73 (does not track
  the program), Vegfa and Il10 (both track it).

Figure 0 marks these rows as "Caveat", and the gene cells as "≈".
