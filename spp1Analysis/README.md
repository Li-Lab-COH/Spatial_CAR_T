# Spp1+ macrophage spatial analysis

Testing the PDF's model in the CAR-T / RT prostate dataset: *RT builds a damaged, hypoxic,
debris-rich, lipid-rich, stromally-remodeled niche; recruited monocytes stabilize there as
SPP1-high TAMs.*

Notebooks run on the `spatial_gpu_py311` HPC kernel. All helper functions are **defined inline**
because the kernel runs on the HPC node and cannot see these local files (same reason as
`Hotspot_Abscopal_Comparison.ipynb`).

## Run order

> **STATUS — the Spp1 *measurement* is broken in spatial; the *biology* is not.**
>
> Notebook 01 §5: B cells (which should be Spp1-silent) were *more* Spp1-positive than
> macrophages at every cutoff (95.7% vs 80.7% at `>=1` UMI), and 80% of all 2M cells scored
> positive. Macrophages showed no enrichment over the all-cell rate. **No value of
> `SPP1_THRESHOLD` fixes this.**
>
> This does **not** mean the Spp1+/lipid-metabolism population is absent — Jonathan's scRNA-seq
> from these same samples shows it clearly, and his `Spp1pos_highLR` vs `Spp1neg` DE reproduced
> the lipid signature. The correct reading is narrower: **raw Spp1 counts cannot isolate that
> population in spatial data.** Real biology, non-specific single-gene spatial measurement.
>
> Note also: at 80% Spp1+, the `Spp1pos` half of the `Spp1pos_highLR` filter selected nearly
> every macrophage — the **high-LR term did the discriminating**. The state was found by
> program and context, not by the single-gene call.
>
> **Path forward: stop binarizing on Spp1.** Derive a multi-gene SPP1-TAM signature from the
> single-cell DE (Spp1 excluded), transfer it as a graded score. Run `01c`. Do not run 02-04
> until the definition is settled — all three take the Spp1 call as given.

> **RESOLVED in 01c — the tumour gate, and why the project continues.**
>
> On the correct reference (`annotation_v4.h5ad`, 135k cells / full genes), the *ligand-source*
> question is settled: **Cancer_cell produces ~53% of tissue Spp1**, neutrophils ~23%, the whole
> macrophage compartment ~8%. So the deck's "SPP1-CD44 interaction-high clusters" are a
> **tumour-derived Spp1 -> CD44 axis**, and that one claim must be worded that way. This is the
> only thing the gate forces.
>
> It does **not** touch the project's actual subject — the **metabolic state** of SPP1+
> macrophages (lipid/lysosomal/TREM2 program), which 01c §4 confirms by single-cell DE
> independently of the ligand source. A tumour-dominated Spp1 niche *is* the PDF's model: tumour
> Spp1/lipid/debris forms the niche; monocyte-derived macrophages become lipid-handling SPP1-TAMs
> **inside** it. Macrophages are the metabolic **readout**, not the source.
>
> The go-forward question (reframed): *among macrophages, does the SPP1-TAM metabolic program
> track proximity to the tumour Spp1 / oxidised-lipid / hypoxia / adenosine niche?* Threshold-free,
> graded `SPP1_TAM_sig`, independent of who secretes Spp1. See 01c §3b for the full reframe.

| Notebook | Answers | Gate |
|---|---|---|
| `01_spp1_definition_and_abundance.ipynb` | *Which conditions have the most Spp1+ macs?* | §5 **FAILED** — see status. §1-4 valid. |
| `01b_spp1_source_and_spillover_diagnostic.ipynb` | *Why did §5 fail — depth or spillover? Who makes Spp1?* | Optional now — `01c` supersedes its definition-picking, but §3 (who makes Spp1) is still an open gate. |
| `01c_singlecell_anchor_and_signature.ipynb` | *Same B-cell control on clean single-cell data; SC-derived signature transferred to spatial* | Done — tumour gate + graded signature. |
| `01d_cd44_source_and_metabolic_convergence.ipynb` | *Who makes Cd44? Metabolic state of Spp1+ tumour vs Spp1- tumour? Do Spp1+ tumour & SPP1-TAMs converge on one metabolic program?* | Single-cell. Q3 (convergence) is the metabolism-angle centrepiece. |
| `02_pathway_hotspots_and_spp1_enrichment.ipynb` | *Which pathway niches (hypoxia, adenosine, lipid) do they live in?* | Result is a **lead**, not a conclusion, until 03 Part A. |
| `03_niche_control_and_celltype_neighborhood.ipynb` | *Does 02 survive a circularity control? What cell types surround them?* | Part A decides whether 02 stands. |
| `04_liana_spp1_cd44.ipynb` | *How does the (tumour-driven) Spp1-CD44 axis vary by condition; do Spp1+ macs localize to it?* | Reframed: axis is tumour-derived (01c). Now asks localization, not macrophage-driving. |

Each ends with a printed `SUMMARY — report these back` block meant to be pasted back for the
next iteration. `spp1Analysis.ipynb` is left empty as scratch space.

## Setup facts (verified, not assumed)

- **Dataset**: `1_C2l_annotated_CC`, 2,088,557 cells × **19,059 genes**. Whole-transcriptome
  (Visium probe set), *not* a targeted panel — so pathway coverage is not a limiting factor and
  the ≥5-matched-gene filter rarely fires. 32 tissues.
- **`.X` is log1p-normalized**; `layers['counts']` is raw UMI. Spp1 thresholding uses counts.
- **Cell types**: `c2l_consolidated` — `M1_like_Mac` (19,170), `M2_like_Mac` (60,720),
  `Intermediate_Mac` (1,675) = **81,565 macrophages**. 32% of cells are `Unknown` and are
  excluded from denominators.
- **Coordinates**: `MICRONS_PER_PIXEL = 0.34` + hires scale, per `Hotspot_Abscopal_Comparison`.
- **Single-cell reference**: use the object carrying **`cell_type_lvl2`** ("Alice's
  annotations") — `results/cell2location/C2L_inputs/sc_with_signatures_lvl2.h5ad`, or its
  pre-training input `ref_adata_c2l_model.h5ad`. This is what `C2L_Training.ipynb` /
  `Mapping_script.py` used to produce the spatial labels, so `M1_like_Mac`/`M2_like_Mac`/
  `Intermediate_Mac`/`B` exist on both sides and **`Cancer_cell` is present**.
  **Do not use** `data/sc-reference/sc_reference_cell2location.h5ad` — coarse `cell_type`
  (Macrophage/Monocyte/…), labels don't match spatial, and it has **no tumour compartment**, so
  the "does the tumour make Spp1?" gate is unanswerable with it.
- **Design**: `tissue` = `COND_loc_rep`. Conditions `NoTx, CyPSCA, CyT72, RTCyPSCA, RTCyT72`;
  in RT groups **loc 1 = irradiated, loc 2 = abscopal**. n = 2–4 tissues per group, so every
  contrast is read as effect size + replicate consistency, never as a p-value.

## Two corrections to the code you sent

1. **`c2l_winner` does not exist in this dataset.** Your snippet did
   `adata.obs["c2l_winner"].isin(["M1_like_Mac", ...])` while copying from `c2l_permissive` —
   two different columns, and neither is right here. The column carrying the M1/M2/Intermediate
   labels is **`c2l_consolidated`**. That snippet came from your older CellCharter object.
2. **`resource_name='mouseconsensus'` does *not* use human symbols**, despite the comment in
   your LIANA call. It is keyed on mouse symbols (`Spp1`, `Cd44`) and matches `var_names`
   directly. Script 04 asserts this rather than trusting the comment.

## The pathway panel (your "what pathways should we use?")

Mined from the 4 GMTs in `references/pathways/` and **verified present** (56/58 on first pass;
the 2 misses were swapped for `GOBP_REGULATION_OF_LIPID_STORAGE` and
`REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION`). Full panel is in script 02 §4, tiered by how
directly each block tests the PDF:

- **T1_niche** — the mechanism itself: hypoxia (`HALLMARK_HYPOXIA`, `GOBP_CELLULAR_RESPONSE_TO_HYPOXIA`,
  `HIF1A` signaling), adenosine (`GOBP_ADENOSINE_METABOLIC_PROCESS`, `GOBP_AMP_METABOLIC_PROCESS`,
  purinergic receptors), oxidized lipid (`GOBP_CELLULAR_RESPONSE_TO_OXIDISED_LOW_DENSITY_LIPOPROTEIN_PARTICLE_STIMULUS`
  — the closest GMT match to the oxLDL mechanism, `REACTOME_BINDING_AND_UPTAKE_OF_LIGANDS_BY_SCAVENGER_RECEPTORS`),
  foam-cell / lipid storage, efferocytosis, glycolysis-lactate.
- **T2_amplifier** — FAO/PPAR, glutamine/arginine/polyamine (the PDF's "reinforce, don't
  initiate"), ECM/fibroblast/angiogenesis (the FAP+ immune-excluding niche).
- **T3_contrast** — IFN-γ/α, inflammatory, p53, apoptosis, complement, ROS. **These are the
  controls that earn the T1 claim.** If Spp1+ macs enrich in T3 as strongly as T1, the
  enrichment is a density artifact, not niche specificity.

There is no GMT for "the SPP1-TAM state" itself, so script 01 builds one from the PDF's readout
list (`Trem2/Gpnmb/Apoe/Fabp5/Marco/...`, plus adenosine, efferocytosis and hypoxia modules) and
uses it to **validate the Spp1+ call independently** — Spp1 itself is excluded from the module,
since including it would guarantee separation and prove nothing.

## The three ways this analysis could fool us

Each has a control built in, because all three would produce a *positive-looking* result:

1. **Ambient RNA / segmentation spillover.** Spp1 is a highly-expressed secreted transcript, so
   `Spp1 > 0` may just mean "next to an Spp1-high cell" — which would manufacture the spatial
   association we're testing. → Notebook 01 §5 sweeps the threshold with a B-cell spillover
   control. **This control fired.** The raw-count definition is dead; `01b` diagnoses whether
   the cause is depth (fix: CP10K normalization) or spatial spillover (fix: subtract local
   background) and picks a replacement. This is the control doing its job — had we skipped it,
   80.7% of macrophages would have been labelled "Spp1+ TAMs" and every downstream result
   would have been a restatement of "macrophage".
2. **Macrophage-ness vs the Spp1 state.** "Spp1+ macs are in hypoxic zones" is trivially true if
   all macrophages are. → Every enrichment uses **Spp1- macrophages as the baseline**, not all
   cells. Both baselines are reported side by side; divergence between them *is* the message.
3. **Circularity.** Spp1+ macs are Apoe/Trem2/lipid-high, so they raise the local lipid score
   themselves — the "hotspot" can be made of the cells we then find enriched in it. → Script 03
   Part A redefines every niche using **non-myeloid cells only** (monocytes excluded too, since
   they're the proposed precursor). If enrichment survives, it's real; if it collapses, script
   02's headline was an artifact and gets reported as one.

## Outputs

Everything lands in `/coh_labs/yunroseli/Jona/CAR-T/results/spp1_analysis/` (+ `figures/`).
Each script ends with a `SUMMARY — report these back` block designed to be pasted straight back
for the next iteration.

## Known open questions

- **Spp1 threshold** defaults to `>= 2` UMI. Notebook 01 §5–6 is what sets it — change
  `SPP1_THRESHOLD_DEFAULT` there and keep `SPP1_THRESHOLD` in 02–04 in sync.
- **Runtime**: notebook 02 is ~30–60 min for all 32 tissues (kNN is built once per tissue and
  reused across pathways, which is what makes it tractable). `TISSUE_SUBSET` smoke-tests first.
- **`Unknown` at 32%** is high. If Spp1+ macs turn out to live near `Unknown`-dense regions,
  that's an annotation-coverage confound, not biology.
