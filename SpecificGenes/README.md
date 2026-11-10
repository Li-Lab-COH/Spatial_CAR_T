# SpecificGenes

Where to look into a specific gene or gene set across the whole dataset.

| file | what |
|---|---|
| `marco_expression_sc.ipynb` | MARCO: is it expressed in tumour cells, SPP1-expressing macrophages or suppressive myeloid cells, and how widely? Executed, with outputs. Result: essentially undetected (0.09% of cells). |
| `cd36_expression_sc.ipynb` | CD36, quick look: which cells express it (per population, share of UMI, mice). Executed. |
| `trem2_expression_sc.ipynb` | TREM2, quick look (same layout as CD36). Executed. |
| `marco_expression_sc.py` | Code-only export of the notebook, used as a template (no outputs). |

Outputs: `results/SpecificGenes/<gene>/` (CSVs, figures). Mouse-specific; adapt symbols, design and signatures for other genes or the human project.
