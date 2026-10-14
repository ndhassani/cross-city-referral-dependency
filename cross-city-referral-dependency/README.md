# Cross-City Referral Dependency in Saudi Arabia

Code for the paper **"Cross-City Referral Dependency in Saudi Arabia: A Diagnosis-Level, City-to-City Analysis of Referral Volume and Destination"** (Nabil Alhassani, Department of Health Administration and Hospital, College of Public Health and Health Informatics, Umm Al-Qura University).

The pipeline turns the five quarterly *Referral of Medical Cases* extracts (Q4 2024 – Q4 2025) into an analysis dataset. It then reproduces every table, figure and sensitivity analysis in the paper and its supplementary material. Diagnoses are classified with **ICD-10-AM only**: each code is assigned to its chapter from the ICD-10-AM chapter code ranges, and no external ICD library is used.

## Repository contents

| File | What it does |
|---|---|
| `01_data_preparation.py` | Raw Excel extracts → analysis dataset. Runs five stages: loading, data-quality review, cleaning and city→region mapping, ICD-10-AM chapters, and data-quality flags. It also writes the preprocessing reports. |
| `02_referral_dependency_analysis.py` | All tables, figures and sensitivity analyses. Covers descriptive statistics, the cross-city dependency index, the quasi-binomial model, sensitivity analyses, hubs, exploratory cluster analyses A–D with validation, and maps. |
| `icd10am_chapters.py` | Assigns an ICD-10-AM code to its chapter from the chapter code ranges, and holds the short chapter labels. |
| `requirements.txt` | Python packages. |
| `data/raw/` | Put the five Excel files here. They are not stored on GitHub. |

## Data

The data are openly available and are **not** included in this repository. The dataset is *Referral of Medical Cases*, published by the **Center for National Health Insurance** on the **Saudi National Open Data Portal**: <https://open.data.gov.sa/en/datasets>.

five excel sheets"1-Referral of Medical Cases Between Healthcare Providers for Central Region Q4 2025 , 2-Referral of Medical Cases Between Healthcare Providers for Eastern Region Q4 2025 , 3-Referral of Medical Cases Between Healthcare Providers for Southern Region Q4 2025, 4- Referral of Medical Cases Between Healthcare Providers for western Region Q4 2025, 5-Referral of Medical Cases Between Healthcare Providers for northern Region Q4 2025" were merged to this file "Referral of Medical Cases 2025 Q4 .xlsx""
Download the five quarterly files and save them in `data/raw/` (see `data/raw/README.md`):

```
data/raw/Referral of Medical Cases 2024 Q4 .xlsx
data/raw/Referral of Medical Cases 2025 Q1 .xlsx
data/raw/Referral of Medical Cases 2025 Q2 .xlsx
data/raw/Referral of Medical Cases 2025 Q3 .xlsx
data/raw/Referral of Medical Cases 2025 Q4 .xlsx
```

Each row of an extract gives the number of referrals for one combination of gender, referring city, receiving city, age group and ICD-10-AM diagnosis code. The referring provider region is recorded only in Q4 2025.

## How to run

### On your computer

```bash
git clone https://github.com/<ndhassani>/<cross-city-referral-dependency>.git
cd <repository-name>
pip install -r requirements.txt
# copy the five Excel files into data/raw/
python 01_data_preparation.py
python 02_referral_dependency_analysis.py
```

### In Google Colab

```python
!git clone https://github.com/<ndhassani>/<cross-city-referral-dependency>.git
%cd <repository-name>
!pip install -q -r requirements.txt
from google.colab import files
import shutil
for name in files.upload():                 # choose the five Excel files
    shutil.move(name, "data/raw/" + name)
!python 01_data_preparation.py
!python 02_referral_dependency_analysis.py
```

`01` takes a few seconds. `02` takes about 2 minutes, most of it the permutation and bootstrap checks of the cluster analyses. For a quick test run, use `python 02_referral_dependency_analysis.py --n-permutations 10 --n-bootstrap 10`, or skip the clustering with `--skip-clustering`.

### Options

| Script | Option | Default | Meaning |
|---|---|---|---|
| `01` | `--raw-dir` | `data/raw` | Folder with the Excel files |
| `01` | `--out-dir` | `.` | Project folder for `data/processed/` and `reports/` |
| `01` | `--icd10am` | none | Optional CSV of the licensed ICD-10-AM tabular list (column `code`). It reports the share of referrals with a listed code or listed parent code. The list is licensed, so it is not distributed here. |
| `02` | `--input` | `data/processed/referrals_analysis.csv` | Analysis dataset |
| `02` | `--quarterly-input` | `data/processed/referrals_quarterly.csv` | Quarter-level file, used for Tables S36–S38 and Figure S18 |
| `02` | `--out-dir` | `reports/analysis` | Output folder |
| `02` | `--min-same-city` | `100` | A city is coverage-adequate if it has at least this many same-city referrals |
| `02` | `--n-permutations` / `--n-bootstrap` | `100` / `200` | Null-benchmark permutations and bootstrap replicates |

## What the pipeline does

**01 – data preparation**
1. **Loading.** Reads every value as text, so age bands are not turned into dates and codes are kept exactly. The quarter is read from each file name.
2. **Data-quality review.** Reports missing values, duplicates, spelling variants (dictionary and fuzzy matching), unexpected categories, invalid and extreme counts, and malformed codes. Nothing is changed at this stage.
3. **Cleaning and regions.**
   - Harmonises city names. The Eastern Province directorate unit ("East", "Eastern", "Eastern Region" in the city columns) is merged with Al-Ahsa.
   - Removes exact duplicate rows and rows with invalid counts.
   - Builds a city→region lookup from Q4 2025, using the majority region of each city's referrals, and reports its accuracy on Q4 2025.
   - Merges rows describing the same group. The referral total is checked to be unchanged.
4. **ICD-10-AM chapters** from the chapter code ranges:
   - Supplementary codes for chronic conditions (U78–U88) are kept as a separate group.
   - Other U codes are grouped as "Other (special/unclassified)".
5. **Flags and datasets.**
   - Count heaping: a value at least 4× as frequent as its neighbouring values within a city.
   - Reporting-coverage changes: a city-quarter total below half or above twice the city's median quarter.
   - Referrals involving facilities outside Saudi Arabia are excluded.
   - The analysis dataset is aggregated over quarters and genders.

**02 – analysis**
- **Cross-city dependency index (DI).** For each referring city, DI = cross-city referrals / all referrals, with Wilson 95% CIs. Exact binomial tests against the national proportion use Benjamini–Hochberg correction.
- **Associations.** Chi-square tests with bias-corrected Cramér's V (Bergsma).
- **Main model.** A quasi-binomial GLM of the cross-city share on referring city, ICD-10 chapter and age group, fitted to referral-weighted cells of the coverage-adequate cities.
  - References: Riyadh, circulatory diseases, 18–29 years.
  - Drop-one deviance with F tests.
  - A test of the city × chapter interaction.
- **Sensitivity analyses.**
  - Coverage-adequate cities only.
  - Regional nesting.
  - Supplementary codes by city.
  - DI by quarter and by gender.
  - Excluding count-heaping cities and flagged city-quarters.
- **Hubs.** Origin–destination matrix, Herfindahl–Hirschman Index (HHI) and top-3 share.
- **Exploratory clustering.**
  - Analyses: A, local city profiles; B, outbound city profiles; C, ICD-10 chapters; D, the 100 highest-volume codes.
  - City profiles are shrunk towards the national profile (Dirichlet prior of 0.5 × K pseudo-referrals) and Hellinger-transformed.
  - K-Means and Ward are compared for each k, selecting the highest silhouette without singleton clusters.
  - Validation: a permutation null benchmark, block-drop/double-weight sensitivity, prior strength, bootstrap, leave-one-out, seed stability, and comparison with simpler partitions (main destination, administrative region, Louvain communities).

All random steps use seed 42.

## Outputs

**`data/processed/`**
- `referrals_clean.csv`: cleaned quarter-level rows, including Outside KSA.
- `referrals_quarterly.csv`: quarter-level analysis rows, with chapter, cross-city and cross-region indicators, and data-quality flags.
- `referrals_analysis.csv`: the analysis dataset. One row is one ICD-10 code within a cell of referral type × referring city × receiving city × age group.

**`reports/preprocessing/`**
- Data-quality report (`stage2_data_quality_report.xlsx`).
- Mapping report (`stage3_mapping_report.xlsx`).
- ICD chapter tables.
- Count-heaping and coverage flags.
- `figure_S1_flow.png` (Figure S1).
- `methods_text.md`: every number needed for the Methods section.

**`reports/analysis/`**
- `all_tables.xlsx`: every table, one sheet each.
- `tables/*.csv` and `figures/*.png` (300 dpi).
- `key_numbers.json`: the numbers quoted in the text.

| Paper | Output file |
|---|---|
| Table 1 | `Table1_volume_by_type` |
| Table 2, Figure 2 | `Table2_dependency_index`, `Figure2_dependency_index_by_city` |
| Table 3 | `Table3_deviance_contributions` |
| Table 4, Figure 1 | `Table4_adjusted_OR_chapter`, `Figure1_adjusted_OR_chapter` |
| Table 5 | `Table5_receiving_hubs` |
| Table 6, Figures 4–5 | `Table6_outbound_typology`, `Figure4_AnalysisB_dendrogram_pca`, `Figure5_AnalysisB_profiles` |
| Figure 3 | `Figure3_DI_city_by_chapter` |
| Figure 6 | `Figure6_maps` |
| Tables S1–S38 | `TableS<number>_...` (Tables S36–S38 need the quarter-level file) |
| Figures S2–S18 | `FigureS<number>_...` (Figure S1 is in `reports/preprocessing/`) |

## Reproducibility notes

- The pipeline is deterministic. A re-run reproduces the analysis dataset and all tables exactly.
- Louvain community detection (Table S31) can differ slightly between NetworkX versions.
- Figure 6 draws the outline of Saudi Arabia from Natural Earth when GeoPandas and an internet connection are available. Without them, the map is drawn without the outline.
- City coordinates (Table S32) are approximate administrative capitals, used only for mapping.

## Citation

If you use this code, please cite the paper (full reference to be added on publication) and the dataset:

> Center for National Health Insurance. Referral of Medical Cases [dataset]. Riyadh: Saudi National Open Data Portal; 2025. Available from: https://open.data.gov.sa/en/datasets

## Licence

MIT. See `LICENSE`.
