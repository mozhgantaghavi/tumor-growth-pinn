# Real tumour-growth data

Three public datasets of tumour volumes measured with calipers in mice. All are distributed on Zenodo under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/); the files here are
unmodified copies. They are numeric tables (text or spreadsheet), not images.

| key | file | reference | contents |
|---|---|---|---|
| `breast` | `LM2-4LUC.txt` | Mastri M., Tracz A., Ebos J. M. L. *Tumor growth kinetics of human LM2-4LUC+ triple negative breast carcinoma cells.* Zenodo (2019). [doi:10.5281/zenodo.3574531](https://doi.org/10.5281/zenodo.3574531) | 66 vehicle-control mice, columns `ID`, `Time` (days after implantation), `Observation` (mm³) |
| `lung` | `LLC_sc_CCSB.txt` | Benzekry S., Lamont C., Weremowicz J., Beheshti A., Hlatky L., Hahnfeldt P. *Tumor growth kinetics of subcutaneously implanted Lewis Lung carcinoma cells.* Zenodo (2019). [doi:10.5281/zenodo.3572401](https://doi.org/10.5281/zenodo.3572401) | 20 mice, columns `ID`, `Time`, `Vol` (mm³) |
| `treated` | `ebos_data_breast.xlsx` | Mastri M., Benzekry S., Ebos J. M. L. *Pre- and post-surgical monitoring of experimental primary tumor growth and metastasis under neo-adjuvant treatment.* Zenodo (2024). [doi:10.5281/zenodo.10607753](https://doi.org/10.5281/zenodo.10607753) | 232 mice (vehicle, sunitinib, axitinib; several doses and schedules), primary-tumour volumes in columns L to AJ of the sheet |

The datasets were also described in Benzekry S. *et al.*, Classical mathematical models for description and prediction
of experimental tumor growth, PLoS Computational Biology 10(8): e1003800 (2014).

## Pre-processing (`src/tumorpinn/real_data.py`)

- Non-positive volumes are dropped; animals with fewer than 7 valid measurements are skipped; each curve is shifted to
  start at day 0.
- 56 of the 66 vehicle mice in the `treated` spreadsheet are the same animals as in `breast` (identical volume
  series). They are removed from `treated`, so no animal is counted twice.
- The `treated` mice received sunitinib or axitinib at different doses and schedules, so growth laws fitted to them
  describe net kinetics under treatment.

`python -m tumorpinn real` re-downloads any missing file from Zenodo.
