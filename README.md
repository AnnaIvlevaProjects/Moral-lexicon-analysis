# Moral Lexicon Frequency Processing

This project implements the article-style pipeline:

1. For each **whole word**, yearly values are scaled so term peak = 100.
2. For each **stem\*** (MFD mode), preselected representative rows are summed, then scaled to peak = 100.
3. Group trajectories are aggregated from normalized term-level data.
4. Normalization peak is computed inside selected year interval (`year-start/year-end`).

## Dictionary compatibility (auto-detected)

- **MFD (with `stem`)**: stem rows ending with `*` are aggregated from their representatives.
- **MFD2 (without `stem`)**: app auto-creates empty `stem` and processes all rows as ready whole words.

## CLI

```bash
python frequency_pipeline.py \
  --dictionary "D:/.../MDF_eng_extended_GBN - en.xlsx" \
  --frequencies "D:/.../Word_freq_MDF_en-GB.xlsx" "D:/.../Word_freq_MDF_en-US.xlsx" \
  --corpora GB US \
  --year-start 1900 --year-end 2022 \
  --aggregation-mode individualism_collectivism \
  --smoothing-window 5 \
  --save-csv-dir output
```

Aggregation modes:
- `base`
- `pairs` (Groups: Harm, Fairness, Ingroup, Authority, Purity, General)
- `individualism_collectivism` (1..4 vs 5..10)
- `virtue_vice` (odd groups vs even groups in 1..10)

## Streamlit

```bash
streamlit run streamlit_app.py
```

Features:
- data source mode: upload files or use preloaded folders,
- you can run only Dataset B (MFD2) without Dataset A, if B dictionary + frequency files are provided,
- corpora selection,
- year-range selection (exact numeric inputs),
- aggregation mode selection,
- smoothing window exact numeric input,
- one-plot or per-corpus plotting (distinct line styles/markers for black-and-white print),
- PNG chart download,
- pair-group Excel export (`Harm`, `Fairness`, `Ingroup`, `Authority`, `Purity`, `General`),
- unsmoothed normalized **term-level** Excel export (`term_id`, `group`, `source_type`, `scale_peak`, `scale_peak_year`, years),
- smoothed group trajectories Excel export,
- Pearson correlation matrix table (with Excel download), including cross-dataset (MDF vs MDF2) correlations when both datasets are loaded.
- avoids full recomputation when settings/files are unchanged (cached analysis for smoother UI).
- export/correlation artifacts are memoized for unchanged settings to reduce UI freezes on reruns.


## Comparing MFD vs MFD2 on one chart

In Streamlit you can upload:
- **Dataset A** (e.g., MFD with stems),
- **Dataset B** (e.g., MFD2 without stems, optional).

When both are uploaded, the app plots both datasets together on the same figure
(using union of corpora from uploaded datasets), with labels like `MFD:Harm` vs `MFD2:Harm`.


Note: corpus labels are compact auto-labels like `MDF_en-GB`, `MDF2_en-GB` (derived from frequency filenames).


## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io/ and sign in.
3. Click **New app** and select your repository/branch.
4. Set **Main file path** to `streamlit_app.py`.
5. Deploy.

Cloud will install dependencies from `requirements.txt` automatically.
After deployment, you will get a public URL (`https://...streamlit.app`).


Cluster/factor analysis outputs in app:
- 03_years_in_pca_space.png
- 04_moral_groups_pca_space.png (PCA of moral groups, clustered)
- PCA loadings table + text interpretation (top contributors for PC1/PC2)


## Windows Git Bash / PyCharm terminal path fix

If you run:

```bash
streamlit run /scripts/streamlit_app.py
```

Git Bash treats `/scripts/...` as an absolute path under Git installation directory
(e.g. `C:/Program Files/Git/scripts/...`), so Streamlit cannot find your project file.

Use one of these commands from the project root instead:

```bash
streamlit run streamlit_app.py
# or
streamlit run ./streamlit_app.py
# or with explicit full path
streamlit run "D:/Moral_lexicon_processing/pythonProject1/streamlit_app.py"
```

Quick check:

```bash
pwd
ls streamlit_app.py
```

`ls` must show the file before running Streamlit.


## Built-in dictionaries from project `data/`

You can keep dictionaries inside the project and load them automatically in Streamlit:

- `data/MFD_en.xlsx` for Dataset A (MFD)
- `data/MFD2_en.xlsx` for Dataset B (MFD2)

In **Upload files** mode, enable:
- `Use project dictionary A (data/MFD_en.xlsx)`
- `Use project dictionary B (data/MFD2_en.xlsx)`

Then upload only frequency files.


## Years in PCA space interpretation

- each point is a year projected into 2D PCA space (PC1 and PC2).
- closer years have more similar moral-profile vectors across selected groups.
- axis labels show explained variance share for PC1/PC2.
- app now provides a loadings table and text summary of top contributing groups for PC1 and PC2, helping interpret what each axis means.


Base group semantics used in app:
`01 HarmVirtue`, `02 HarmVice`, `03 FairnessVirtue`, `04 FairnessVice`, `05 IngroupVirtue`, `06 IngroupVice`, `07 AuthorityVirtue`, `08 AuthorityVice`, `09 PurityVirtue`, `10 PurityVice`, `11 MoralityGeneral`.


PCA loadings columns:
- `PC1_loading`, `PC2_loading`: signed contribution direction to each principal component.
- `PC1_abs`, `PC2_abs`: absolute loading magnitudes (contribution strength, independent of sign).

Both PCA charts now include a text interpretation of principal coordinates.
