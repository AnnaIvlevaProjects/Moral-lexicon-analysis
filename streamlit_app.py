from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from frequency_pipeline import MoralFrequencyAnalyzer, infer_corpus_name_from_filename


st.set_page_config(page_title="Moral Frequency Explorer", layout="wide")

st.title("Moral Frequency Explorer")

st.markdown(
    """
Загрузите один или два набора данных (словарь + частотные файлы).
Это позволяет строить на одном графике кривые для MFD (со stem) и MFD2 (без stem).
"""
)

AGGREGATION_LABELS = {
    "base": "Base groups (HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral)",
    "pairs": "Groups (Harm, Fairness, Ingroup, Authority, Purity, General)",
    "individualism_collectivism": "Individualism vs Collectivism",
    "virtue_vice": "Virtue vs Vice",
}

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "x", "*", "+", "v", "P", "h"]


def _check_sklearn_available() -> Tuple[bool, str]:
    try:
        importlib.import_module("sklearn")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _load_sklearn_components():
    ok, import_error = _check_sklearn_available()
    if not ok:
        raise ValueError(
            "ModuleNotFoundError: No module named 'sklearn'. The import name is `sklearn`, but the pip package is `scikit-learn`. "
            f"Current interpreter: `{sys.executable}`. "
            f"Install into this interpreter: `{sys.executable} -m pip install scikit-learn`. "
            f"Original import error: {import_error}"
        )
    cluster_mod = importlib.import_module("sklearn.cluster")
    decomposition_mod = importlib.import_module("sklearn.decomposition")
    metrics_mod = importlib.import_module("sklearn.metrics")
    preprocessing_mod = importlib.import_module("sklearn.preprocessing")
    return cluster_mod.KMeans, decomposition_mod.PCA, metrics_mod.silhouette_score, preprocessing_mod.StandardScaler


def _series_style(index: int):
    return {
        "linestyle": LINE_STYLES[index % len(LINE_STYLES)],
        "marker": MARKERS[index % len(MARKERS)],
        "markersize": 4,
        "markevery": max(1, 12 - (index % 5)),
        "linewidth": 1.8,
    }


def _figure_to_png(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _terms_workbook(results_by_dataset: Dict[str, Dict], year_start: int, year_end: int) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for dataset_name, dataset_results in results_by_dataset.items():
            if not dataset_results:
                continue
            sample = next(iter(dataset_results.values())).term_timeseries
            year_cols = [str(c) for c in sample.columns if str(c).isdigit()]
            selected_years = [y for y in year_cols if year_start <= int(y) <= year_end]
            for corpus, res in dataset_results.items():
                tdf = res.term_timeseries.copy()
                base_cols = ["term_id", "group", "source_type", "scale_peak", "scale_peak_year"]
                export_cols = [*base_cols, *selected_years]
                tdf = tdf[export_cols]
                sheet_name = f"{dataset_name}_{corpus}"[:31]
                tdf.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()




def _correlation_workbook(corr_df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        corr_df.to_excel(writer, sheet_name="correlation_matrix", index=True)
    output.seek(0)
    return output.getvalue()


def _build_correlation_matrix(
    results_by_dataset: Dict[str, Dict],
    selected_groups: List[str],
    selected_years: List[str],
    smoothing_window: int,
) -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {}
    for dataset_name, dataset_results in results_by_dataset.items():
        for corpus, res in dataset_results.items():
            gdf = res.group_timeseries
            gdf = gdf[gdf["group"].astype(str).isin(selected_groups)]
            for _, row in gdf.iterrows():
                y = row[selected_years].astype(float)
                if smoothing_window > 1:
                    y = y.rolling(window=smoothing_window, center=True, min_periods=1).mean()
                key = f"{dataset_name}:{corpus}:{row['group']}"
                series_map[key] = pd.Series(y.values, index=selected_years)

    if not series_map:
        return pd.DataFrame()

    corr_input = pd.DataFrame(series_map)
    return corr_input.corr(method="pearson")





def _cross_dataset_corr_matrix(corr_df: pd.DataFrame, dataset_a: str, dataset_b: str) -> pd.DataFrame:
    if corr_df.empty:
        return pd.DataFrame()
    cols_a = [c for c in corr_df.columns if str(c).startswith(f"{dataset_a}:")]
    cols_b = [c for c in corr_df.columns if str(c).startswith(f"{dataset_b}:")]
    if not cols_a or not cols_b:
        return pd.DataFrame()
    return corr_df.loc[cols_a, cols_b]

def _smoothed_groups_workbook(
    results_by_dataset: Dict[str, Dict],
    selected_groups: List[str],
    selected_years: List[str],
    smoothing_window: int,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for dataset_name, dataset_results in results_by_dataset.items():
            rows = []
            for corpus, res in dataset_results.items():
                gdf = res.group_timeseries
                gdf = gdf[gdf["group"].astype(str).isin(selected_groups)]
                for _, row in gdf.iterrows():
                    y = row[selected_years].astype(float)
                    if smoothing_window > 1:
                        y = y.rolling(window=smoothing_window, center=True, min_periods=1).mean()
                    rec = {"corpus": corpus, "group": str(row["group"])}
                    rec.update({year: float(val) for year, val in zip(selected_years, y.values)})
                    rows.append(rec)
            pd.DataFrame(rows).to_excel(writer, sheet_name=str(dataset_name)[:31], index=False)
    output.seek(0)
    return output.getvalue()





def _results_signature(results_by_dataset: Dict[str, Dict]) -> Tuple:
    parts = []
    for dataset_name in sorted(results_by_dataset.keys()):
        dataset_results = results_by_dataset[dataset_name]
        for corpus in sorted(dataset_results.keys()):
            res = dataset_results[corpus]
            gdf = res.group_timeseries
            tdf = res.term_timeseries
            parts.append(
                (
                    dataset_name,
                    corpus,
                    tuple(gdf.shape),
                    tuple(str(c) for c in gdf.columns[:5]),
                    tuple(str(c) for c in gdf.columns[-5:]),
                    tuple(tdf.shape),
                    tuple(str(c) for c in tdf.columns[:5]),
                    tuple(str(c) for c in tdf.columns[-5:]),
                )
            )
    return tuple(parts)

def _build_year_feature_matrix(
    group_timeseries: pd.DataFrame,
    selected_groups: List[str],
    selected_years: List[str],
    smoothing_window: int,
) -> pd.DataFrame:
    gdf = group_timeseries.copy()
    gdf = gdf[gdf["group"].astype(str).isin(selected_groups)]
    if gdf.empty:
        return pd.DataFrame()

    feature_data = {}
    for _, row in gdf.iterrows():
        group_name = str(row["group"])
        y = row[selected_years].astype(float)
        if smoothing_window > 1:
            y = y.rolling(window=smoothing_window, center=True, min_periods=1).mean()
        feature_data[group_name] = y.values

    years_int = [int(y) for y in selected_years]
    years_df = pd.DataFrame(feature_data, index=years_int)
    years_df.index.name = "Year"
    return years_df


def _choose_k_by_metrics(X: np.ndarray, min_k: int = 2, max_k: int = 6) -> Tuple[int, pd.DataFrame]:
    n_samples = X.shape[0]
    if n_samples < 3:
        return 2, pd.DataFrame({"k": [2], "inertia": [np.nan], "silhouette": [np.nan]})

    max_k = min(max_k, n_samples - 1)
    min_k = min(min_k, max_k)

    KMeans, _, silhouette_score, _ = _load_sklearn_components()

    rows = []
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(X)
        inertia = model.inertia_
        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
        rows.append({"k": k, "inertia": inertia, "silhouette": sil})

    metrics_df = pd.DataFrame(rows)
    if metrics_df["silhouette"].notna().any():
        best_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    else:
        best_k = int(metrics_df.iloc[0]["k"])
    return best_k, metrics_df



def _plot_years_in_pca_space(coords_df: pd.DataFrame, pca2) -> bytes:
    fig, ax = plt.subplots(figsize=(12, 9))
    clusters = sorted(coords_df["Cluster"].unique())
    cmap = plt.cm.get_cmap("tab10", len(clusters))
    for i, cluster in enumerate(clusters):
        part = coords_df[coords_df["Cluster"] == cluster]
        ax.scatter(part["PC1"], part["PC2"], label=f"Cluster {cluster}", alpha=0.85, s=70, color=cmap(i))

    for _, row in coords_df.iterrows():
        year = int(row["Year"])
        if year == int(coords_df["Year"].min()) or year == int(coords_df["Year"].max()) or year % 10 == 0:
            ax.annotate(str(year), (row["PC1"], row["PC2"]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Years in PCA space")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _figure_to_png(fig)


def _pretty_group_name(name: str) -> str:
    mapping = {
        "HarmVirtue": "Harm_Virtue",
        "HarmVice": "Harm_Vice",
        "FairnessVirtue": "Fairness_Virtue",
        "FairnessVice": "Fairness_Vice",
        "IngroupVirtue": "Ingroup_Virtue",
        "IngroupVice": "Ingroup_Vice",
        "AuthorityVirtue": "Authority_Virtue",
        "AuthorityVice": "Authority_Vice",
        "PurityVirtue": "Purity_Virtue",
        "PurityVice": "Purity_Vice",
        "MoralityGeneral": "Morality_General",
    }
    return mapping.get(str(name), str(name))


def _plot_groups_in_pca_space(groups_coords_df: pd.DataFrame, pca2_groups) -> bytes:
    fig, ax = plt.subplots(figsize=(12, 9))
    clusters = sorted(groups_coords_df["Cluster"].unique())
    cmap = plt.cm.get_cmap("tab10", len(clusters))

    for i, cluster in enumerate(clusters):
        part = groups_coords_df[groups_coords_df["Cluster"] == cluster]
        ax.scatter(part["PC1"], part["PC2"], label=f"Кластер {cluster + 1}", alpha=0.85, s=90, color=cmap(i))

    for _, row in groups_coords_df.iterrows():
        ax.annotate(str(row["group"]), (row["PC1"], row["PC2"]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel(f"PC1 ({pca2_groups.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca2_groups.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Moral groups in PCA space")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _figure_to_png(fig)

def _run_cluster_factor_analysis(
    group_timeseries: pd.DataFrame,
    selected_groups: List[str],
    selected_years: List[str],
    smoothing_window: int,
) -> Dict:
    years_df = _build_year_feature_matrix(group_timeseries, selected_groups, selected_years, smoothing_window)
    if years_df.empty or years_df.shape[0] < 3 or years_df.shape[1] < 2:
        raise ValueError("Not enough data for clustering/PCA. Need at least 3 years and 2 groups.")

    KMeans, PCA, _, StandardScaler = _load_sklearn_components()

    scaler = StandardScaler()
    X = scaler.fit_transform(years_df.values)

    best_k, _ = _choose_k_by_metrics(X, min_k=2, max_k=6)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=30)
    labels = model.fit_predict(X)

    pca2 = PCA(n_components=2)
    coords = pca2.fit_transform(X)
    coords_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Year": years_df.index.values,
        "Cluster": labels,
    })

    loadings = pd.DataFrame(
        {
            "group": years_df.columns,
            "PC1_loading": pca2.components_[0],
            "PC2_loading": pca2.components_[1],
        }
    )
    loadings["PC1_abs"] = loadings["PC1_loading"].abs()
    loadings["PC2_abs"] = loadings["PC2_loading"].abs()
    loadings = loadings.sort_values(["PC1_abs", "PC2_abs"], ascending=False).reset_index(drop=True)

    top_pc1 = ", ".join([_pretty_group_name(g) for g in loadings.sort_values("PC1_abs", ascending=False)["group"].head(3).tolist()])
    top_pc2 = ", ".join([_pretty_group_name(g) for g in loadings.sort_values("PC2_abs", ascending=False)["group"].head(3).tolist()])
    interpretation = (
        f"PC1 explains {pca2.explained_variance_ratio_[0] * 100:.1f}% and is mostly driven by: {top_pc1}. "
        f"PC2 explains {pca2.explained_variance_ratio_[1] * 100:.1f}% and is mostly driven by: {top_pc2}."
    )

    groups_X = scaler.fit_transform(years_df.T.values)
    group_k = min(3, max(1, groups_X.shape[0]))
    if group_k == 1:
        group_labels = np.zeros(groups_X.shape[0], dtype=int)
    else:
        group_labels = KMeans(n_clusters=group_k, random_state=42, n_init=30).fit_predict(groups_X)
    pca2_groups = PCA(n_components=2)
    groups_coords = pca2_groups.fit_transform(groups_X)
    groups_coords_df = pd.DataFrame({
        "PC1": groups_coords[:, 0],
        "PC2": groups_coords[:, 1],
        "group": [_pretty_group_name(g) for g in years_df.columns],
        "Cluster": group_labels,
    })

    groups_pc1_top = ", ".join(
        groups_coords_df.reindex(groups_coords_df["PC1"].abs().sort_values(ascending=False).index)["group"].head(3).tolist()
    )
    groups_pc2_top = ", ".join(
        groups_coords_df.reindex(groups_coords_df["PC2"].abs().sort_values(ascending=False).index)["group"].head(3).tolist()
    )
    groups_interpretation = (
        f"Groups-PCA PC1 explains {pca2_groups.explained_variance_ratio_[0] * 100:.1f}% and strongest-position groups are: {groups_pc1_top}. "
        f"Groups-PCA PC2 explains {pca2_groups.explained_variance_ratio_[1] * 100:.1f}% and strongest-position groups are: {groups_pc2_top}."
    )

    loadings_view = loadings[["group", "PC1_loading", "PC2_loading", "PC1_abs", "PC2_abs"]].copy()
    loadings_view["group"] = loadings_view["group"].map(_pretty_group_name)

    return {
        "03_years_in_pca_space.png": _plot_years_in_pca_space(coords_df, pca2),
        "04_moral_groups_pca_space.png": _plot_groups_in_pca_space(groups_coords_df, pca2_groups),
        "pca_loadings": loadings_view,
        "pca_interpretation": interpretation,
        "groups_pca_interpretation": groups_interpretation,
        "pc1_explained": float(pca2.explained_variance_ratio_[0]),
        "pc2_explained": float(pca2.explained_variance_ratio_[1]),
    }


def _prepare_dataset(temp_dir: Path, dict_file, freq_files) -> Tuple[Path, Dict[str, Path]]:
    dict_path = temp_dir / dict_file.name
    dict_path.write_bytes(dict_file.getvalue())

    freq_paths: Dict[str, Path] = {}
    for f in freq_files:
        p = temp_dir / f.name
        p.write_bytes(f.getvalue())
        corpus = infer_corpus_name_from_filename(p)
        if corpus in freq_paths:
            corpus = f"{corpus}__{p.stem}"
        freq_paths[corpus] = p
    return dict_path, freq_paths




def _prepare_dataset_with_dict_path(temp_dir: Path, dict_path: Path, freq_files) -> Tuple[Path, Dict[str, Path]]:
    if not dict_path.exists():
        raise ValueError(f"Dictionary file not found: {dict_path}")

    copied_dict_path = temp_dir / dict_path.name
    copied_dict_path.write_bytes(dict_path.read_bytes())

    freq_paths: Dict[str, Path] = {}
    for f in freq_files:
        p = temp_dir / f.name
        p.write_bytes(f.getvalue())
        corpus = infer_corpus_name_from_filename(p)
        if corpus in freq_paths:
            corpus = f"{corpus}__{p.stem}"
        freq_paths[corpus] = p
    return copied_dict_path, freq_paths

def _prepare_dataset_from_directory(dataset_dir: Path) -> Tuple[Path, Dict[str, Path]]:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    xlsx_files = sorted(dataset_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise ValueError(f"No .xlsx files found in: {dataset_dir}")

    dict_candidates = [f for f in xlsx_files if "word_freq" not in f.stem.lower()]
    if not dict_candidates:
        raise ValueError(
            "Dictionary file not found. Expected one .xlsx without 'word_freq' in file name."
        )
    dict_path = dict_candidates[0]

    freq_candidates = [f for f in xlsx_files if "word_freq" in f.stem.lower()]
    if not freq_candidates:
        raise ValueError(
            "Frequency files not found. Expected files containing 'word_freq' in file name."
        )

    freq_paths: Dict[str, Path] = {}
    for f in freq_candidates:
        corpus = infer_corpus_name_from_filename(f)
        if corpus in freq_paths:
            corpus = f"{corpus}__{f.stem}"
        freq_paths[corpus] = f

    return dict_path, freq_paths



def _path_fingerprint(path: Path) -> Tuple[str, float, int]:
    p = Path(path)
    stat = p.stat()
    return str(p), float(stat.st_mtime), int(stat.st_size)


def _freq_signature(freq_paths: Dict[str, Path]) -> Tuple[Tuple[str, str, float, int], ...]:
    rows = []
    for corpus, path in sorted(freq_paths.items(), key=lambda x: str(x[0])):
        p_str, mtime, size = _path_fingerprint(Path(path))
        rows.append((str(corpus), p_str, mtime, size))
    return tuple(rows)


@st.cache_data(show_spinner=False)
def _analyze_many_cached(
    dict_fp: Tuple[str, float, int],
    freq_sig: Tuple[Tuple[str, str, float, int], ...],
    selected_corpora: Tuple[str, ...],
    year_start: int | None,
    year_end: int | None,
    aggregation_mode: str,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    dict_path = Path(dict_fp[0])
    freq_paths = {corpus: Path(path) for corpus, path, _, _ in freq_sig}
    analyzer = MoralFrequencyAnalyzer(dict_path, freq_paths)

    available = [c for c in selected_corpora if c in freq_paths]
    if not available:
        return {}

    kwargs = {"aggregation_mode": aggregation_mode}
    if year_start is not None:
        kwargs["year_start"] = int(year_start)
    if year_end is not None:
        kwargs["year_end"] = int(year_end)

    results = analyzer.analyze_many(available, **kwargs)
    return {
        corpus: {
            "group_timeseries": res.group_timeseries,
            "term_timeseries": res.term_timeseries,
        }
        for corpus, res in results.items()
    }


def _restore_results(cached_results: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, SimpleNamespace]:
    restored: Dict[str, SimpleNamespace] = {}
    for corpus, payload in cached_results.items():
        restored[corpus] = SimpleNamespace(
            corpus_name=corpus,
            group_timeseries=payload["group_timeseries"],
            term_timeseries=payload["term_timeseries"],
        )
    return restored




data_source_mode = st.radio(
    "Data source",
    options=["Upload files", "Use preloaded folders"],
    horizontal=True,
)

datasets_meta = []

if data_source_mode == "Upload files":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset A")
        dataset_a_name = st.text_input("Label A", value="MFD")
        use_project_dict_a = st.checkbox(
            "Use project dictionary A (`data/MFD_en.xlsx`)",
            value=Path("data/MFD_en.xlsx").exists(),
            key="use_project_dict_a",
        )
        dict_file_a = None if use_project_dict_a else st.file_uploader("Dictionary A (xlsx)", type=["xlsx"], key="dict_a")
        freq_files_a = st.file_uploader(
            "Frequency files A (xlsx)", type=["xlsx"], accept_multiple_files=True, key="freq_a"
        )

    with col2:
        st.subheader("Dataset B (optional)")
        dataset_b_name = st.text_input("Label B", value="MFD2")
        use_project_dict_b = st.checkbox(
            "Use project dictionary B (`data/MFD2_en.xlsx`)",
            value=Path("data/MFD2_en.xlsx").exists(),
            key="use_project_dict_b",
        )
        dict_file_b = None if use_project_dict_b else st.file_uploader("Dictionary B (xlsx)", type=["xlsx"], key="dict_b")
        freq_files_b = st.file_uploader(
            "Frequency files B (xlsx)", type=["xlsx"], accept_multiple_files=True, key="freq_b"
        )

    dict_ready_a = use_project_dict_a or (dict_file_a is not None)
    dict_ready_b = use_project_dict_b or (dict_file_b is not None)

    if (dict_ready_a and freq_files_a) or (dict_ready_b and freq_files_b):
        temp_dir = Path(".streamlit_tmp")
        temp_dir.mkdir(exist_ok=True)

        if dict_ready_a and freq_files_a:
            if use_project_dict_a:
                dict_path_a, freq_paths_a = _prepare_dataset_with_dict_path(temp_dir, Path("data/MFD_en.xlsx"), freq_files_a)
            else:
                dict_path_a, freq_paths_a = _prepare_dataset(temp_dir, dict_file_a, freq_files_a)
            datasets_meta.append((dataset_a_name.strip() or "DatasetA", dict_path_a, freq_paths_a))

        if dict_ready_b and freq_files_b:
            if use_project_dict_b:
                dict_path_b, freq_paths_b = _prepare_dataset_with_dict_path(temp_dir, Path("data/MFD2_en.xlsx"), freq_files_b)
            else:
                dict_path_b, freq_paths_b = _prepare_dataset(temp_dir, dict_file_b, freq_files_b)
            datasets_meta.append((dataset_b_name.strip() or "DatasetB", dict_path_b, freq_paths_b))

else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preloaded Dataset A")
        dataset_a_name = st.text_input("Label A", value="MFD", key="pre_label_a")
        dir_a = st.text_input("Folder A", value="preloaded/dataset_a")
    with col2:
        st.subheader("Preloaded Dataset B (optional)")
        use_b = st.checkbox("Use dataset B", value=False)
        dataset_b_name = st.text_input("Label B", value="MFD2", key="pre_label_b")
        dir_b = st.text_input("Folder B", value="preloaded/dataset_b")

    load_btn = st.button("Load preloaded datasets")
    if load_btn:
        try:
            dict_path_a, freq_paths_a = _prepare_dataset_from_directory(Path(dir_a))
            datasets_meta.append((dataset_a_name.strip() or "DatasetA", dict_path_a, freq_paths_a))
            if use_b:
                dict_path_b, freq_paths_b = _prepare_dataset_from_directory(Path(dir_b))
                datasets_meta.append((dataset_b_name.strip() or "DatasetB", dict_path_b, freq_paths_b))
        except Exception as e:
            st.error(str(e))

if datasets_meta:
    # choose corpora by UNION (all corpora from uploaded datasets)
    corpora_sets = [set(freq_paths.keys()) for _, _, freq_paths in datasets_meta]
    all_corpora = sorted(set.union(*corpora_sets)) if corpora_sets else []
    if not all_corpora:
        st.error("No corpora found in uploaded files.")
        st.stop()

    selected_corpora = st.multiselect("Corpora", options=all_corpora, default=all_corpora)

    if selected_corpora:
        # pass 1 for year bounds (per dataset on available corpora only)
        base_results_by_dataset = {}
        for name, dict_path, freq_paths in datasets_meta:
            dict_fp = _path_fingerprint(dict_path)
            freq_sig = _freq_signature(freq_paths)
            cached = _analyze_many_cached(
                dict_fp,
                freq_sig,
                tuple(selected_corpora),
                None,
                None,
                "base",
            )
            restored = _restore_results(cached)
            if restored:
                base_results_by_dataset[name] = restored

        if not base_results_by_dataset:
            st.error("No selected corpora are available in uploaded datasets.")
            st.stop()

        sample = next(iter(next(iter(base_results_by_dataset.values())).values())).group_timeseries
        year_cols = [int(c) for c in sample.columns if str(c).isdigit()]
        y_min, y_max = min(year_cols), max(year_cols)

        ycol1, ycol2 = st.columns(2)
        with ycol1:
            exact_start = st.number_input("Start year", min_value=y_min, max_value=y_max, value=y_min, step=1)
        with ycol2:
            exact_end = st.number_input("End year", min_value=y_min, max_value=y_max, value=y_max, step=1)

        if exact_start > exact_end:
            st.warning("Start year is greater than end year. Values were swapped automatically.")
            exact_start, exact_end = exact_end, exact_start
        year_start, year_end = int(exact_start), int(exact_end)

        aggregation_mode = st.selectbox(
            "Aggregation mode",
            options=list(AGGREGATION_LABELS.keys()),
            index=0,
            format_func=lambda x: AGGREGATION_LABELS[x],
        )

        smoothing_window = int(
            st.number_input(
                "Smoothing window (rolling mean)",
                min_value=1,
                max_value=25,
                value=1,
                step=1,
            )
        )
        st.caption(f"Selected smoothing window: **{smoothing_window}**")

        # pass 2 with selected settings
        results_by_dataset = {}
        for name, dict_path, freq_paths in datasets_meta:
            dict_fp = _path_fingerprint(dict_path)
            freq_sig = _freq_signature(freq_paths)
            cached = _analyze_many_cached(
                dict_fp,
                freq_sig,
                tuple(selected_corpora),
                int(year_start),
                int(year_end),
                aggregation_mode,
            )
            restored = _restore_results(cached)
            if restored:
                results_by_dataset[name] = restored

        all_groups = sorted(
            {
                g
                for dataset_results in results_by_dataset.values()
                for r in dataset_results.values()
                for g in r.group_timeseries["group"].dropna().astype(str).tolist()
            }
        )
        selected_groups = st.multiselect("Groups", options=all_groups, default=all_groups)
        one_plot = st.checkbox("One subplot per corpus", value=True)

        if st.button("Plot"):
            selected_years = [str(y) for y in year_cols if year_start <= y <= year_end]
            x = [int(y) for y in selected_years]

            if one_plot:
                fig, axes = plt.subplots(len(selected_corpora), 1, figsize=(12, 4 * len(selected_corpora)), sharex=True)
                if len(selected_corpora) == 1:
                    axes = [axes]

                for ax, corpus in zip(axes, selected_corpora):
                    style_idx = 0
                    for dataset_name, dataset_results in results_by_dataset.items():
                        if corpus not in dataset_results:
                            continue
                        gdf = dataset_results[corpus].group_timeseries
                        gdf = gdf[gdf["group"].astype(str).isin(selected_groups)]
                        for _, row in gdf.iterrows():
                            y = row[selected_years].astype(float)
                            if smoothing_window > 1:
                                y = y.rolling(window=smoothing_window, center=True, min_periods=1).mean()
                            ax.plot(x, y, label=f"{dataset_name}:{row['group']}", **_series_style(style_idx))
                            style_idx += 1
                    ax.set_title(corpus)
                    ax.grid(alpha=0.25)
                    ax.legend(ncol=2, fontsize=8)
                axes[-1].set_xlabel("Year")
                axes[0].set_ylabel("Scaled frequency (peak=100)")
            else:
                fig, ax = plt.subplots(figsize=(13, 7))
                style_idx = 0
                for dataset_name, dataset_results in results_by_dataset.items():
                    for corpus, res in dataset_results.items():
                        gdf = res.group_timeseries
                        gdf = gdf[gdf["group"].astype(str).isin(selected_groups)]
                        for _, row in gdf.iterrows():
                            y = row[selected_years].astype(float)
                            if smoothing_window > 1:
                                y = y.rolling(window=smoothing_window, center=True, min_periods=1).mean()
                            ax.plot(x, y, label=f"{dataset_name}:{corpus}:{row['group']}", **_series_style(style_idx))
                            style_idx += 1
                ax.set_title("Moral group trajectories")
                ax.set_xlabel("Year")
                ax.set_ylabel("Scaled frequency (peak=100)")
                ax.grid(alpha=0.25)
                ax.legend(ncol=2, fontsize=8)

            st.pyplot(fig)
            st.session_state["plot_png"] = _figure_to_png(fig)
            st.session_state["plot_name"] = "moral_groups_comparison.png"

        if "plot_png" in st.session_state:
            st.download_button(
                "Download chart (PNG)",
                data=st.session_state["plot_png"],
                file_name=st.session_state.get("plot_name", "moral_groups_comparison.png"),
                mime="image/png",
            )

        selected_years_for_corr = [str(y) for y in year_cols if year_start <= y <= year_end]
        exports_sig = (
            _results_signature(results_by_dataset),
            int(year_start),
            int(year_end),
            tuple(selected_groups),
            tuple(selected_years_for_corr),
            int(smoothing_window),
        )
        if st.session_state.get("exports_cache_sig") != exports_sig:
            st.session_state["exports_cache_sig"] = exports_sig
            st.session_state["terms_excel_cached"] = _terms_workbook(results_by_dataset, year_start, year_end)
            st.session_state["smoothed_groups_excel_cached"] = _smoothed_groups_workbook(
                results_by_dataset,
                selected_groups,
                selected_years_for_corr,
                smoothing_window,
            )
            st.session_state["corr_df_cached"] = _build_correlation_matrix(
                results_by_dataset,
                selected_groups,
                selected_years_for_corr,
                smoothing_window,
            )

        st.download_button(
            "Download unsmoothed normalized term-level data (Excel)",
            data=st.session_state["terms_excel_cached"],
            file_name="normalized_terms_pre_averaging.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download smoothed group trajectories (Excel)",
            data=st.session_state["smoothed_groups_excel_cached"],
            file_name="smoothed_group_trajectories.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        corr_df = st.session_state["corr_df_cached"]

        st.subheader("Pearson correlation matrix")
        if corr_df.empty:
            st.warning("Correlation matrix is empty for current selection.")
        else:
            st.dataframe(corr_df)
            st.download_button(
                "Download correlation matrix (Excel)",
                data=_correlation_workbook(corr_df),
                file_name="correlation_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            dataset_names = list(results_by_dataset.keys())
            if len(dataset_names) >= 2:
                cross_corr = _cross_dataset_corr_matrix(corr_df, dataset_names[0], dataset_names[1])
                if not cross_corr.empty:
                    st.markdown(f"**Cross-dataset correlations: {dataset_names[0]} vs {dataset_names[1]}**")
                    st.dataframe(cross_corr)
                    st.download_button(
                        f"Download cross-dataset correlation ({dataset_names[0]} vs {dataset_names[1]})",
                        data=_correlation_workbook(cross_corr),
                        file_name=f"correlation_{dataset_names[0]}_vs_{dataset_names[1]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

        st.subheader("Cluster and factor analysis")
        sklearn_ok, sklearn_error = _check_sklearn_available()
        if not sklearn_ok:
            st.warning(
                "`sklearn` is not available in the same Python interpreter that runs Streamlit. "
                f"Interpreter: `{sys.executable}`. "
                f"Run: `{sys.executable} -m pip install scikit-learn`. "
                f"Import error: {sklearn_error}"
            )

        c1, c2 = st.columns(2)
        dataset_for_cluster = c1.selectbox("Dataset", options=list(results_by_dataset.keys()), key="cluster_dataset")
        corpus_for_cluster = c2.selectbox(
            "Corpus",
            options=list(results_by_dataset[dataset_for_cluster].keys()),
            key="cluster_corpus",
        )
        st.caption("PCA plot shows years projected to first two principal components; clusters are used only for coloring.")

        if st.button("Run cluster/factor analysis", disabled=not sklearn_ok):
            try:
                analysis_artifacts = _run_cluster_factor_analysis(
                    results_by_dataset[dataset_for_cluster][corpus_for_cluster].group_timeseries,
                    selected_groups,
                    [str(y) for y in year_cols if year_start <= y <= year_end],
                    smoothing_window,
                )
                st.session_state["cluster_artifacts"] = analysis_artifacts
            except Exception as e:
                st.error(str(e))

        if "cluster_artifacts" in st.session_state:
            arts = st.session_state["cluster_artifacts"]
            st.image(arts["03_years_in_pca_space.png"], caption="03_years_in_pca_space.png")
            st.markdown("**Interpretation of principal coordinates (years PCA)**")
            st.write(arts["pca_interpretation"])
            st.caption("`PC1_abs` / `PC2_abs` are absolute loadings: contribution strength to PC1/PC2 regardless of sign.")
            st.dataframe(arts["pca_loadings"])
            st.image(arts["04_moral_groups_pca_space.png"], caption="04_moral_groups_pca_space.png")
            st.markdown("**Interpretation of principal coordinates (groups PCA)**")
            st.write(arts["groups_pca_interpretation"])

            st.download_button(
                "Download 03_years_in_pca_space.png",
                data=arts["03_years_in_pca_space.png"],
                file_name="03_years_in_pca_space.png",
                mime="image/png",
            )
            st.download_button(
                "Download PCA loadings (Excel)",
                data=_correlation_workbook(arts["pca_loadings"].set_index("group")),
                file_name="pca_loadings.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.download_button(
                "Download 04_moral_groups_pca_space.png",
                data=arts["04_moral_groups_pca_space.png"],
                file_name="04_moral_groups_pca_space.png",
                mime="image/png",
            )

        with st.expander("Preview aggregated tables"):
            for dataset_name, dataset_results in results_by_dataset.items():
                st.markdown(f"### {dataset_name}")
                for corpus, res in dataset_results.items():
                    st.subheader(f"{corpus}: groups")
                    st.dataframe(res.group_timeseries.head())
else:
    st.info("Upload at least one dataset (A or B): dictionary + frequency file(s).")
