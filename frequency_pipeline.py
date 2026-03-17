from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


AGGREGATION_MODES = {
    "base",
    "pairs",
    "individualism_collectivism",
    "virtue_vice",
}


def infer_corpus_name_from_filename(file_path: Path) -> str:
    """Infer compact corpus label like MDF_en-GB / MDF2_en-GB from frequency filename."""
    stem = Path(file_path).stem
    lower_stem = stem.lower()

    dataset_tag = ""
    if re.search(r"(?:^|[_-])(mfd2|mdf2)(?:[_-]|$)", lower_stem):
        dataset_tag = "MDF2"
    elif re.search(r"(?:^|[_-])(mfd|mdf)(?:[_-]|$)", lower_stem):
        dataset_tag = "MDF"

    remainder = re.sub(r"(?i)^word[_-]?freq[_-]?(mfd2|mdf2|mfd|mdf)[_-]?", "", stem)
    if remainder == stem:
        remainder = re.sub(r"(?i)^word[_-]?freq[_-]?", "", stem)
    remainder = remainder.lstrip("-_").strip()

    lang_match = re.search(r"(?i)(en[-_][a-z0-9]+)", remainder)
    if lang_match:
        corpus = lang_match.group(1).replace("_", "-")
    else:
        corpus = remainder if remainder else stem

    if dataset_tag:
        return f"{dataset_tag}_{corpus}"
    return corpus


@dataclass
class CorpusResult:
    """Aggregated time-series for one corpus."""

    corpus_name: str
    group_timeseries: pd.DataFrame
    term_timeseries: pd.DataFrame


class MoralFrequencyAnalyzer:
    """Pipeline for scaling and aggregating frequency trajectories."""

    def __init__(self, dictionary_path: Path, frequency_paths: Dict[str, Path]):
        self.dictionary_path = Path(dictionary_path)
        self.frequency_paths = {k: Path(v) for k, v in frequency_paths.items()}
        self.dictionary = self._load_dictionary(self.dictionary_path)

    @staticmethod
    def _load_dictionary(path: Path) -> pd.DataFrame:
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]

        required = {"word", "group"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dictionary is missing required columns: {missing}")

        # MFD has `stem`, MFD2 can have no `stem`.
        if "stem" not in df.columns:
            df["stem"] = ""

        df = df.copy()
        df["stem"] = df["stem"].fillna("").astype(str).str.strip()
        df["word"] = df["word"].fillna("").astype(str).str.strip()
        df["group"] = df["group"].fillna("").astype(str).str.strip()
        return df

    @staticmethod
    def _year_columns(df: pd.DataFrame) -> List[str]:
        cols: List[str] = []
        for col in df.columns:
            col_s = str(col)
            if col_s.isdigit():
                cols.append(col_s)
        if not cols:
            raise ValueError("No year columns found in frequency table (e.g., 1800..2022).")
        return cols

    def _build_terms(
        self,
        freq_df: pd.DataFrame,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build one scaled trajectory per term:
        - whole word: scale own trajectory to peak=100
        - stem (*): sum preselected rows for this stem, then scale to peak=100
        - if dictionary has no stems (MFD2), process all rows as whole words
        """
        freq_df = freq_df.copy()
        freq_df.columns = [str(c).strip() for c in freq_df.columns]
        year_cols = self._year_columns(freq_df)

        year_int = [int(y) for y in year_cols]
        if year_start is None:
            year_start = min(year_int)
        if year_end is None:
            year_end = max(year_int)

        peak_year_cols = [str(y) for y in year_int if year_start <= y <= year_end]
        if not peak_year_cols:
            raise ValueError("Selected interval does not intersect available year columns.")

        if len(freq_df) != len(self.dictionary):
            raise ValueError(
                f"Row count mismatch: dictionary={len(self.dictionary)}, frequencies={len(freq_df)}"
            )

        merged = pd.concat([self.dictionary.reset_index(drop=True), freq_df.reset_index(drop=True)], axis=1)
        for yc in year_cols:
            merged[yc] = pd.to_numeric(merged[yc], errors="coerce").fillna(0.0)

        term_rows: List[dict] = []

        whole_mask = ~merged["stem"].astype(str).str.endswith("*")
        for idx, row in merged[whole_mask].iterrows():
            raw = row[year_cols].astype(float)
            peak = raw[peak_year_cols].max(skipna=True)
            peak_year = ""
            if pd.isna(peak) or peak <= 0:
                scaled = pd.Series([0.0] * len(raw), index=raw.index)
                peak = 0.0
            else:
                scaled = raw / peak * 100.0
                peak_year = str(raw[peak_year_cols].astype(float).idxmax())

            term_id = row["word"] if row["word"] else f"row_{idx}"
            payload = {
                "term_id": term_id,
                "group": row["group"],
                "source_type": "word",
                "scale_peak": float(peak),
                "scale_peak_year": peak_year,
            }
            payload.update(scaled.to_dict())
            term_rows.append(payload)

        stem_mask = merged["stem"].astype(str).str.endswith("*")
        stem_df = merged[stem_mask].copy()
        for stem, block in stem_df.groupby("stem", sort=False):
            summed = block[year_cols].sum(axis=0)
            peak = summed[peak_year_cols].max(skipna=True)
            peak_year = ""
            if pd.isna(peak) or peak <= 0:
                scaled = pd.Series([0.0] * len(summed), index=summed.index)
                peak = 0.0
            else:
                scaled = summed / peak * 100.0
                peak_year = str(summed[peak_year_cols].astype(float).idxmax())

            group = block["group"].mode().iloc[0] if not block["group"].empty else ""
            payload = {
                "term_id": stem,
                "group": group,
                "source_type": "stem_selected_rows",
                "scale_peak": float(peak),
                "scale_peak_year": peak_year,
            }
            payload.update(scaled.to_dict())
            term_rows.append(payload)

        terms_df = pd.DataFrame(term_rows)
        return terms_df[["term_id", "group", "source_type", "scale_peak", "scale_peak_year", *year_cols]]

    @staticmethod
    def _pair_base_name(group_name: str) -> str:
        for suffix in ("Virtue", "Vice"):
            if group_name.endswith(suffix):
                return group_name[: -len(suffix)]
        return group_name

    @staticmethod
    def _make_group_index_map(group_series: pd.Series) -> Dict[str, int]:
        ordered = list(dict.fromkeys(group_series.astype(str).tolist()))
        return {name: i + 1 for i, name in enumerate(ordered)}

    @staticmethod
    def _map_to_aggregation_label(
        group_base: str,
        idx: int,
        mode: str,
    ) -> str:
        if mode == "base":
            base_name_map = {
                1: "HarmVirtue",
                2: "HarmVice",
                3: "FairnessVirtue",
                4: "FairnessVice",
                5: "IngroupVirtue",
                6: "IngroupVice",
                7: "AuthorityVirtue",
                8: "AuthorityVice",
                9: "PurityVirtue",
                10: "PurityVice",
                11: "MoralityGeneral",
            }
            return base_name_map.get(idx, group_base)
        if mode == "pairs":
            pair_start = idx if idx % 2 == 1 else idx - 1
            pair_name_map = {
                1: "Harm",
                3: "Fairness",
                5: "Ingroup",
                7: "Authority",
                9: "Purity",
                11: "General",
            }
            if pair_start in pair_name_map:
                return pair_name_map[pair_start]
            return group_base
        if mode == "individualism_collectivism":
            if 1 <= idx <= 4:
                return "Individualism"
            if 5 <= idx <= 10:
                return "Collectivism"
            return f"Other_{idx}"
        if mode == "virtue_vice":
            if 1 <= idx <= 10:
                return "Virtue" if idx % 2 == 1 else "Vice"
            return f"Other_{idx}"
        raise ValueError(f"Unknown aggregation mode: {mode}")

    def _aggregate_groups(self, terms_df: pd.DataFrame, aggregation_mode: str = "base") -> pd.DataFrame:
        if aggregation_mode not in AGGREGATION_MODES:
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode}. Allowed: {sorted(AGGREGATION_MODES)}")

        year_cols = self._year_columns(terms_df)
        tmp = terms_df.copy()
        tmp["group_base"] = tmp["group"].apply(self._pair_base_name)

        index_map = self._make_group_index_map(tmp["group_base"])
        tmp["group"] = tmp["group_base"].astype(str).map(
            lambda g: self._map_to_aggregation_label(g, index_map[g], aggregation_mode)
        )

        grouped = (
            tmp.groupby("group", sort=False, dropna=False)[year_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        return grouped

    def analyze_corpus(
        self,
        corpus_name: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        combine_index_pairs: bool = False,
        aggregation_mode: str = "base",
    ) -> CorpusResult:
        if corpus_name not in self.frequency_paths:
            raise KeyError(f"Unknown corpus: {corpus_name}. Available: {list(self.frequency_paths)}")

        if combine_index_pairs and aggregation_mode == "base":
            aggregation_mode = "pairs"

        freq_df = pd.read_excel(self.frequency_paths[corpus_name])
        freq_df.columns = [str(c).strip() for c in freq_df.columns]

        terms_df = self._build_terms(freq_df, year_start=year_start, year_end=year_end)
        groups_df = self._aggregate_groups(terms_df, aggregation_mode=aggregation_mode)
        return CorpusResult(corpus_name, groups_df, terms_df)

    def analyze_many(
        self,
        corpora: Sequence[str],
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        combine_index_pairs: bool = False,
        aggregation_mode: str = "base",
    ) -> Dict[str, CorpusResult]:
        return {
            name: self.analyze_corpus(
                name,
                year_start=year_start,
                year_end=year_end,
                combine_index_pairs=combine_index_pairs,
                aggregation_mode=aggregation_mode,
            )
            for name in corpora
        }


def _smooth(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values
    return values.rolling(window=window, center=True, min_periods=1).mean()


def plot_group_trajectories(
    results: Dict[str, CorpusResult],
    groups: Optional[Iterable[str]] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    one_plot_per_corpus: bool = True,
    combine_pairs: bool = False,
    smoothing_window: int = 1,
):
    if not results:
        raise ValueError("No data to plot.")

    first = next(iter(results.values())).group_timeseries
    year_cols = [c for c in first.columns if str(c).isdigit()]
    years = [int(y) for y in year_cols]

    if year_start is None:
        year_start = min(years)
    if year_end is None:
        year_end = max(years)

    selected_years = [str(y) for y in years if year_start <= y <= year_end]
    selected_year_int = [int(y) for y in selected_years]
    requested = set(groups) if groups else None

    if one_plot_per_corpus:
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)), sharex=True)
        if len(results) == 1:
            axes = [axes]

        for ax, (corpus, res) in zip(axes, results.items()):
            gdf = res.group_timeseries
            for _, row in gdf.iterrows():
                grp = str(row["group"])
                if requested and grp not in requested:
                    continue
                y = _smooth(row[selected_years].astype(float), smoothing_window)
                ax.plot(selected_year_int, y, label=grp)
            suffix = " (pairs)" if combine_pairs else ""
            ax.set_title(f"{corpus}: trajectories by moral group{suffix}")
            ax.set_ylabel("Scaled frequency (peak=100)")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", ncol=2, fontsize=8)
        axes[-1].set_xlabel("Year")
        plt.tight_layout()
        plt.show()
        return

    plt.figure(figsize=(13, 7))
    for corpus, res in results.items():
        gdf = res.group_timeseries
        for _, row in gdf.iterrows():
            grp = str(row["group"])
            if requested and grp not in requested:
                continue
            y = _smooth(row[selected_years].astype(float), smoothing_window)
            plt.plot(selected_year_int, y, label=f"{corpus}:{grp}")
    plt.title("Moral groups trajectories")
    plt.xlabel("Year")
    plt.ylabel("Scaled frequency (peak=100)")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def parse_frequency_paths(files: List[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for file in files:
        p = Path(file)
        corpus = infer_corpus_name_from_filename(p)
        if corpus in mapping:
            corpus = f"{corpus}__{p.stem}"
        mapping[corpus] = p
    return mapping


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Moral frequency processing pipeline")
    p.add_argument("--dictionary", required=True, help="Path to dictionary Excel with columns word/group/(optional stem)")
    p.add_argument(
        "--frequencies",
        nargs="+",
        required=True,
        help="Frequency Excel files (e.g., Word_freq_MDF_en-GB.xlsx ...)",
    )
    p.add_argument("--corpora", nargs="*", default=None, help="Corpora to plot (default: all)")
    p.add_argument("--groups", nargs="*", default=None, help="Group names to plot")
    p.add_argument("--year-start", type=int, default=None)
    p.add_argument("--year-end", type=int, default=None)
    p.add_argument("--single-figure", action="store_true", help="Draw all corpora in one figure")
    p.add_argument("--combine-index-pairs", action="store_true", help="Backward-compatible alias for --aggregation-mode pairs")
    p.add_argument(
        "--aggregation-mode",
        default="base",
        choices=sorted(AGGREGATION_MODES),
        help="Group aggregation mode: base, pairs, individualism_collectivism, virtue_vice",
    )
    p.add_argument("--smoothing-window", type=int, default=1, help="Rolling mean window in years")
    p.add_argument("--save-csv-dir", default=None, help="Folder to save aggregated csv files")
    return p


def main():
    args = build_arg_parser().parse_args()
    freq_paths = parse_frequency_paths(args.frequencies)
    analyzer = MoralFrequencyAnalyzer(Path(args.dictionary), freq_paths)

    corpora = args.corpora if args.corpora else list(freq_paths.keys())
    aggregation_mode = "pairs" if args.combine_index_pairs and args.aggregation_mode == "base" else args.aggregation_mode
    results = analyzer.analyze_many(
        corpora,
        year_start=args.year_start,
        year_end=args.year_end,
        combine_index_pairs=args.combine_index_pairs,
        aggregation_mode=aggregation_mode,
    )

    if args.save_csv_dir:
        out_dir = Path(args.save_csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for corpus, res in results.items():
            res.group_timeseries.to_csv(out_dir / f"groups_{corpus}.csv", index=False)
            res.term_timeseries.to_csv(out_dir / f"terms_{corpus}.csv", index=False)

    plot_group_trajectories(
        results,
        groups=args.groups,
        year_start=args.year_start,
        year_end=args.year_end,
        one_plot_per_corpus=not args.single_figure,
        combine_pairs=aggregation_mode == "pairs",
        smoothing_window=max(1, int(args.smoothing_window)),
    )


if __name__ == "__main__":
    main()
