"""Canonical chronological fit/validation split with boundary purging."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VAL_FRACTION = 0.2
REQUIRED = {"file", "ticker", "split", "week_start", "end_date", "target_date", "label"}


def assign_fit_split(index: pd.DataFrame, val_fraction: float = VAL_FRACTION) -> pd.DataFrame:
    missing = REQUIRED - set(index.columns)
    if missing or index.empty:
        raise ValueError(f"Empty or incomplete chart index: missing {sorted(missing)}")
    idx = index.copy()
    if idx[list(REQUIRED)].isna().any().any() or idx.file.duplicated().any():
        raise ValueError("Missing chart identities or duplicate files")
    if idx.duplicated(["ticker", "end_date"]).any():
        raise ValueError("Duplicate ticker/week")
    if not idx.label.isin(["Up", "Flat", "Down"]).all() or not idx.split.isin(["train", "test"]).all():
        raise ValueError("Unknown label or split")
    dates = idx[["week_start", "end_date", "target_date"]].apply(pd.to_datetime, errors="raise")
    if not ((dates.week_start <= dates.end_date) & (dates.end_date < dates.target_date)).all():
        raise ValueError("Invalid chart/target chronology")
    if (idx.groupby("end_date").split.nunique() != 1).any():
        raise ValueError("A week occurs in both train and test")
    train, test = idx.split.eq("train"), idx.split.eq("test")
    if not train.any() or not test.any() or dates.loc[train, "target_date"].max() >= dates.loc[test, "week_start"].min():
        raise ValueError("Train targets overlap the first test image or a split is empty")
    weeks = sorted(idx.loc[train, "end_date"].unique())
    if len(weeks) < 2 or not 0 < val_fraction < 1:
        raise ValueError("Validation requires at least two training weeks and a fraction in (0, 1)")
    n_val = max(1, min(len(weeks) - 1, round(len(weeks) * val_fraction)))
    val = train & idx.end_date.isin(weeks[-n_val:])
    idx["fit_split"] = "test"
    idx.loc[train & ~val, "fit_split"] = "fit"
    idx.loc[val, "fit_split"] = "val"
    boundary = dates.loc[val, "week_start"].min()
    idx.loc[idx.fit_split.eq("fit") & dates.target_date.ge(boundary), "fit_split"] = "purged"
    if not idx.fit_split.eq("fit").any():
        raise ValueError("Purging leaves no fitting rows")
    return idx


def load_index(path: Path | None = None):
    return assign_fit_split(pd.read_csv(path or ROOT / "charts/index.csv"))


def representative_fixtures(index: pd.DataFrame) -> pd.DataFrame:
    """Fit-only fixtures cover every ticker, available label, and temporal extremes."""
    chosen = set()
    fit = index[index.fit_split == "fit"]
    if fit.empty:
        raise ValueError("No pre-test fitting fixtures")
    for _, rows in fit.groupby("ticker", sort=True):
        rows = rows.sort_values(["end_date", "file"])
        chosen.update(rows.iloc[[0, (len(rows) - 1) // 2, len(rows) - 1]].file)
        for _, group in rows.groupby("label", sort=True):
            chosen.add(group.iloc[(len(group) - 1) // 2].file)
    return fit[fit.file.isin(chosen)].sort_values(["ticker", "end_date"]).reset_index(drop=True)
