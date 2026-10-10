"""Shared 04 split: last ~20% of TRAIN weeks (chronological) are validation."""
import pandas as pd
from data import ROOT

VAL_FRACTION = 0.2


def load_index():
    idx = pd.read_csv(ROOT / "charts/index.csv")
    weeks = sorted(idx.loc[idx.split == "train", "end_date"].unique())
    n_val = round(len(weeks) * VAL_FRACTION)
    val = set(weeks[-n_val:])
    idx["fit_split"] = idx.split
    idx.loc[idx.end_date.isin(val), "fit_split"] = "val"
    idx.loc[(idx.split == "train") & ~idx.end_date.isin(val), "fit_split"] = "fit"
    return idx
