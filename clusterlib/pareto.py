#Copyright (c) 2026 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pandas as pd


def pareto_front(df: pd.DataFrame, objectives: list, maximize: bool = True) -> np.ndarray:
    """Identify non-dominated rows in a multi-objective DataFrame.

    A row is on the Pareto front if no other row dominates it, i.e. no other
    row is at least as good on all objectives and strictly better on at least
    one.

    Parameters
    ----------
    df : pd.DataFrame
    objectives : list of str
        Column names to use as objectives.
    maximize : bool
        If True (default), all objectives are maximised.  Pass False to
        minimise all objectives, or negate individual columns beforehand for
        mixed directions.

    Returns
    -------
    on_front : np.ndarray of bool, shape (len(df),)
    """
    vals = df[objectives].values.astype(float)
    if not maximize:
        vals = -vals

    on_front = np.ones(len(vals), dtype=bool)
    for i in range(len(vals)):
        if not on_front[i]:
            continue
        dominated_by = (np.all(vals >= vals[i], axis=1) &
                        np.any(vals > vals[i],  axis=1))
        if dominated_by.any():
            on_front[i] = False

    return on_front


def save_front(front: pd.DataFrame, path: str) -> None:
    """Save a Pareto front DataFrame to CSV or Parquet.

    The format is inferred from the file extension (.parquet -> Parquet,
    everything else -> CSV).  The original run index is preserved.
    """
    if path.endswith('.parquet'):
        front.to_parquet(path, index=True)
    else:
        if not path.endswith('.csv'):
            path += '.csv'
        front.to_csv(path, index=True, index_label='run_id')
    print(f"Saved: {path}  ({len(front)} rows on front)")


def load_front(path: str) -> pd.DataFrame:
    """Load a Pareto front from CSV or Parquet back into a DataFrame."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    if not path.endswith('.csv'):
        path += '.csv'
    return pd.read_csv(path, index_col='run_id')
