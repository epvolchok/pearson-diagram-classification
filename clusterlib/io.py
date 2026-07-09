#Copyright (c) 2026 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import numpy as np



_BOOT_SCALARS = {'stability_mean', 'ari_mean', 'ari_std', 'cover_mean', 'pres_mean'}


def save_bootstrap(boot: dict, path: str) -> str:
    """Save the output of bootstrap_stability or seed_stability to .npz.

    Returns the resolved file path (with .npz extension added if missing).
    """
    if not path.endswith('.npz'):
        path += '.npz'
    np.savez_compressed(path, **boot)
    return path


def load_bootstrap(path: str) -> dict:
    """Load a bootstrap / seed stability result from .npz back into a dict.

    Scalar values (stored as 0-d arrays) are unwrapped to Python floats.
    """
    if not path.endswith('.npz'):
        path += '.npz'
    with np.load(path, allow_pickle=False) as data:
        result = {}
        for key in data.files:
            arr = data[key]
            result[key] = float(arr) if key in _BOOT_SCALARS else arr
    return result



def save_stability_field(res: dict, path: str) -> str:
    """Save the output of compute_stability_field to .npz."""
    if not path.endswith('.npz'):
        path += '.npz'
    np.savez_compressed(
        path,
        mean=res['mean'],
        std=res['std'],
        runs=res['runs'],
        P=res['P'],
        ari=np.asarray(res['ari']),
        cover=np.asarray(res['cover']),
    )
    print(f"Saved: {path}")
    return path


def load_stability_field(path: str) -> dict:
    """Load a stability field from .npz into a dict matching compute_stability_field output."""
    if not path.endswith('.npz'):
        path += '.npz'
    with np.load(path) as data:
        return {
            'mean':  data['mean'],
            'std':   data['std'],
            'runs':  data['runs'],
            'P':     data['P'],
            'ari':   data['ari'].tolist(),
            'cover': data['cover'].tolist(),
        }



def save_stability_vs_k(res_by_k: dict, path: str) -> str:
    """Save the output of stability_vs_k / grid_stability_vs_k to .npz.

    Keys are encoded as ``'k{K}_{field}'``.
    """
    if not path.endswith('.npz'):
        path += '.npz'
    flat: dict = {}
    for k, res in res_by_k.items():
        for field in ('mean', 'std', 'runs', 'P', 'ari', 'cover'):
            if field in res:
                flat[f'k{k}_{field}'] = np.asarray(res[field])
    flat['k_values'] = np.array(sorted(res_by_k.keys()))
    np.savez_compressed(path, **flat)
    print(f"Saved: {path}  (k = {sorted(res_by_k.keys())})")
    return path


def load_stability_vs_k(path: str) -> dict:
    """Load a k-indexed stability result back into a nested dict."""
    if not path.endswith('.npz'):
        path += '.npz'
    with np.load(path) as data:
        k_values = data['k_values']
        out: dict = {}
        for k in k_values:
            k = int(k)
            entry: dict = {}
            for field in ('mean', 'std', 'runs', 'P', 'ari', 'cover'):
                key = f'k{k}_{field}'
                if key in data:
                    entry[field] = data[key]
            out[k] = entry
    return out
