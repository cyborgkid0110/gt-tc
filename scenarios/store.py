"""Persist and load a deployment scenario as a CSV file.

Format (one file per scenario):

    # gt-tc-scenario v1
    # source=poisson num_nodes=200 seed=42 area=250 bs_x=0.0 bs_y=0.0 coverage_radius=
    x,y,vpre
    12.3,-45.6,3.21
    ...

The node table is pure floats (loadable with numpy). Provenance and the
base-station position live in `#` header comments so the table stays homogeneous.
"""
import csv

import numpy as np

SCHEMA_VERSION = "v1"


def save_scenario(path, positions, vpre, bs_pos, meta):
    """Write positions (N,2), vpre (N,), bs_pos (2,) and a meta dict to `path`."""
    positions = np.asarray(positions, dtype=float)
    vpre = np.asarray(vpre, dtype=float)
    assert positions.shape[0] == vpre.shape[0], "positions/vpre length mismatch"

    full_meta = dict(meta)
    full_meta['bs_x'] = float(bs_pos[0])
    full_meta['bs_y'] = float(bs_pos[1])
    for k, v in full_meta.items():
        if ' ' in str(v):
            raise ValueError(f"meta value for '{k}' contains a space: {v!r} "
                             "(the meta header line is space-delimited)")
    meta_str = " ".join(f"{k}={full_meta[k]}" for k in full_meta)

    with open(path, 'w', newline='') as f:
        f.write(f"# gt-tc-scenario {SCHEMA_VERSION}\n")
        f.write(f"# {meta_str}\n")
        w = csv.writer(f)
        w.writerow(['x', 'y', 'vpre'])
        for (x, y), v in zip(positions, vpre):
            w.writerow([repr(float(x)), repr(float(y)), repr(float(v))])


def load_scenario(path):
    """Return (positions (N,2), vpre (N,), bs_pos (2,), meta dict).

    All meta values are returned as strings; callers must cast as needed
    (e.g. ``int(meta['num_nodes'])``).
    """
    meta = {}
    skip = 0
    col_names = None
    with open(path) as f:
        for line in f:
            skip += 1
            if line.startswith('#'):
                # parse key=value tokens from meta comment lines
                body = line[1:].strip()
                if not body.startswith('gt-tc-scenario'):
                    for tok in body.split():
                        if '=' in tok:
                            k, v = tok.split('=', 1)
                            meta[k] = v
                continue
            # first non-comment line is the column-header row
            col_names = [c.strip() for c in line.split(',')]
            break

    if col_names is None:
        raise ValueError(f"No column-header row found in {path}")

    # skip_header skips all comment lines + the column-header row so only
    # float data rows remain; names= attaches the column labels read above.
    table = np.genfromtxt(path, delimiter=',', skip_header=skip, names=col_names)
    # np.genfromtxt returns a 0-d structured array for a single data row;
    # normalise to at least 1-d so np.column_stack works correctly.
    table = np.atleast_1d(table)
    positions = np.column_stack([table['x'], table['y']]).astype(float)
    vpre = np.asarray(table['vpre'], dtype=float)
    bs_pos = (float(meta.get('bs_x', 0.0)), float(meta.get('bs_y', 0.0)))
    return positions, vpre, bs_pos, meta
