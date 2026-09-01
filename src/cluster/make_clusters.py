#!/usr/bin/env python3
"""Pick a CROWN cluster metric and a cutoff, get cluster labels. One tool, all metrics.

Every metric (sequence, pocket, protein-ligand, ligand) is stored as a minimum spanning
tree; cutting a tree at a similarity threshold IS single-linkage clustering. This wraps
that in a small command line so you never touch the MST internals.

    # what's available, and the valid cutoff range for each metric
    python make_clusters.py list

    # (re)build the one combined MST file from the per-metric outputs
    python make_clusters.py combine

    # summarise a single cut: how many clusters, singletons, biggest cluster
    python make_clusters.py label --metric pli-sim --threshold 0.7

    # sweep several cutoffs and save the labels
    python make_clusters.py label --metric lig-sim --thresholds 0.5 0.7 0.9 --out lig_clusters.parquet

    # add the new column(s) onto the metadata table (writes a copy by default)
    python make_clusters.py label --metric seq-sim --thresholds 0.7 --annotate

Metric names: seq-sim, pocket-sim (nodes = complex basename); pli-sim (basename);
lig-sim (nodes = lig_name). A metric is available once its clustering script has run.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.cluster import crown_mst as cm
from src.config import DATA_DIR

META_DIR = f'{DATA_DIR}/metadata'
METADATA = f'{META_DIR}/CROWN_metadata.parquet'


def _cluster_stats(labels):
    """(n_clusters, n_singletons, largest_cluster_size) for an int label array."""
    _, sizes = np.unique(labels, return_counts=True)
    return len(sizes), int((sizes == 1).sum()), int(sizes.max())


# ------------------------------------------------------------------ subcommands

def cmd_list(args):
    present = cm.available_metrics()
    if not present:
        print("No MST source files found in", META_DIR)
        print("Run cluster_mmseqs / cluster_plec / cluster_ecfp4 first.")
        return

    combined = os.path.exists(cm.COMBINED_MST)
    print(f"Combined MST: {'present' if combined else 'NOT built (run: make_clusters.py combine)'}\n")
    _, meta = cm.load_combined() if combined else (None, None)

    print(f"{'metric':12s} {'nodes':>9s} {'edges':>9s} {'sim range':>15s} "
          f"{'valid cut >=':>12s}  maps onto")
    print("-" * 74)
    for m in present:
        s = cm.SOURCES[m]
        if meta and m in meta:
            info = meta[m]
            rng = f"{info['sim_min']:.3f}-{info['sim_max']:.3f}"
            print(f"{m:12s} {info['n_nodes']:>9,} {info['n_edges']:>9,} {rng:>15s} "
                  f"{info['edge_floor']:>12.2f}  {info['id_column']}")
        else:
            print(f"{m:12s} {'?':>9s} {'?':>9s} {'?':>15s} "
                  f"{s['edge_floor']:>12.2f}  {s['id_column']}  (combine to inspect)")


def cmd_combine(args):
    floors = dict(zip(args.floor_metric or [], args.floor_value or [])) or None
    cm.combine_msts(floors=floors)


def cmd_label(args):
    mst, meta = cm.load_combined()
    if args.metric not in meta:
        raise SystemExit(f"metric '{args.metric}' not available; have {list(meta)}. "
                         f"Did its clustering script run, then `combine`?")

    id_column = meta[args.metric]['id_column']
    ids = meta[args.metric]['ids']
    out = pd.DataFrame({id_column: ids})

    for t in args.thresholds:
        ids_, labels, _ = cm.labels_for(args.metric, t, mst=mst, meta=meta,
                                         force = True)
        n_clu, n_single, largest = _cluster_stats(labels)
        col = f'{t} {args.metric} cluster'
        out[col] = labels
        print(f"{args.metric} @ {t}: {n_clu:,} clusters "
              f"({n_single:,} singletons, largest {largest:,})")

    if not os.path.exists(METADATA):
        raise SystemExit(f"{METADATA} not found; cannot annotate.")
    md = pd.read_parquet(METADATA)
    if id_column not in md.columns:
        raise SystemExit(f"metadata has no '{id_column}' column to merge on.")
    label_cols = [c for c in out.columns if c != id_column]
    md = md.drop(columns=[c for c in label_cols if c in md.columns])  # refresh
    md = md.merge(out, on=id_column, how='left')
    md.to_parquet(METADATA, index=False)
    print(f"Annotated metadata written to {METADATA}")

# ------------------------------------------------------------------ arg parsing

def build_parser():
    p = argparse.ArgumentParser(
        description="Pick a CROWN cluster metric and cutoff, get single-linkage labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('list', help='show available metrics and their valid cutoff ranges'
                   ).set_defaults(func=cmd_list)

    c = sub.add_parser('combine', help='(re)build the combined MST from per-metric outputs')
    c.add_argument('--floor-metric', nargs='*', help='metric name(s) to override edge_floor')
    c.add_argument('--floor-value', nargs='*', type=float, help='matching floor value(s)')
    c.set_defaults(func=cmd_combine)

    l = sub.add_parser('label', help='cut a metric at one or more thresholds into labels')
    l.add_argument('--metric', required=True, help='seq-sim | pocket-sim | pli-sim | lig-sim')
    g = l.add_mutually_exclusive_group(required=True)
    g.add_argument('--threshold', type=float, dest='threshold',
                   help='single similarity cutoff')
    g.add_argument('--thresholds', type=float, nargs='+',
                   help='several cutoffs (sweep)')
    return p


def main():
    args = build_parser().parse_args()
    if getattr(args, 'threshold', None) is not None:  # normalise to a list
        args.thresholds = [args.threshold]
    args.func(args)


if __name__ == '__main__':
    main()
