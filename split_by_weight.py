"""Split muons_data.h5 into two datasets by weight threshold W: 'below_W' (weight < W)
and 'above_W' (weight >= W), each holding its own copy of every column. Streamed in
chunks, so peak memory is one chunk regardless of file size.

Run: python split_by_weight.py 1.0
"""
import argparse

import h5py

from utils import DEFAULT_DATA, CHUNK, COLS


def split(path, out_path, W, chunk=CHUNK):
    with h5py.File(path, 'r') as fin, h5py.File(out_path, 'w') as fout:
        n = fin[COLS[0]].shape[0]
        groups = {name: fout.create_group(name) for name in ('below_W', 'above_W')}
        ds = {name: {c: g.create_dataset(c, shape=(0,), maxshape=(None,),
                                         dtype=fin[c].dtype, chunks=True)
                    for c in COLS} for name, g in groups.items()}
        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            mask = fin['weight'][sl] < W
            for name, m in (('below_W', mask), ('above_W', ~mask)):
                if not m.any():
                    continue
                for c in COLS:
                    vals = fin[c][sl][m]
                    d = ds[name][c]
                    old = d.shape[0]
                    d.resize(old + vals.shape[0], axis=0)
                    d[old:] = vals
            print(f"  {sl.stop:,}/{n:,}", end='\r', flush=True)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('W', type=float, help="Weight threshold to split on")
    parser.add_argument('--data', default=DEFAULT_DATA, help="Input muon file")
    parser.add_argument('--out', default='muons_data_split.h5', help="Output file")
    args = parser.parse_args()
    split(args.data, args.out, args.W)
    print(f"Saved {args.out}")
