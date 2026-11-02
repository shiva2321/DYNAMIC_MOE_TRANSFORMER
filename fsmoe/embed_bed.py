"""One idempotent command that embeds the whole test bed (blocks already on disk are skipped, so a power cut costs one block): ``python -m fsmoe.embed_bed [--seed 700]``."""

import argparse
import time

from .bed import build_bed, default_readers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    args = ap.parse_args()
    readers = default_readers()
    for seed in args.seeds:
        t0 = time.time()
        bed = build_bed(seed, readers)
        print(f"seed {seed}: {len(bed.text)} rows, {len(bed.skills)} skills, far sets {sorted(bed.far)}, {time.time() - t0:.0f}s", flush=True)
    for rd in readers:
        rd.close()


if __name__ == "__main__":
    main()
