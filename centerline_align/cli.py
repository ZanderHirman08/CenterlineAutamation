"""Command line: ``python -m centerline_align <command> config.yaml``."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import Config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="centerline_align",
                                 description="Align road centerlines to imagery.")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, help_ in (("run", "train if needed, refine every grid, write outputs"),
                        ("train", "only train and save the road classifier"),
                        ("check", "validate the config and inputs without processing")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", help="path to the YAML config")
        p.add_argument("--grids", nargs="+", help="only these grid ids (overrides config)")
        p.add_argument("-v", "--verbose", action="store_true")
        if name == "run":
            p.add_argument("--fresh", action="store_true",
                           help="ignore finished grids and the saved classifier; redo everything")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = Config.load(args.config)
    if args.grids:
        cfg.grid_ids = args.grids

    from . import pipeline

    if args.command == "check":
        return _check(cfg)
    if args.command == "train":
        pipeline.run(cfg, resume=False, train_only=True)
        return 0
    if args.fresh:
        import shutil
        from pathlib import Path
        grid_dir = Path(cfg.output.folder) / "grids"
        if grid_dir.exists():
            shutil.rmtree(grid_dir)
        (Path(cfg.output.folder) / "grid_summary.csv").unlink(missing_ok=True)
    pipeline.run(cfg, resume=not args.fresh)
    return 0


def _check(cfg: Config) -> int:
    from .imagery import ImageryReader
    from .pipeline import load_vectors

    reader = ImageryReader(cfg)
    try:
        roads, grids = load_vectors(cfg, reader.crs)
        print(f"Imagery: {cfg.inputs.imagery}" +
              (f" -> {reader.path}" if reader.path != cfg.inputs.imagery else ""))
        units = reader.crs.linear_units or "CRS units"
        print(f"  CRS {reader.crs.to_string()} ({units}), pixel size {reader.res:.3g}, "
              f"bands {reader.band_indexes} (NIR: {'yes' if reader.has_nir else 'no'})")
        print(f"  All distances in the config are in {units}.")
        _band_report(reader, grids)
        print(f"Roads: {len(roads)} line pieces, total length {roads.length.sum():,.0f}")
        print(f"Grids: {len(grids)}")
        ib = reader.bounds
        from shapely.geometry import box
        cover = box(*ib)
        inside = roads.intersects(cover).mean() if len(roads) else 0
        print(f"Roads touching the imagery extent: {inside:.0%}")
        if inside < 0.5:
            print("WARNING: most roads fall outside the imagery - check CRS / coverage")
    finally:
        reader.close()
    return 0


def _band_report(reader, grids):
    """Coarse look at band values in one grid, to sanity-check the band setup."""
    import numpy as np

    minx, miny, maxx, maxy = grids.total_bounds if len(grids) else reader.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = min(2640.0, (maxx - minx) / 2)   # about one 1-mile grid around the centre
    res = max(reader.res, 2 * half / 512)
    n = int(2 * half / res)
    chip = reader.read(cx - half, cy + half, n, n, res=res)
    if not chip.valid.any():
        print("  (no imagery at the centre of the grids to sample)")
        return
    v = chip.data[:, chip.valid]
    names = ["red", "green", "blue", "nir"][: len(v)]
    print("  Sample band means: " + ", ".join(f"{k} {m:.0f}" for k, m in zip(names, v.mean(1))))
    if reader.has_nir:
        brighter = float((v[3] > v[0]).mean())
        print(f"  NIR brighter than red on {brighter:.0%} of pixels.")
        if brighter < 0.25:
            print("  NOTE: band 4 is mostly darker than red. In 8-bit orthos each band is often\n"
                  "  stretched separately, so this alone doesn't mean band 4 isn't NIR. Check a\n"
                  "  4-1-2 false-colour view in Pro (vegetation should look red). The classifier\n"
                  "  learns how much to rely on band 4 either way.")


if __name__ == "__main__":
    sys.exit(main())
