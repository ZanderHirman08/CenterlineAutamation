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
        print(f"Imagery: {cfg.inputs.imagery}")
        print(f"  CRS {reader.crs.to_string()}, pixel size {reader.res:.3g}, "
              f"bands {reader.band_indexes} (NIR: {'yes' if reader.has_nir else 'no'})")
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


if __name__ == "__main__":
    sys.exit(main())
