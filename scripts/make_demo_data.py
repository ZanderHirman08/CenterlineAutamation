"""Write a small synthetic ortho, offset local roads and two grids to ./demo.

    python scripts/make_demo_data.py
    python -m centerline_align run demo/config.yaml
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.synthetic import write_scene  # noqa: E402

out = ROOT / "demo"
paths = write_scene(out)
(out / "config.yaml").write_text(f"""inputs:
  imagery: {paths['imagery'].as_posix()}
  roads: {paths['roads'].as_posix()}
  grids: {paths['grids'].as_posix()}
  road_id_field: RD_ID
  grid_id_field: GRID_ID
classifier:
  training_grids: 2
  samples_per_class: 6000
output:
  folder: {(out / 'output').as_posix()}
  write_stations: true
  write_probability: true
""")
print(f"Demo data written to {out}. Run:\n  python -m centerline_align run {out.as_posix()}/config.yaml")
