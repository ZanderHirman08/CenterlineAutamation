"""Build a GDAL virtual mosaic (.vrt) from a folder of imagery tiles.

County orthos usually come as hundreds of GeoTIFF tiles. A VRT is a small XML
file that makes them read like one raster, without copying any pixels. This
does what ``gdalbuildvrt`` does for the simple case (same CRS, same bands) so
no separate GDAL install is needed. Sidecar ``.tif.aux.xml`` files are read by
GDAL automatically; keep them next to their tiles.
"""
from __future__ import annotations

import glob
import logging
import math
import os
from pathlib import Path
from xml.sax.saxutils import escape

import rasterio

log = logging.getLogger(__name__)

TILE_EXTS = (".tif", ".tiff", ".img", ".jp2")


def find_tiles(spec: str) -> list[Path]:
    """A folder (searched recursively), a glob pattern, or a single file."""
    p = Path(spec)
    if p.is_dir():
        files = [f for f in p.rglob("*") if f.suffix.lower() in TILE_EXTS]
    elif any(ch in spec for ch in "*?["):
        files = [Path(f) for f in glob.glob(spec, recursive=True)]
    else:
        return [p]
    sids = list(p.rglob("*.sid")) if p.is_dir() else []
    if not files and sids:
        raise ValueError(
            f"{spec} only has MrSID (.sid) files, which this GDAL build cannot read. "
            "Use the GeoTIFF copies, or convert with ArcGIS Pro 'Copy Raster' / gdal_translate."
        )
    return sorted(files)


def build_vrt(tiles: list[Path], out_path: Path) -> Path:
    if not tiles:
        raise ValueError("No imagery tiles found")
    infos = []
    for t in tiles:
        with rasterio.open(t) as src:
            if src.crs is None:
                raise ValueError(f"{t} has no CRS (missing .aux.xml / .prj / .tfw?)")
            infos.append(dict(path=t.resolve(), crs=src.crs, count=src.count,
                              dtype=src.dtypes[0], nodata=src.nodata, res=src.res,
                              bounds=src.bounds, w=src.width, h=src.height))
    crs = infos[0]["crs"]
    count = infos[0]["count"]
    for i in infos[1:]:
        if i["crs"] != crs:
            raise ValueError(f"{i['path']} is in a different CRS from {infos[0]['path']}; "
                             "reproject the tiles to one CRS first")
        if i["count"] != count:
            raise ValueError(f"{i['path']} has {i['count']} bands, expected {count}")
    resx = min(i["res"][0] for i in infos)
    resy = min(i["res"][1] for i in infos)
    minx = min(i["bounds"].left for i in infos)
    maxy = max(i["bounds"].top for i in infos)
    maxx = max(i["bounds"].right for i in infos)
    miny = min(i["bounds"].bottom for i in infos)
    width = int(math.ceil((maxx - minx) / resx))
    height = int(math.ceil((maxy - miny) / resy))
    nodata = infos[0]["nodata"]
    vrt_nodata = 0 if nodata is None else nodata  # gaps between tiles must read as no data
    dtype = _gdal_type(infos[0]["dtype"])

    lines = [f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
             f"  <SRS>{escape(crs.to_wkt())}</SRS>",
             f"  <GeoTransform>{minx!r}, {resx!r}, 0, {maxy!r}, 0, {-resy!r}</GeoTransform>"]
    for b in range(1, count + 1):
        lines.append(f'  <VRTRasterBand dataType="{dtype}" band="{b}">')
        lines.append(f"    <NoDataValue>{vrt_nodata}</NoDataValue>")
        for i in infos:
            xo = (i["bounds"].left - minx) / resx
            yo = (maxy - i["bounds"].top) / resy
            xs = (i["bounds"].right - i["bounds"].left) / resx
            ys = (i["bounds"].top - i["bounds"].bottom) / resy
            nd = f"<NODATA>{i['nodata']}</NODATA>" if i["nodata"] is not None else ""
            lines.append(
                "    <ComplexSource>"
                f'<SourceFilename relativeToVRT="0">{escape(str(i["path"]))}</SourceFilename>'
                f"<SourceBand>{b}</SourceBand>"
                f'<SrcRect xOff="0" yOff="0" xSize="{i["w"]}" ySize="{i["h"]}"/>'
                f'<DstRect xOff="{xo!r}" yOff="{yo!r}" xSize="{xs!r}" ySize="{ys!r}"/>'
                f"{nd}</ComplexSource>")
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, out_path)
    log.info("Built mosaic of %d tiles: %s", len(infos), out_path)
    return out_path


def resolve_imagery(spec: str, work_dir: Path) -> str:
    """Return a single readable raster path for ``inputs.imagery``."""
    if Path(spec).suffix.lower() == ".sid":
        raise ValueError(
            f"{spec} is MrSID, which this GDAL build cannot read. Use the GeoTIFF copy, "
            "or convert it with ArcGIS Pro 'Copy Raster' / gdal_translate."
        )
    tiles = find_tiles(spec)
    if len(tiles) == 1 and not Path(spec).is_dir():
        return str(tiles[0])
    return str(build_vrt(tiles, work_dir / "imagery_mosaic.vrt"))


def _gdal_type(dtype: str) -> str:
    return {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16", "uint32": "UInt32",
            "int32": "Int32", "float32": "Float32", "float64": "Float64"}[dtype]
