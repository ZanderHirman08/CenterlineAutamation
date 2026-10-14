"""A small stand-in for arcpy, built on geopandas / rasterio, for testing the .pyt.

It implements only the calls the toolbox makes, following the documented arcpy
behaviour (e.g. RasterToNumPyArray reads starting at the cell containing
``lower_left_corner`` and returns (bands, rows, cols) for multiband rasters).
It cannot prove the toolbox works in real ArcGIS Pro, but it does run every
line of the toolbox code.
"""
from __future__ import annotations

import math
import os
import types

import geopandas as gpd
import rasterio
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import Point as SPoint

messages: list[str] = []
warnings: list[str] = []


class _Env:
    overwriteOutput = True


env = _Env()


def AddMessage(m):
    messages.append(str(m))


def AddWarning(m):
    warnings.append(str(m))
    messages.append("WARNING: " + str(m))


def AddError(m):
    messages.append("ERROR: " + str(m))


def SetProgressor(*a, **k): pass
def SetProgressorLabel(*a, **k): pass
def SetProgressorPosition(*a, **k): pass
def ResetProgressor(*a, **k): pass


class SpatialReference:
    def __init__(self, crs):
        self._crs = rasterio.crs.CRS.from_user_input(crs)
        self.name = self._crs.to_string()
        self.type = "Geographic" if self._crs.is_geographic else "Projected"
        self.linearUnitName = self._crs.linear_units or "Unknown"


class Extent:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.XMin, self.YMin, self.XMax, self.YMax = xmin, ymin, xmax, ymax


class Point:
    def __init__(self, X=0.0, Y=0.0):
        self.X, self.Y = X, Y


class Array(list):
    pass


class _Geometry:
    def __init__(self, shp, sr=None):
        self._shp, self.spatialReference = shp, sr

    @property
    def extent(self):
        return Extent(*self._shp.bounds)

    def disjoint(self, other):
        return self._shp.disjoint(other._shp)

    @property
    def area(self):
        return self._shp.area

    def intersect(self, other, dimension):
        res = self._shp.intersection(other._shp)
        if dimension == 4:
            return Polygon(res, self.spatialReference)
        return _wrap(res, self.spatialReference)

    def union(self, other):
        return type(self)(self._shp.union(other._shp), self.spatialReference)

    def buffer(self, distance):
        return Polygon(self._shp.buffer(distance), self.spatialReference)

    def projectAs(self, sr):
        g = gpd.GeoSeries([self._shp], crs=self.spatialReference._crs).to_crs(sr._crs).iloc[0]
        return type(self)(g, sr)


class Polyline(_Geometry):
    def __init__(self, arr, sr=None):
        if not isinstance(arr, (LineString, MultiLineString)):
            arr = LineString([(p.X, p.Y) for p in arr])
        super().__init__(arr, sr)

    def __iter__(self):
        geoms = getattr(self._shp, "geoms", [self._shp])
        for g in geoms:
            if not g.is_empty:
                yield Array(Point(x, y) for x, y, *_ in g.coords)

    def intersect(self, other, dimension):
        assert dimension == 2
        res = self._shp.intersection(other._shp)
        lines = [g for g in getattr(res, "geoms", [res])
                 if g.geom_type == "LineString" and not g.is_empty]
        return Polyline(MultiLineString(lines) if lines else LineString(), self.spatialReference)


class Polygon(_Geometry):
    def __init__(self, arr, sr=None):
        from shapely.geometry.base import BaseGeometry
        if not isinstance(arr, BaseGeometry):
            from shapely.geometry import Polygon as SPolygon
            arr = SPolygon([(p.X, p.Y) for p in arr])
        super().__init__(arr, sr)


class PointGeometry(_Geometry):
    def __init__(self, pt, sr=None):
        super().__init__(SPoint(pt.X, pt.Y), sr)


def _wrap(geom, sr):
    if geom.geom_type in ("LineString", "MultiLineString"):
        return Polyline(geom, sr)
    return Polygon(geom, sr)


# ----------------------------------------------------------------------- rasters

class Raster:
    def __init__(self, path):
        with rasterio.open(path) as src:
            b = src.bounds
            self.extent = Extent(b.left, b.bottom, b.right, b.top)
            self.meanCellWidth = src.res[0]
            self.meanCellHeight = src.res[1]
            self.bandCount = src.count
            self.spatialReference = SpatialReference(src.crs)


def RasterToNumPyArray(in_raster, lower_left_corner=None, ncols=None, nrows=None,
                       nodata_to_value=None):
    with rasterio.open(in_raster) as src:
        res = src.res[0]
        b = src.bounds
        col0 = math.floor((lower_left_corner.X - b.left) / res)
        row_bottom = math.floor((b.top - lower_left_corner.Y) / res)
        row0 = row_bottom - nrows + 1
        if col0 < 0 or row0 < 0 or col0 + ncols > src.width or row_bottom >= src.height:
            raise RuntimeError("RasterToNumPyArray: block outside raster")
        win = rasterio.windows.Window(col0, row0, ncols, nrows)
        arr = src.read(window=win)
    return arr[0] if arr.shape[0] == 1 else arr


# ----------------------------------------------------------------------- vectors

_schemas: dict[str, dict] = {}


def Exists(path):
    return os.path.exists(path) or path in _schemas


def ListFields(fc):
    df = gpd.read_file(fc)
    out = []
    for name, dt in df.dtypes.items():
        if name == "geometry":
            continue
        from pandas.api import types as pt
        t = "Integer" if pt.is_integer_dtype(dt) else (
            "Double" if pt.is_float_dtype(dt) else "String")
        out.append(types.SimpleNamespace(name=name, type=t))
    return out


class _Management:
    @staticmethod
    def CreateFeatureclass(out_path, out_name, geometry_type, template=None, has_m=None,
                           has_z=None, spatial_reference=None):
        path = os.path.join(out_path, out_name)
        _schemas[path] = dict(fields=[], sr=spatial_reference)

    @staticmethod
    def AddField(in_table, field_name, field_type, field_precision=None, field_scale=None,
                 field_length=None):
        _schemas[in_table]["fields"].append((field_name, field_type))

    @staticmethod
    def Delete(path):
        _schemas.pop(path, None)
        base = os.path.splitext(path)[0]
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            if os.path.exists(base + ext):
                os.remove(base + ext)


management = _Management()


class _SearchCursor:
    def __init__(self, fc, fields, where_clause=None, spatial_reference=None):
        df = gpd.read_file(fc)
        if spatial_reference is not None:
            df = df.to_crs(spatial_reference._crs)
        else:  # like arcpy: geometries carry the dataset's spatial reference
            spatial_reference = SpatialReference(df.crs)
        self.rows = []
        for k, (_, r) in enumerate(df.iterrows(), 1):
            self.rows.append(tuple(_wrap(r.geometry, spatial_reference) if f == "SHAPE@"
                                   else k if f == "OID@" else r[f] for f in fields))

    def __enter__(self):
        return iter(self.rows)

    def __exit__(self, *a):
        return False


class _InsertCursor:
    def __init__(self, fc, fields):
        self.fc, self.fields, self.rows = fc, fields, []
        known = {f for f, _ in _schemas[fc]["fields"]} | {"SHAPE@"}
        missing = set(fields) - known
        if missing:
            raise RuntimeError(f"InsertCursor: unknown fields {missing}")

    def insertRow(self, row):
        assert len(row) == len(self.fields)
        self.rows.append(row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *a):
        if exc_type is None:
            gi = self.fields.index("SHAPE@")
            attrs = {f: [r[i] for r in self.rows] for i, f in enumerate(self.fields) if i != gi}
            sr = _schemas[self.fc]["sr"]
            gpd.GeoDataFrame(attrs, geometry=[r[gi]._shp for r in self.rows],
                             crs=sr._crs).to_file(self.fc)
        return False


da = types.SimpleNamespace(SearchCursor=_SearchCursor, InsertCursor=_InsertCursor)


# ----------------------------------------------------------------------- toolbox params

class _Filter:
    def __init__(self):
        self.list = []
        self.type = None


class Parameter:
    def __init__(self, name=None, displayName=None, datatype=None, parameterType=None,
                 direction=None, category=None, multiValue=False):
        self.name, self.displayName, self.datatype = name, displayName, datatype
        self.parameterType, self.direction, self.category = parameterType, direction, category
        self.value = None
        self.filter = _Filter()
        self.parameterDependencies = []

    @property
    def valueAsText(self):
        return None if self.value is None else str(self.value)


# ----------------------------------------------------------------------- numpy rasters

class _NumpyRaster:
    def __init__(self, arr, ll, xres, yres):
        self.arr, self.ll, self.xres, self.yres = arr, ll, xres, yres

    def save(self, path):
        arr = self.arr[None] if self.arr.ndim == 2 else self.arr
        b, h, w = arr.shape
        from rasterio.transform import from_origin
        tr = from_origin(self.ll.X, self.ll.Y + h * self.yres, self.xres, self.yres)
        with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=b,
                           dtype=arr.dtype, transform=tr) as dst:
            dst.write(arr)


def NumPyArrayToRaster(in_array, lower_left_corner=None, x_cell_size=None, y_cell_size=None,
                       value_to_nodata=None):
    return _NumpyRaster(in_array, lower_left_corner, x_cell_size, y_cell_size)


def _define_projection(path, sr):
    with rasterio.open(path, "r+") as dst:
        dst.crs = sr._crs


_Management.DefineProjection = staticmethod(_define_projection)
