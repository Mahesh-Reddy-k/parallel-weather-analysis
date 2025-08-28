import os
import argparse
from mpi4py import MPI
import numpy as np
import xarray as xr
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log(msg):
    print(f"[rank {rank}/{size}] {msg}", flush=True)

def split_indices(n, parts, pid):
    """Return (start, end) indices for pid when [0..n) is split into 'parts' chunks."""
    base = n // parts
    extra = n % parts
    start = pid * base + min(pid, extra)
    end   = start + base + (1 if pid < extra else 0)
    return start, end

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def open_var(ds_path, var_guess=None):
    ds = xr.open_dataset(ds_path, engine="netcdf4")
    if var_guess and var_guess in ds:
        return ds, ds[var_guess]
    # pick first data var if not provided
    for v in ds.data_vars:
        return ds, ds[v]
    raise RuntimeError(f"No data vars found in {ds_path}")

def subset_bbox(da, lat_name='lat', lon_name='lon', bbox=None):
    """bbox = (lat_min, lat_max, lon_min, lon_max) in degrees."""
    if bbox is None:
        return da
    lat_min, lat_max, lon_min, lon_max = bbox
    # Normalize lon to [0,360) if dataset uses that
    lon = da[lon_name]
    if lon.max() > 180:
        # convert requested bbox to 0..360 if needed
        lon_min = lon_min % 360
        lon_max = lon_max % 360
    # Handle wrap-around intelligently
    da_sub = da.sel({lat_name: slice(lat_min, lat_max)})
    if lon_min <= lon_max:
        da_sub = da_sub.sel({lon_name: slice(lon_min, lon_max)})
    else:
        # wrap case (e.g., 350..10)
        da_sub = xr.concat([
            da_sub.sel({lon_name: slice(lon_min, 360)}),
            da_sub.sel({lon_name: slice(0, lon_max)})
        ], dim=lon_name)
    return da_sub

def resample_daily(da, how="mean"):
    if how == "sum":
        return da.resample(time="1D").sum()
    return da.resample(time="1D").mean()

def write_part(da, out_dir, base, pid):
    ensure_dir(out_dir)
    part_path = os.path.join(out_dir, f"{base}.part{pid:04d}.nc")
    encoding = {da.name: {"zlib": True, "complevel": 4}}
    da.to_netcdf(part_path, encoding=encoding)
    return part_path

def concat_parts(out_dir, base, final_name):
    parts = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir)
                    if f.startswith(base + ".part") and f.endswith(".nc")])
    if not parts:
        raise RuntimeError(f"No parts found for {base} in {out_dir}")
    das = [xr.open_dataarray(p, engine="netcdf4") for p in parts]
    combined = xr.concat(das, dim="time")
    final_path = os.path.join(out_dir, final_name)
    encoding = {combined.name: {"zlib": True, "complevel": 4}}
    combined.to_netcdf(final_path, encoding=encoding)
    for p in parts:
        try:
            os.remove(p)
        except Exception:
            pass
    return final_path

def main(args):
    root = args.root
    out_dir = args.out
    bbox = tuple(args.bbox) if args.bbox else None

    # ---- File paths (adjust if your names differ) ----
    f_tmp2m = os.path.join(root, "Surface_variables", "IMDAA_TMP_2m_1.08_1990_2020.nc")
    f_apcp  = os.path.join(root, "Surface_variables", "IMDAA_APCP_sfc_1.08_1990_2020.nc")
    f_u10   = os.path.join(root, "Surface_variables", "IMDAA_UGRD_10m_1.08_1990_2020.nc")
    f_v10   = os.path.join(root, "Surface_variables", "IMDAA_VGRD_10m_1.08_1990_2020.nc")

    # ---------- TEMP 2m (K -> °C), daily mean ----------
    ds_t, da_t = open_var(f_tmp2m, var_guess="TMP_2m")
    time_len = da_t.sizes.get("time", None)
    if time_len is None:
        raise RuntimeError("No 'time' dimension found in TMP_2m file.")

    s, e = split_indices(time_len, size, rank)
    da_t_slab = da_t.isel(time=slice(s, e))
    da_t_slab = subset_bbox(da_t_slab, lat_name=[k for k in da_t_slab.coords if 'lat' in k][0],
                            lon_name=[k for k in da_t_slab.coords if 'lon' in k][0],
                            bbox=bbox)

    # Kelvin -> Celsius if units indicate Kelvin or values look Kelvin
    units = da_t_slab.attrs.get("units", "").lower()
    if "k" in units or da_t_slab.max() > 200:
        da_t_slab = da_t_slab - 273.15
        da_t_slab.attrs["units"] = "degC"

    da_t_day = resample_daily(da_t_slab, how="mean")
    da_t_day.name = "t2m_c"
    p1 = write_part(da_t_day, out_dir, "tmp2m_daily_c", rank)
    log(f"Wrote {p1}")

    # ---------- APCP precip, daily sum ----------
    ds_p, da_p = open_var(f_apcp, var_guess="APCP_sfc")
    time_len_p = da_p.sizes.get("time", None)
    s2, e2 = split_indices(time_len_p, size, rank)
    da_p_slab = da_p.isel(time=slice(s2, e2))
    da_p_slab = subset_bbox(da_p_slab,
                            lat_name=[k for k in da_p_slab.coords if 'lat' in k][0],
                            lon_name=[k for k in da_p_slab.coords if 'lon' in k][0],
                            bbox=bbox)

    da_p_day = resample_daily(da_p_slab, how="sum")
    # Try to standardize to mm/day if possible (many IMDAA APCP are in kg m-2 ~= mm)
    p_units = da_p_day.attrs.get("units", "")
    if not p_units:
        da_p_day.attrs["units"] = "mm/day"
    da_p_day.name = "prcp_mm_day"
    p2 = write_part(da_p_day, out_dir, "prcp_daily_mm", rank)
    log(f"Wrote {p2}")

    # ---------- 10m wind speed from U/V, daily mean ----------
    ds_u, da_u = open_var(f_u10, var_guess="UGRD_10m")
    ds_v, da_v = open_var(f_v10, var_guess="VGRD_10m")

    # match time slices across U and V
    n_u = da_u.sizes.get("time", None)
    n_v = da_v.sizes.get("time", None)
    n_w = min(n_u, n_v)
    s3, e3 = split_indices(n_w, size, rank)
    da_u_slab = subset_bbox(da_u.isel(time=slice(s3, e3)),
                            lat_name=[k for k in da_u.coords if 'lat' in k][0],
                            lon_name=[k for k in da_u.coords if 'lon' in k][0],
                            bbox=bbox)
    da_v_slab = subset_bbox(da_v.isel(time=slice(s3, e3)),
                            lat_name=[k for k in da_v.coords if 'lat' in k][0],
                            lon_name=[k for k in da_v.coords if 'lon' in k][0],
                            bbox=bbox)

    # Align (in case coords differ slightly)
    da_u_slab, da_v_slab = xr.align(da_u_slab, da_v_slab, join="inner")
    wind_speed = xr.apply_ufunc(
        lambda uu, vv: np.sqrt(uu**2 + vv**2),
        da_u_slab, da_v_slab,
        dask="parallelized", output_dtypes=[float]
    )
    wind_speed.attrs["units"] = "m s-1"
    wind_speed.name = "wind10m_speed"
    wind_speed_day = resample_daily(wind_speed, how="mean")
    p3 = write_part(wind_speed_day, out_dir, "wind10m_speed_daily", rank)
    log(f"Wrote {p3}")

    # ------------ Merge per-rank shards on rank 0 ------------
    comm.Barrier()
    if rank == 0:
        log("Concatenating parts…")
        fA = concat_parts(out_dir, "tmp2m_daily_c", "tmp2m_daily_c.nc")
        fB = concat_parts(out_dir, "prcp_daily_mm", "prcp_daily_mm.nc")
        fC = concat_parts(out_dir, "wind10m_speed_daily", "wind10m_speed_daily.nc")
        log(f"Final files:\n  {fA}\n  {fB}\n  {fC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to BharatBench root folder")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--bbox", nargs=4, type=float,
                        help="lat_min lat_max lon_min lon_max (e.g., 5 38 68 98 for India)")
    args = parser.parse_args()
    main(args)
