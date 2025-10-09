# Import necessary libraries for file handling, command-line arguments, parallel processing, and data manipulation
import os
import argparse
from mpi4py import MPI
import numpy as np
import xarray as xr
import pandas as pd

# Initialize MPI communicator, rank, and size for parallel processing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get the process ID (rank) of the current process
size = comm.Get_size()  # Get the total number of processes

# Function to log messages with rank and size information for debugging
def log(msg):
    print(f"[rank {rank}/{size}] {msg}", flush=True)  # Print message with rank/size prefix and flush output

# Function to split a range of indices [0..n) into 'parts' chunks for parallel processing
def split_indices(n, parts, pid):
    """Return (start, end) indices for pid when [0..n) is split into 'parts' chunks."""
    base = n // parts  # Base size of each chunk
    extra = n % parts  # Remainder to distribute extra elements
    start = pid * base + min(pid, extra)  # Calculate start index for this process
    end = start + base + (1 if pid < extra else 0)  # Calculate end index
    return start, end

# Function to create a directory if it doesn't exist
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)  # Create directory at path p, ignore if it already exists

# Function to open a NetCDF dataset and select a variable
def open_var(ds_path, var_guess=None):
    ds = xr.open_dataset(ds_path, engine="netcdf4")  # Open dataset using NetCDF4 engine
    if var_guess and var_guess in ds:  # If a variable is specified and exists, return it
        return ds, ds[var_guess]
    # If no variable specified or not found, select the first data variable
    for v in ds.data_vars:
        return ds, ds[v]
    raise RuntimeError(f"No data vars found in {ds_path}")  # Raise error if no variables found

# Function to subset a dataset by a bounding box (latitude/longitude)
def subset_bbox(da, lat_name='lat', lon_name='lon', bbox=None):
    """bbox = (lat_min, lat_max, lon_min, lon_max) in degrees."""
    if bbox is None:
        return da  # Return original data if no bounding box is provided
    lat_min, lat_max, lon_min, lon_max = bbox  # Unpack bounding box coordinates
    # Normalize longitude to [0,360) if dataset uses that range
    lon = da[lon_name]
    if lon.max() > 180:
        lon_min = lon_min % 360  # Convert lon_min to [0,360)
        lon_max = lon_max % 360  # Convert lon_max to [0,360)
    # Handle latitude subsetting
    da_sub = da.sel({lat_name: slice(lat_min, lat_max)})
    # Handle longitude subsetting, including wrap-around cases
    if lon_min <= lon_max:
        da_sub = da_sub.sel({lon_name: slice(lon_min, lon_max)})  # Normal case
    else:
        # Handle wrap-around (e.g., crossing 360°/0°)
        da_sub = xr.concat([
            da_sub.sel({lon_name: slice(lon_min, 360)}),  # Left side of wrap
            da_sub.sel({lon_name: slice(0, lon_max)})     # Right side of wrap
        ], dim=lon_name)
    return da_sub

# Function to resample data to daily frequency
def resample_daily(da, how="mean"):
    if how == "sum":
        return da.resample(time="1D").sum()  # Resample to daily sum
    return da.resample(time="1D").mean()  # Resample to daily mean (default)

# Function to write a data array to a NetCDF file for a specific process
def write_part(da, out_dir, base, pid):
    ensure_dir(out_dir)  # Ensure output directory exists
    part_path = os.path.join(out_dir, f"{base}.part{pid:04d}.nc")  # Create file path with part number
    encoding = {da.name: {"zlib": True, "complevel": 4}}  # Set compression encoding
    da.to_netcdf(part_path, encoding=encoding)  # Write data array to NetCDF file
    return part_path  # Return the path to the written file

# Function to concatenate partial NetCDF files into a final file (run by rank 0)
def concat_parts(out_dir, base, final_name):
    # Find all part files matching the base name
    parts = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir)
                    if f.startswith(base + ".part") and f.endswith(".nc")])
    if not parts:
        raise RuntimeError(f"No parts found for {base} in {out_dir}")  # Raise error if no parts found
    das = [xr.open_dataarray(p, engine="netcdf4") for p in parts]  # Open all part files
    combined = xr.concat(das, dim="time")  # Concatenate along time dimension
    final_path = os.path.join(out_dir, final_name)  # Define final output file path
    encoding = {combined.name: {"zlib": True, "complevel": 4}}  # Set compression encoding
    combined.to_netcdf(final_path, encoding=encoding)  # Write combined data to final file
    # Clean up part files
    for p in parts:
        try:
            os.remove(p)  # Attempt to delete part files
        except Exception:
            pass  # Ignore any errors during deletion
    return final_path  # Return path to final file

# Main function to process the data
def main(args):
    root = args.root  # Root directory for input files
    out_dir = args.out  # Output directory for processed files
    bbox = tuple(args.bbox) if args.bbox else None  # Bounding box, if provided

    # Define file paths for input datasets
    f_tmp2m = os.path.join(root, "Surface_variables", "IMDAA_TMP_2m_1.08_1990_2020.nc")
    f_apcp  = os.path.join(root, "Surface_variables", "IMDAA_APCP_sfc_1.08_1990_2020.nc")
    f_u10   = os.path.join(root, "Surface_variables", "IMDAA_UGRD_10m_1.08_1990_2020.nc")
    f_v10   = os.path.join(root, "Surface_variables", "IMDAA_VGRD_10m_1.08_1990_2020.nc")

    # ---------- Process 2m Temperature (convert K to °C, daily mean) ----------
    ds_t, da_t = open_var(f_tmp2m, var_guess="TMP_2m")  # Open temperature dataset
    time_len = da_t.sizes.get("time", None)  # Get size of time dimension
    if time_len is None:
        raise RuntimeError("No 'time' dimension found in TMP_2m file.")  # Check for time dimension
    s, e = split_indices(time_len, size, rank)  # Split time indices for this rank
    da_t_slab = da_t.isel(time=slice(s, e))  # Select time slice for this rank
    # Subset by bounding box, dynamically finding lat/lon coordinate names
    da_t_slab = subset_bbox(da_t_slab, 
                           lat_name=[k for k in da_t_slab.coords if 'lat' in k][0],
                           lon_name=[k for k in da_t_slab.coords if 'lon' in k][0],
                           bbox=bbox)
    # Convert from Kelvin to Celsius if necessary
    units = da_t_slab.attrs.get("units", "").lower()
    if "k" in units or da_t_slab.max() > 200:  # Check if data is in Kelvin
        da_t_slab = da_t_slab - 273.15  # Convert to Celsius
        da_t_slab.attrs["units"] = "degC"  # Update units
    da_t_day = resample_daily(da_t_slab, how="mean")  # Resample to daily mean
    da_t_day.name = "t2m_c"  # Name the variable
    p1 = write_part(da_t_day, out_dir, "tmp2m_daily_c", rank)  # Write to file
    log(f"Wrote {p1}")  # Log output file path

    # ---------- Process Precipitation (daily sum, standardize to mm/day) ----------
    ds_p, da_p = open_var(f_apcp, var_guess="APCP_sfc")  # Open precipitation dataset
    time_len_p = da_p.sizes.get("time", None)  # Get size of time dimension
    s2, e2 = split_indices(time_len_p, size, rank)  # Split time indices for this rank
    da_p_slab = da_p.isel(time=slice(s2, e2))  # Select time slice
    # Subset by bounding box
    da_p_slab = subset_bbox(da_p_slab,
                           lat_name=[k for k in da_p_slab.coords if 'lat' in k][0],
                           lon_name=[k for k in da_p_slab.coords if 'lon' in k][0],
                           bbox=bbox)
    da_p_day = resample_daily(da_p_slab, how="sum")  # Resample to daily sum
    # Standardize units to mm/day
    p_units = da_p_day.attrs.get("units", "")
    if not p_units:
        da_p_day.attrs["units"] = "mm/day"  # Assume mm/day if units not specified
    da_p_day.name = "prcp_mm_day"  # Name the variable
    p2 = write_part(da_p_day, out_dir, "prcp_daily_mm", rank)  # Write to file
    log(f"Wrote {p2}")  # Log output file path

    # ---------- Process 10m Wind Speed (from U/V components, daily mean) ----------
    ds_u, da_u = open_var(f_u10, var_guess="UGRD_10m")  # Open U-component wind dataset
    ds_v, da_v = open_var(f_v10, var_guess="VGRD_10m")  # Open V-component wind dataset
    # Ensure consistent time slices for U and V
    n_u = da_u.sizes.get("time", None)
    n_v = da_v.sizes.get("time", None)
    n_w = min(n_u, n_v)  # Use minimum time length to avoid mismatch
    s3, e3 = split_indices(n_w, size, rank)  # Split time indices
    # Subset U and V components by bounding box
    da_u_slab = subset_bbox(da_u.isel(time=slice(s3, e3)),
                           lat_name=[k for k in da_u.coords if 'lat' in k][0],
                           lon_name=[k for k in da_u.coords if 'lon' in k][0],
                           bbox=bbox)
    da_v_slab = subset_bbox(da_v.isel(time=slice(s3, e3)),
                           lat_name=[k for k in da_v.coords if 'lat' in k][0],
                           lon_name=[k for k in da_v.coords if 'lon' in k][0],
                           bbox=bbox)
    # Align U and V to ensure consistent coordinates
    da_u_slab, da_v_slab = xr.align(da_u_slab, da_v_slab, join="inner")
    # Calculate wind speed as sqrt(u^2 + v^2)
    wind_speed = xr.apply_ufunc(
        lambda uu, vv: np.sqrt(uu**2 + vv**2),  # Compute wind speed
        da_u_slab, da_v_slab,
        dask="parallelized", output_dtypes=[float]  # Use Dask for parallel computation
    )
    wind_speed.attrs["units"] = "m s-1"  # Set units
    wind_speed.name = "wind10m_speed"  # Name the variable
    wind_speed_day = resample_daily(wind_speed, how="mean")  # Resample to daily mean
    p3 = write_part(wind_speed_day, out_dir, "wind10m_speed_daily", rank)  # Write to file
    log(f"Wrote {p3}")  # Log output file path

    # ---------- Merge partial files into final files (rank 0 only) ----------
    comm.Barrier()  # Synchronize all processes
    if rank == 0:  # Only rank 0 concatenates the parts
        log("Concatenating parts…")  # Log start of concatenation
        # Concatenate temperature, precipitation, and wind speed files
        fA = concat_parts(out_dir, "tmp2m_daily_c", "tmp2m_daily_c.nc")
        fB = concat_parts(out_dir, "prcp_daily_mm", "prcp_daily_mm.nc")
        fC = concat_parts(out_dir, "wind10m_speed_daily", "wind10m_speed_daily.nc")
        log(f"Final files:\n  {fA}\n  {fB}\n  {fC}")  # Log paths to final files

# Entry point for the script
if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to BharatBench root folder")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--bbox", nargs=4, type=float,
                        help="lat_min lat_max lon_min lon_max (e.g., 5 38 68 98 for India)")
    args = parser.parse_args()  # Parse arguments
    main(args)  # Run main function with parsed arguments