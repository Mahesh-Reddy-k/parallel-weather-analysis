# Import required libraries
import os                         # For file and directory operations
import argparse                   # For parsing command-line arguments
from mpi4py import MPI            # For MPI parallel processing
import numpy as np                # For numerical operations
import xarray as xr               # For handling netCDF datasets
import pandas as pd               # For data manipulation (not used in this code, but imported)

# Initialize MPI communication objects
comm = MPI.COMM_WORLD             # Create MPI communicator
rank = comm.Get_rank()            # Get the rank (ID) of the current process
size = comm.Get_size()            # Get the total number of processes

def log(msg):
    """Print a log message prefixed with the rank and total size of MPI processes."""
    print(f"[rank {rank}/{size}] {msg}", flush=True)  # Ensure immediate output with flush

def split_indices(n, parts, pid):
    """
    Calculate start and end indices for splitting a range [0..n) into 'parts' chunks for process 'pid'.
    Args:
        n (int): Total number of elements to split.
        parts (int): Number of parts to split into (usually number of processes).
        pid (int): Process ID (rank) for which to compute indices.
    Returns:
        tuple: (start, end) indices for the given process.
    """
    base = n // parts             # Base size of each chunk
    extra = n % parts             # Extra elements to distribute
    start = pid * base + min(pid, extra)  # Start index for this process
    end = start + base + (1 if pid < extra else 0)  # End index for this process
    return start, end

def ensure_dir(p):
    """Create directory at path 'p' if it doesn't exist."""
    os.makedirs(p, exist_ok=True)  # Create directories, ignore if already exists

def open_var(ds_path, var_guess=None):
    """
    Open a netCDF dataset and select a data variable.
    Args:
        ds_path (str): Path to the netCDF file.
        var_guess (str, optional): Name of the variable to select, if known.
    Returns:
        tuple: (dataset, data_array) containing the opened dataset and selected variable.
    Raises:
        RuntimeError: If no data variables are found in the dataset.
    """
    ds = xr.open_dataset(ds_path, engine="netcdf4")  # Open netCDF file with xarray
    if var_guess and var_guess in ds:                # If var_guess is provided and exists
        return ds, ds[var_guess]
    # Pick the first data variable if var_guess is not provided or invalid
    for v in ds.data_vars:
        return ds, ds[v]
    raise RuntimeError(f"No data vars found in {ds_path}")

def subset_bbox(da, lat_name='lat', lon_name='lon', bbox=None):
    """
    Subset a DataArray by a bounding box (latitude and longitude).
    Args:
        da (xarray.DataArray): Input DataArray to subset.
        lat_name (str): Name of the latitude coordinate.
        lon_name (str): Name of the longitude coordinate.
        bbox (tuple, optional): (lat_min, lat_max, lon_min, lon_max) in degrees.
    Returns:
        xarray.DataArray: Subsetted DataArray.
    """
    if bbox is None:  # Return original DataArray if no bounding box is provided
        return da
    lat_min, lat_max, lon_min, lon_max = bbox  # Unpack bounding box coordinates
    # Normalize longitude to [0,360) if dataset uses that convention
    lon = da[lon_name]
    if lon.max() > 180:  # Check if longitudes are in [0,360)
        lon_min = lon_min % 360  # Normalize requested longitude bounds
        lon_max = lon_max % 360
    # Subset by latitude
    da_sub = da.sel({lat_name: slice(lat_min, lat_max)})
    # Handle longitude subsetting, including wrap-around cases
    if lon_min <= lon_max:
        da_sub = da_sub.sel({lon_name: slice(lon_min, lon_max)})
    else:  # Handle wrap-around (e.g., crossing 360°/0° boundary)
        da_sub = xr.concat([
            da_sub.sel({lon_name: slice(lon_min, 360)}),
            da_sub.sel({lon_name: slice(0, lon_max)})
        ], dim=lon_name)
    return da_sub

def resample_daily(da, how="mean"):
    """
    Resample a DataArray to daily frequency.
    Args:
        da (xarray.DataArray): Input DataArray with a time dimension.
        how (str): Resampling method ('mean' or 'sum').
    Returns:
        xarray.DataArray: Resampled DataArray with daily frequency.
    """
    if how == "sum":
        return da.resample(time="1D").sum()  # Compute daily sum
    return da.resample(time="1D").mean()     # Compute daily mean

def write_part(da, out_dir, base, pid):
    """
    Write a DataArray to a netCDF file for a specific process.
    Args:
        da (xarray.DataArray): DataArray to write.
        out_dir (str): Output directory.
        base (str): Base name for the output file.
        pid (int): Process ID (rank) for naming the file.
    Returns:
        str: Path to the written netCDF file.
    """
    ensure_dir(out_dir)  # Ensure output directory exists
    part_path = os.path.join(out_dir, f"{base}.part{pid:04d}.nc")  # Create file path
    encoding = {da.name: {"zlib": True, "complevel": 4}}  # Set compression encoding
    da.to_netcdf(part_path, encoding=encoding)  # Write DataArray to netCDF
    return part_path

def concat_parts(out_dir, base, final_name):
    """
    Concatenate partial netCDF files from all processes into a single file.
    Args:
        out_dir (str): Directory containing partial files.
        base (str): Base name of partial files.
        final_name (str): Name of the final concatenated file.
    Returns:
        str: Path to the final concatenated netCDF file.
    Raises:
        RuntimeError: If no partial files are found.
    """
    # Find all partial files matching the base name
    parts = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir)
                    if f.startswith(base + ".part") and f.endswith(".nc")])
    if not parts:
        raise RuntimeError(f"No parts found for {base} in {out_dir}")
    # Open and concatenate DataArrays along the time dimension
    das = [xr.open_dataarray(p, engine="netcdf4") for p in parts]
    combined = xr.concat(das, dim="time")
    # Write concatenated DataArray to final netCDF file
    final_path = os.path.join(out_dir, final_name)
    encoding = {combined.name: {"zlib": True, "complevel": 4}}  # Set compression
    combined.to_netcdf(final_path, encoding=encoding)
    # Clean up partial files
    for p in parts:
        try:
            os.remove(p)  # Attempt to delete each partial file
        except Exception:
            pass  # Ignore errors during deletion
    return final_path

def main(args):
    """Main function to process meteorological data in parallel using MPI."""
    root = args.root  # Root directory for input files
    out_dir = args.out  # Output directory for processed files
    bbox = tuple(args.bbox) if args.bbox else None  # Bounding box for subsetting, if provided

    # Define file paths for input datasets
    f_tmp2m = os.path.join(root, "Surface_variables", "IMDAA_TMP_2m_1.08_1990_2020.nc")  # 2m temperature
    f_apcp  = os.path.join(root, "Surface_variables", "IMDAA_APCP_sfc_1.08_1990_2020.nc")  # Precipitation
    f_u10   = os.path.join(root, "Surface_variables", "IMDAA_UGRD_10m_1.08_1990_2020.nc")  # U-wind at 10m
    f_v10   = os.path.join(root, "Surface_variables", "IMDAA_VGRD_10m_1.08_1990_2020.nc")  # V-wind at 10m

    # ---------- Process 2m Temperature (convert K to °C, compute daily mean) ----------
    ds_t, da_t = open_var(f_tmp2m, var_guess="TMP_2m")  # Open temperature dataset
    time_len = da_t.sizes.get("time", None)  # Get size of time dimension
    if time_len is None:
        raise RuntimeError("No 'time' dimension found in TMP_2m file.")
    
    # Split time dimension across processes
    s, e = split_indices(time_len, size, rank)
    da_t_slab = da_t.isel(time=slice(s, e))  # Select time slice for this process
    # Subset by bounding box, dynamically finding lat/lon coordinate names
    da_t_slab = subset_bbox(da_t_slab, 
                           lat_name=[k for k in da_t_slab.coords if 'lat' in k][0],
                           lon_name=[k for k in da_t_slab.coords if 'lon' in k][0],
                           bbox=bbox)

    # Convert from Kelvin to Celsius if necessary
    units = da_t_slab.attrs.get("units", "").lower()
    if "k" in units or da_t_slab.max() > 200:  # Assume Kelvin if max value > 200
        da_t_slab = da_t_slab - 273.15
        da_t_slab.attrs["units"] = "degC"

    # Resample to daily mean and write output
    da_t_day = resample_daily(da_t_slab, how="mean")
    da_t_day.name = "t2m_c"  # Name the variable
    p1 = write_part(da_t_day, out_dir, "tmp2m_daily_c", rank)  # Write to netCDF
    log(f"Wrote {p1}")  # Log the output file

    # ---------- Process Precipitation (compute daily sum) ----------
    ds_p, da_p = open_var(f_apcp, var_guess="APCP_sfc")  # Open precipitation dataset
    time_len_p = da_p.sizes.get("time", None)  # Get size of time dimension
    s2, e2 = split_indices(time_len_p, size, rank)  # Split time dimension
    da_p_slab = da_p.isel(time=slice(s2, e2))  # Select time slice
    # Subset by bounding box
    da_p_slab = subset_bbox(da_p_slab,
                           lat_name=[k for k in da_p_slab.coords if 'lat' in k][0],
                           lon_name=[k for k in da_p_slab.coords if 'lon' in k][0],
                           bbox=bbox)

    # Resample to daily sum and standardize units to mm/day
    da_p_day = resample_daily(da_p_slab, how="sum")
    p_units = da_p_day.attrs.get("units", "")
    if not p_units:
        da_p_day.attrs["units"] = "mm/day"  # Assume mm/day if units are missing
    da_p_day.name = "prcp_mm_day"  # Name the variable
    p2 = write_part(da_p_day, out_dir, "prcp_daily_mm", rank)  # Write to netCDF
    log(f"Wrote {p2}")  # Log the output file

    # ---------- Process 10m Wind Speed (compute from U and V components, daily mean) ----------
    ds_u, da_u = open_var(f_u10, var_guess="UGRD_10m")  # Open U-wind dataset
    ds_v, da_v = open_var(f_v10, var_guess="VGRD_10m")  # Open V-wind dataset

    # Ensure consistent time slices for U and V components
    n_u = da_u.sizes.get("time", None)
    n_v = da_v.sizes.get("time", None)
    n_w = min(n_u, n_v)  # Use the minimum time length
    s3, e3 = split_indices(n_w, size, rank)  # Split time dimension
    # Subset U and V components by bounding box
    da_u_slab = subset_bbox(da_u.isel(time=slice(s3, e3)),
                           lat_name=[k for k in da_u.coords if 'lat' in k][0],
                           lon_name=[k for k in da_u.coords if 'lon' in k][0],
                           bbox=bbox)
    da_v_slab = subset_bbox(da_v.isel(time=slice(s3, e3)),
                           lat_name=[k for k in da_v.coords if 'lat' in k][0],
                           lon_name=[k for k in da_v.coords if 'lon' in k][0],
                           bbox=bbox)

    # Align U and V components to ensure matching coordinates
    da_u_slab, da_v_slab = xr.align(da_u_slab, da_v_slab, join="inner")
    # Compute wind speed as sqrt(u^2 + v^2)
    wind_speed = xr.apply_ufunc(
        lambda uu, vv: np.sqrt(uu**2 + vv**2),
        da_u_slab, da_v_slab,
        dask="parallelized", output_dtypes=[float]
    )
    wind_speed.attrs["units"] = "m s-1"  # Set units
    wind_speed.name = "wind10m_speed"  # Name the variable
    wind_speed_day = resample_daily(wind_speed, how="mean")  # Resample to daily mean
    p3 = write_part(wind_speed_day, out_dir, "wind10m_speed_daily", rank)  # Write to netCDF
    log(f"Wrote {p3}")  # Log the output file

    # ---------- Concatenate partial files on rank 0 ----------
    comm.Barrier()  # Synchronize all processes
    if rank == 0:   # Only rank 0 performs concatenation
        log("Concatenating parts…")  # Log start of concatenation
        # Concatenate partial files for each variable
        fA = concat_parts(out_dir, "tmp2m_daily_c", "tmp2m_daily_c.nc")
        fB = concat_parts(out_dir, "prcp_daily_mm", "prcp_daily_mm.nc")
        fC = concat_parts(out_dir, "wind10m_speed_daily", "wind10m_speed_daily.nc")
        log(f"Final files:\n  {fA}\n  {fB}\n  {fC}")  # Log final output files

if __name__ == "__main__":
    """Parse command-line arguments and run the main function."""
    parser = argparse.ArgumentParser()  # Initialize argument parser
    parser.add_argument("--root", required=True, help="Path to BharatBench root folder")  # Root directory
    parser.add_argument("--out", required=True, help="Output folder")  # Output directory
    parser.add_argument("--bbox", nargs=4, type=float,
                        help="lat_min lat_max lon_min lon_max (e.g., 5 38 68 98 for India)")  # Bounding box
    args = parser.parse_args()  # Parse arguments
    main(args)  # Execute main function