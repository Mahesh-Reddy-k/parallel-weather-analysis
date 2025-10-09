import os
import argparse
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

def plot_and_save(da, title, out_path, cmap='viridis'):
    """Helper function to create and save a plot."""
    if da.isnull().all():
        raise ValueError(f"Data for {title} contains only NaNs, cannot plot.")
    plt.figure(figsize=(10, 8))
    da.plot(cmap=cmap)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Rank {MPI.COMM_WORLD.Get_rank()}: Saved plot: {out_path}")

def compute_climatology_parallel(var, var_name, display_name, cmap, args, comm, rank, size):
    """Compute climatology by distributing years across MPI ranks."""
    
    # Define the full time range
    years = list(range(1990, 2021))  # 31 years
    
    # Divide years among ranks
    years_per_rank = np.array_split(years, size)
    my_years = years_per_rank[rank]
    
    if rank == 0:
        print(f"⏳ Computing climatology for {display_name}...")
        print(f"  Total years: {len(years)}, MPI ranks: {size}")
    
    print(f"  Rank {rank}: Processing years {my_years[0]}-{my_years[-1]}")
    
    # Select data for my years
    time_slices = []
    for year in my_years:
        time_slices.append(var.sel(time=slice(f'{year}-01-01', f'{year}-12-31')))
    
    my_data = xr.concat(time_slices, dim='time')
    
    # Compute monthly mean for my years
    my_monthly_clim = my_data.groupby('time.month').mean(dim='time').compute()
    
    # Gather results from all ranks
    all_monthly_clims = comm.gather(my_monthly_clim, root=0)
    
    # Rank 0 combines and plots
    if rank == 0:
        print(f"  Rank 0: Combining results from all ranks...")
        # Average the climatologies from all ranks
        combined_clim = xr.concat(all_monthly_clims, dim='temp_dim').mean(dim='temp_dim')
        
        for month in range(1, 13):
            print(f"  Plotting month {month} for {display_name}...")
            clim = combined_clim.sel(month=month)
            if clim.isnull().any():
                print(f"  Warning: NaNs detected in {display_name} for month {month}")
            plot_and_save(
                clim,
                title=f"Mean {display_name} Climatology - Month {month} (1990-2020)",
                out_path=os.path.join(args.out, f"climatology_{var_name}_month{month:02d}.png"),
                cmap=cmap
            )
    
    comm.Barrier()

def compute_anomaly_parallel(variables_data, args, comm, rank, size):
    """Compute anomalies by distributing variables across MPI ranks."""
    
    # Define variables to process
    var_configs = [
        ('temp_c', 't2m_c', 'Temperature (°C)', 'coolwarm'),
        ('prcp_mm', 'prcp_mm_day', 'Precipitation (mm/day)', 'RdBu'),
        ('wind_speed', 'wind10m_speed', 'Wind Speed (m/s)', 'RdYlBu')
    ]
    
    # Assign one variable per rank (assuming at least 3 ranks for 3 variables)
    if rank < len(var_configs):
        var_key, var_name, display_name, cmap = var_configs[rank]
        var = variables_data[var_key]
        
        print(f"⏳ Rank {rank}: Computing anomalies for {display_name}...")
        
        # Compute daily climatology (1990-2020)
        climatology_period = slice('1990-01-01', '2020-12-31')
        clim_daily = var.sel(time=climatology_period).groupby('time.dayofyear').mean(dim='time').compute()
        
        # Compute anomalies
        anomalies = var.groupby('time.dayofyear') - clim_daily
        
        if anomalies.isnull().all():
            raise ValueError(f"Anomaly data for {display_name} contains only NaNs")
        
        print(f"  Rank {rank}: Finding extreme anomaly for {display_name}...")
        
        # Find extreme anomaly
        abs_anomalies = xr.ufuncs.absolute(anomalies)
        extreme_idx = abs_anomalies.argmax(dim='time')
        
        if isinstance(extreme_idx, xr.DataArray):
            # Get spatial maximum
            extreme_idx = int(extreme_idx.max())
        
        extreme_anomaly = anomalies.isel(time=extreme_idx).compute()
        extreme_date = str(np.datetime_as_string(extreme_anomaly.time.values, unit='D'))
        
        if extreme_anomaly.isnull().any():
            print(f"  Rank {rank}: Warning: NaNs detected in extreme anomaly for {display_name}")
        
        plot_and_save(
            extreme_anomaly,
            title=f"Extreme {display_name} Anomaly on {extreme_date} (Relative to 1990-2020 Daily Average)",
            out_path=os.path.join(args.out, f"extreme_{var_name}_anomaly.png"),
            cmap=cmap
        )
        
        print(f"✅ Rank {rank}: Completed anomaly analysis for {display_name}")
    else:
        print(f"  Rank {rank}: Idle (more ranks than variables)")
    
    comm.Barrier()

def main(args):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"🚀 Starting MPI parallel analysis with {size} ranks")
        # Create output directory
        os.makedirs(args.out, exist_ok=True)
        if not os.access(args.out, os.W_OK):
            raise PermissionError(f"No write permission for output directory: {args.out}")
    
    comm.Barrier()
    
    # ---- 1. Load Datasets (all ranks) ----
    if rank == 0:
        print("➡️ Loading datasets...")
    
    files = {
        "tmp2m": os.path.join(args.in_dir, "tmp2m_daily_c.nc"),
        "prcp": os.path.join(args.in_dir, "prcp_daily_mm.nc"),
        "wind": os.path.join(args.in_dir, "wind10m_speed_daily.nc")
    }
    
    for f in files.values():
        check_file_exists(f)
    
    ds_temp = xr.open_dataset(files["tmp2m"], chunks={'time': 500})
    ds_prcp = xr.open_dataset(files["prcp"], chunks={'time': 500})
    ds_wind = xr.open_dataset(files["wind"], chunks={'time': 500})
    
    if 't2m_c' not in ds_temp:
        raise KeyError(f"Variable 't2m_c' not found in {files['tmp2m']}")
    if 'prcp_mm_day' not in ds_prcp:
        raise KeyError(f"Variable 'prcp_mm_day' not found in {files['prcp']}")
    if 'wind10m_speed' not in ds_wind:
        raise KeyError(f"Variable 'wind10m_speed' not found in {files['wind']}")
    
    # ---- 2. Standardize Time Range ----
    if rank == 0:
        print("➡️ Standardizing time range to 1990-2020...")
    
    common_time_slice = slice('1990-01-01', '2020-12-31')
    temp_c = ds_temp.t2m_c.sel(time=common_time_slice)
    prcp_mm = ds_prcp.prcp_mm_day.sel(time=common_time_slice)
    wind_speed = ds_wind.wind10m_speed.sel(time=common_time_slice)
    
    # ---- 3. Compute Climatology Maps in Parallel (years distributed) ----
    if rank == 0:
        print("\n" + "="*60)
        print("PHASE 1: CLIMATOLOGY COMPUTATION (Years Distributed)")
        print("="*60)
    
    for var, var_name, display_name, cmap in [
        (temp_c, "t2m_c", "Temperature (°C)", "YlOrRd"),
        (prcp_mm, "prcp_mm_day", "Precipitation (mm/day)", "Blues"),
        (wind_speed, "wind10m_speed", "Wind Speed (m/s)", "Greens")
    ]:
        compute_climatology_parallel(var, var_name, display_name, cmap, args, comm, rank, size)
    
    # ---- 4. Compute Anomalies in Parallel (variables distributed) ----
    if rank == 0:
        print("\n" + "="*60)
        print("PHASE 2: ANOMALY COMPUTATION (Variables Distributed)")
        print("="*60)
    
    variables_data = {
        'temp_c': temp_c,
        'prcp_mm': prcp_mm,
        'wind_speed': wind_speed
    }
    
    compute_anomaly_parallel(variables_data, args, comm, rank, size)
    
    if rank == 0:
        print("\n🎉 Analysis complete! Check the 'plots' folder for outputs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute climatology maps and detect extreme anomalies using MPI parallelization.")
    parser.add_argument("--in_dir", required=True, help="Input directory containing preprocessed .nc files (e.g., 'Preprocessed').")
    parser.add_argument("--out", required=True, help="Output directory to save plots (e.g., 'plots').")
    args = parser.parse_args()
    main(args)
