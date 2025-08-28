import os, argparse, xarray as xr, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def describe(path):
    ds = xr.open_dataset(path)
    name = os.path.basename(path)
    print(f"\n=== {name} ===")
    print(ds)
    t = pd.to_datetime(ds.time.values)
    print(f"time: {t.min()}  →  {t.max()}  (n={len(t)})")
    lat_name = [c for c in ds.coords if 'lat' in c][0]
    lon_name = [c for c in ds.coords if 'lon' in c][0]
    var = list(ds.data_vars)[0]
    # national daily mean
    daily_mean = ds[var].mean(dim=(lat_name, lon_name)).to_pandas()
    print(f"{var} daily mean (first 5):\n{daily_mean.head()}\n")

    # plot time series (first 2 years to keep it quick)
    head = daily_mean.iloc[:730]
    plt.figure()
    head.plot()
    plt.title(f"{var} national daily mean (first 2 years)")
    plt.xlabel("date"); plt.ylabel(ds[var].attrs.get("units",""))
    plt.tight_layout()
    plt.show()

    # quick spatial map for first day
    first = ds[var].isel(time=0)
    plt.figure()
    first.plot()  # pcolormesh
    plt.title(f"{var} map – first day")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Preprocessed folder")
    args = ap.parse_args()

    describe(os.path.join(args.inp, "tmp2m_daily_c.nc"))
    describe(os.path.join(args.inp, "prcp_daily_mm.nc"))
    describe(os.path.join(args.inp, "wind10m_speed_daily.nc"))
