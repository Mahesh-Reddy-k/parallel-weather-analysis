# Parallel Weather Data Analysis and Visualization System  

## Team A10  
- **Chaitanya Mahesh** – [CB.AI.U4AID23058]  
- **Mahesh Reddy** – [CB.AI.U4AID23059]  
- **Kowshik Reddy** – [CB.AI.U4AID23060]  
- **Mohith** – [CB.AI.U4AID23045]  

---

## Problem Statement  
Climate and weather data are growing rapidly in both size and complexity, coming from satellites, sensors, and climate models.  
Analyzing these datasets using traditional sequential methods is slow and inefficient.  

Real-time or near-real-time insights into patterns such as temperature anomalies, precipitation extremes, or wind speed variations are critical for applications in:  
- Agriculture  
- Disaster management  
- Climate research  

This project builds a **parallelized system** that leverages High-Performance Computing (HPC) to preprocess, analyze, and visualize large-scale weather datasets.  

---

## Dataset  
We used the **BharatBench – IMDAA Reanalysis Dataset (1990–2020)** in **NetCDF format (`.nc`)**.  
Key variables:  
- **TMP_2m** → 2 m temperature (Kelvin → converted to °C)  
- **APCP_sfc** → surface precipitation (kg m⁻² → mm/day)  
- **UGRD_10m & VGRD_10m** → 10 m wind components (converted to wind speed magnitude)  

Spatial coverage was subset to **India** (lat: 5°–38°N, lon: 68°–98°E).  

---

## Step-by-Step Procedure  

### 1. Preprocessing with MPI  
- Implemented using **`mpi4py`**.  
- **Parallelization strategy:**  
  - Data split along **time dimension**.  
  - Each MPI rank loads its own time slice of all variables.  
  - Preprocessing performed per rank:  
    - Unit conversions (K → °C; precipitation to mm/day)  
    - Daily resampling (mean for temp/wind, sum for precip)  
    - Deriving new variables (10 m wind speed = √(U² + V²))  
  - Ranks write partial outputs (`.partXXXX.nc`).  
  - **Rank 0** concatenates parts into final datasets. 


### 2. Verification  
- Loaded preprocessed outputs using **xarray**.  
- Calculated national daily mean for each variable.  
- Verified seasonal cycle of temperature (summer peaks ~25 °C, winter troughs ~12 °C).  
- Checked dataset shape and values:  
  - Time: 12,331 days (1990–2023)  
  - Spatial grid: 31 lat × 28 lon  

### 3. Visualization  
- Generated time-series plots of daily national means.  
- Created spatial maps (first-day snapshots).  
- Example:  
  - **t2m_c.nc** → seasonal cycle confirmed.  
  - **prcp_daily_mm.nc** → daily accumulated precipitation.  
  - **wind10m_speed_daily.nc** → average wind speed trends.  

### 4. Performance Benchmarking (in-progress)  
- Planned tests: `-n 1`, `-n 2`, `-n 4` MPI ranks.  
- Measure wall-clock times and compute speedup.  
- Expected outcome: near-linear scaling for large slices.  

---

## Deliverables  
1. **MPI Preprocessing Script:** `preprocess_mpi.py`  
   - Parallel resampling and unit conversion for IMDAA NetCDF files.  
2. **Verification Script:** `verify_outputs.py`  
   - Loads final `.nc` files, prints dataset summaries, and produces validation plots.  
3. **Preprocessed Datasets:**  
   - `tmp2m_daily_c.nc` (temperature in °C)  
   - `prcp_daily_mm.nc` (daily precipitation)  
   - `wind10m_speed_daily.nc` (10 m wind speed in m/s)  
4. **Sample Visualizations:**  
   - Seasonal cycle of national average temperature (1990–1992)  
   - Spatial heatmaps and wind vector maps (planned).  
5. **Report & Slides:**  
   - Problem motivation, pipeline diagram, MPI log outputs, verification plots, and scaling results.  

---

## Next Steps  
- Extend analysis to compute **climatology maps** (30-year mean).  
- Add **anomaly detection** for extreme weather events.  
- Benchmark performance across larger node counts.  
- Integrate visualizations into a dashboard for interactive exploration.  
