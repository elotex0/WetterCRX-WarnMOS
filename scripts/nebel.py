import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
from matplotlib import patheffects
from zoneinfo import ZoneInfo
import warnings
from pyproj import Proj, Transformer

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe / Ausgabe
# ------------------------------
DATAPATH = "data/warnmos"
OUTPUT_DIR = "warnmos/nebel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Deutschland-Extent
extent = [5, 16, 47, 56]

# Städte
cities = pd.DataFrame({
    'name': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Dresden', 'Stuttgart', 'Düsseldorf',
             'Nürnberg', 'Erfurt', 'Leipzig', 'Bremen', 'Saarbrücken', 'Hannover'],
    'lat': [52.52, 53.55, 48.14, 50.94, 50.11, 51.05, 48.78, 51.23,
            49.45, 50.98, 51.34, 53.08, 49.24, 52.37],
    'lon': [13.40, 9.99, 11.57, 6.96, 8.68, 13.73, 9.18, 6.78,
            11.08, 11.03, 12.37, 8.80, 6.99, 9.73]
})

# Frost-Wahrscheinlichkeit Farben
fz_bounds = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

colors = ListedColormap([
    "#FE9226", "#FFC02B", "#FFEE32", "#DDE02D", "#BBD629",
    "#9AC925", "#79BC21", "#37A319", "#367C40",
    "#366754", "#4A3E7C", "#593192", "#800080"
])

fz_cmap = colors
fz_norm = BoundaryNorm(fz_bounds, fz_cmap.N)


# ------------------------------
# Kartenparameter (klein, wie im Beispiel)
# ------------------------------
FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT = FIG_W_PX / TOP_AREA_PX


# ------------------------------
# Polar Stereographic Projektion definieren
# ------------------------------
# Basierend auf den GRIB-Parametern:
# LoV = 10° (orientation)
# LaD = 60° (standard parallel, typical for DWD)
proj_ps = Proj(proj='stere', lat_0=90, lon_0=10, lat_ts=60, x_0=0, y_0=0, ellps='WGS84')
proj_geo = Proj(proj='latlong', datum='WGS84')
transformer = Transformer.from_proj(proj_ps, proj_geo, always_xy=True)

# ------------------------------
# GRIB-Dateien durchgehen
# ------------------------------
files = sorted([os.path.join(DATAPATH, f) for f in os.listdir(DATAPATH) if f.endswith(".grb2")])
if not files:
    raise FileNotFoundError("Keine GRIB2-Dateien gefunden!")

for filename in files:
    print("Lade Datei:", filename)
    ds = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={"shortName": "max_vis", "lowerLimit": 150.0})
    varname = list(ds.data_vars)[0]
    data = ds[varname]

    if data.ndim == 3:
        n_steps = data.shape[0]
    else:
        n_steps = 1

    # Native Grid-Koordinaten aus GRIB-Parametern berechnen
    # First point: 46.957°N, 3.594°E
    # Dx, Dy = 1000m = 1km
    nx, ny = 900, 900
    dx, dy = 1000.0, 1000.0  # in Metern
    
    # Ersten Punkt in Projektion transformieren
    x0, y0 = proj_ps(3.594, 46.957)
    
    # Grid erstellen
    x = x0 + np.arange(nx) * dx
    y = y0 + np.arange(ny) * dy
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Zurück in geografische Koordinaten transformieren
    print("Berechne geografische Koordinaten...")
    lon2d, lat2d = transformer.transform(x_grid, y_grid)
    
    print("lon-range:", lon2d.min(), lon2d.max())
    print("lat-range:", lat2d.min(), lat2d.max())


    # Laufzeit
    if 'time' in ds:
        tval = ds['time'].values
        if np.ndim(tval) > 0:
            run_time_utc = pd.to_datetime(tval[0])
        else:
            run_time_utc = pd.to_datetime(tval)
    else:
        run_time_utc = None

    # Für jeden Step
    for step in range(n_steps):
        if data.ndim == 3:
            data_plot = data[step].values
            
            if 'step' in ds:
                step_val = ds['step'].values[step]
                if isinstance(step_val, np.timedelta64):
                    valid_time_utc = run_time_utc + pd.to_timedelta(step_val)
                else:
                    valid_time_utc = run_time_utc + pd.to_timedelta(step_val, unit='s')
            else:
                valid_time_utc = run_time_utc + pd.to_timedelta(step, unit='h')

            valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))
        else:
            data_plot = data.values
            valid_time_local = run_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))




        # --------------------------
        # Figure + Plot
        # --------------------------
        scale = 0.9
        shift_up = 0.02
        fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
        ax = fig.add_axes([0.0, BOTTOM_AREA_PX / FIG_H_PX + shift_up, 1.0, TOP_AREA_PX / FIG_H_PX],
                          projection=ccrs.PlateCarree())
        ax.set_extent(extent)
        ax.set_axis_off()
        ax.set_aspect('auto')

        # Grenzen, Küste, Bundesländer
        ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)

        # Frost-Wahrscheinlichkeit plotten
        print(f"Plotte Step {step}...")
        # 1. Basisfarbe über die ganze Karte
        ax.contourf(lon2d, lat2d, np.full_like(data_plot, fz_bounds[0]),
                    levels=fz_bounds, cmap=fz_cmap, norm=fz_norm, extend='neither')

        # 2. Eigentliche Daten darüber plotten
        im = ax.contourf(lon2d, lat2d, data_plot,
                        levels=fz_bounds, cmap=fz_cmap, norm=fz_norm, extend='neither')

        # Städte plotten
        for _, city in cities.iterrows():
            ax.plot(city["lon"], city["lat"], "o", markersize=6, markerfacecolor="black",
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            txt = ax.text(city["lon"]+0.1, city["lat"]+0.1, city["name"], fontsize=9,
                          color="black", weight="bold", zorder=6)
            txt.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground="white")])

        # --------------------------
        # Farbskala (Legende) unten
        # --------------------------
        legend_h_px = 50
        legend_bottom_px = 45
        cbar_ax = fig.add_axes([0.03, legend_bottom_px/FIG_H_PX, 0.94, legend_h_px/FIG_H_PX])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=fz_bounds)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.set_facecolor("white")
        cbar.outline.set_edgecolor("black")

        # --------------------------
        # Footer über Legende
        # --------------------------
        footer_ax = fig.add_axes([0.0, (legend_bottom_px + legend_h_px)/FIG_H_PX, 1.0,
                                  (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px)/FIG_H_PX])
        footer_ax.axis("off")
        left_text = f"Nebel Wahrscheinlichkeit (%)\nWarnMOS ({run_time_utc.strftime('%Hz') if run_time_utc else 'run'}), Deutscher Wetterdienst"
        footer_ax.text(0.01, 0.85, left_text, fontsize=12, fontweight="bold", va="top", ha="left")
        footer_ax.text(0.734, 0.92, "Prognose für:", fontsize=12, va="top", ha="left", fontweight="bold")
        footer_ax.text(0.99, 0.68, f"{valid_time_local:%d.%m.%Y %H:%M} Uhr",
                   fontsize=12, va="top", ha="right", fontweight="bold")

        # --------------------------
        # Speichern
        # --------------------------
        outname = f"warnnebel_{valid_time_local:%Y%m%d_%H%M}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, outname), dpi=100, bbox_inches=None, pad_inches=0)
        plt.close()
        print("PNG erzeugt:", outname)
