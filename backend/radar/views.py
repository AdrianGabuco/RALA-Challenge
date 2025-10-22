import os, gzip, tempfile, time
import numpy as np
import xarray as xr
import requests
from PIL import Image
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_GET
from .utils import find_latest_grib_url

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_BOUNDS = [20.0, -130.0, 55.0, -60.0]  # south, west, north, east

def _reflectivity_to_rgba(data):
    # Define reflectivity bins (in dBZ)
    bins = [-999, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
    colors = [
        (0, 0, 0, 0),          # No data / very low
        (157,160,255,180),
        (96,122,255,200),
        (40,156,255,200),
        (0,219,255,200),
        (0,255,170,200),
        (0,255,80,200),
        (132,255,0,200),
        (231,255,0,200),
        (255,238,0,200),
        (255,206,0,200),
        (255,150,0,200),
        (255,100,0,200),
        (255,40,40,200),
        (204,0,0,200),
        (143,0,0,200),
        (90,0,0,200),
        (50,0,0,200)
    ]

    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

    # Handle missing values
    data = np.where((data < -50) | (data > 80), np.nan, data)

    for i in range(len(bins)-1):
        mask = (data >= bins[i]) & (data < bins[i+1])
        rgba[mask] = colors[i]

    rgba[np.isnan(data)] = (0,0,0,0)
    return rgba


def _open_grib_with_fallback(filepath):
    filters = [
        {"shortName": "REFL"},  
        {"shortName": "DZ"}, 
        {"shortName": "RALA"},   
        {"typeOfLevel": "surface"},
        {}  
    ]

    for f in filters:
        try:
            ds = xr.open_dataset(
                filepath,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": f, "indexpath": ""}
            )
            print(f"✅ Opened with filter: {f}")
            if len(ds.data_vars) > 0:
                return ds
        except Exception as e:
            print(f"❌ Failed filter {f}: {e}")
    return None


@require_GET
def metadata(request):
    meta_file = os.path.join(CACHE_DIR, "latest_meta.txt")
    last_updated = open(meta_file).read().strip() if os.path.exists(meta_file) else None
    return JsonResponse({
        "image_url": request.build_absolute_uri("/api/radar/latest.png"),
        "bounds": DEFAULT_BOUNDS,
        "last_updated": last_updated
    })


@require_GET
def latest_png(request):
    try:
        grib_url = find_latest_grib_url()
        if not grib_url:
            return HttpResponse("No MRMS file found", status=502)

        basename = os.path.basename(grib_url)
        cache_gz = os.path.join(CACHE_DIR, basename)
        cache_png = os.path.join(CACHE_DIR, "latest.png")
        cache_meta = os.path.join(CACHE_DIR, "latest_meta.txt")

        # Download if not cached
        if not os.path.exists(cache_gz):
            r = requests.get(grib_url, timeout=30)
            r.raise_for_status()
            with open(cache_gz, "wb") as f:
                f.write(r.content)

        with gzip.open(cache_gz, "rb") as gz:
            grib_bytes = gz.read()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
        tmp.write(grib_bytes)
        tmp.close()

        ds = _open_grib_with_fallback(tmp.name)
        if ds is None:
            raise ValueError("Failed to open GRIB file with any filter.")

        print("✅ Successfully opened GRIB file")
        print("Available dimensions:", ds.dims)
        print("Available variables:", list(ds.data_vars.keys()))

        vars_list = list(ds.data_vars.keys())

        if not vars_list:       
            raise ValueError(f"No data variables found. Dataset summary: {ds}")


        vars_list = list(ds.data_vars.keys())

        if not vars_list:
            raise ValueError(f"No data variables found. Dataset summary: {ds}")

        vars_list = list(ds.data_vars.keys())
        if not vars_list:
            raise ValueError(f"No data variables found. Dataset summary: {ds}")

        var_name = vars_list[0]
        print(f"✅ Using variable: {var_name}")

        try:
            # Safely read without assuming extra dimensions
            arr = ds[var_name].values
            if hasattr(arr, "shape"):
                print(f"📏 Raw data shape: {arr.shape}")
            else:
                raise ValueError("No .values shape attribute found")

            # Handle scalars or empty data
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                raise ValueError("Empty reflectivity array in GRIB variable.")

            # Convert to float array
            data = np.array(arr, dtype=float)

            # Flatten time/height dimensions if needed
            if data.ndim > 2:
                print(f"⚠️ Flattening {data.ndim}D array -> 2D")
                data = data.reshape(data.shape[-2], data.shape[-1])

            print(f"✅ Final data shape: {data.shape}, min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}")

        except Exception as e:
            print(f"⚠️ Could not extract data normally: {e}")
            # Try using cfgrib raw data API
            try:
                from cfgrib import open_file
                with open_file(ds.encoding["source"]) as f:
                    messages = list(f)
                    if not messages:
                        raise ValueError("No GRIB messages found.")
                    msg = messages[0]
                    values = msg.values
                    if not isinstance(values, np.ndarray):
                        raise ValueError("No valid numeric data in message.")
                    data = np.array(values, dtype=float)
                    print(f"✅ Extracted raw GRIB data with shape {data.shape}")
            except Exception as e2:
                raise ValueError(f"Failed to extract reflectivity data: {e2}")

        rgba = _reflectivity_to_rgba(data)
        Image.fromarray(rgba, "RGBA").save(cache_png)
        ds.close()
        os.unlink(tmp.name)

        with open(cache_meta, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

        return HttpResponse(open(cache_png, "rb").read(), content_type="image/png")

    except Exception as e:
        print("❌ ERROR:", e)
        if os.path.exists(cache_png):
            return HttpResponse(open(cache_png, "rb").read(), content_type="image/png")
        return HttpResponse(f"Error: {e}", status=500)