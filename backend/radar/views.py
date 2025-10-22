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

DEFAULT_BOUNDS = [20.0, -130.0, 55.0, -60.0]

# --- NEW MEMORY OPTIMIZATION FUNCTION ---
def _downsample_data(data_array, factor=4):
    """
    Downsamples the xarray DataArray using block averaging.
    This is essential to reduce memory usage from large GRIB grids.
    A factor of 4 reduces the data size by 16x.
    """
    if data_array.ndim != 2:
        # If not 2D, return as is or handle reshaping before calling this function
        # (your existing code handles flattening later, but we prefer 2D here)
        return data_array

    # 1. Access the underlying Dask array (lazy operation, no huge memory load yet)
    data = data_array.data

    # 2. Reshape and mean calculation (still often lazy)
    # Get original dimensions
    ny, nx = data.shape

    # Calculate new dimensions for downsampling
    ny_new = ny // factor
    nx_new = nx // factor

    # Truncate data to be perfectly divisible by factor
    data_truncated = data[:ny_new * factor, :nx_new * factor]

    # Reshape for block averaging: (ny_new, factor, nx_new, factor)
    # Then take the mean over the two 'factor' axes (axes 1 and 3)
    downsampled_array = data_truncated.reshape(
        ny_new, factor, nx_new, factor
    ).mean(axis=(1, 3))

    # 3. FORCE the calculation to a NumPy array to load the *smaller* result
    # into memory. This is where memory is actually used.
    print(f"📉 Downsampling from ({ny}, {nx}) to ({ny_new}, {nx_new}) (Factor: {factor})")
    return np.array(downsampled_array, dtype=float)

# --- END NEW FUNCTION ---

def _reflectivity_to_rgba(data):

    bins = [-999, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
    colors = [
        (0, 0, 0, 0),
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
            # We are NOT forcing ds.load() here. Keep it lazy.
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
    # Ensure all file variables are defined at the start for cleanup
    tmp_file_path = None
    ds = None
    
    try:
        grib_url = find_latest_grib_url()
        if not grib_url:
            return HttpResponse("No MRMS file found", status=502)

        basename = os.path.basename(grib_url)
        cache_gz = os.path.join(CACHE_DIR, basename)
        cache_png = os.path.join(CACHE_DIR, "latest.png")
        cache_meta = os.path.join(CACHE_DIR, "latest_meta.txt")


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
        tmp_file_path = tmp.name # Save temp file path for cleanup

        ds = _open_grib_with_fallback(tmp_file_path)
        if ds is None:
            raise ValueError("Failed to open GRIB file with any filter.")

        print("✅ Successfully opened GRIB file")
        
        vars_list = list(ds.data_vars.keys())

        if not vars_list:
            raise ValueError(f"No data variables found. Dataset summary: {ds}")

        var_name = vars_list[0]
        print(f"✅ Using variable: {var_name}")

        # --- MEMORY OPTIMIZATION IMPLEMENTATION ---
        
        # 1. Get the DataArray (still likely a Dask array, thus lazy)
        da = ds[var_name]

        # 2. Downsample the data array *before* forcing it into NumPy memory
        # If the original data is (3500, 7000) (24.5 million points):
        # Downsampling by factor=4 reduces it to (875, 1750) (1.5 million points), a 16x reduction.
        # This will fit within the 512MB limit.
        data = _downsample_data(da, factor=4)
        
        # NOTE: If memory errors persist, increase factor to 8 (64x reduction) or more.
        
        # --- END MEMORY OPTIMIZATION ---

        print(f"✅ Final data shape: {data.shape}, min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}")


        rgba = _reflectivity_to_rgba(data)
        Image.fromarray(rgba, "RGBA").save(cache_png)
        
        # IMPORTANT: Close the dataset immediately after processing to release memory/file handles
        ds.close()
        os.unlink(tmp_file_path) # Delete the temporary file

        with open(cache_meta, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

        return HttpResponse(open(cache_png, "rb").read(), content_type="image/png")

    except Exception as e:
        print("❌ ERROR:", e)
        
        # Cleanup routine
        if ds is not None:
             try:
                 ds.close()
             except Exception:
                 pass
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass
        
        # Existing fallback to cached PNG
        if os.path.exists(cache_png):
            return HttpResponse(open(cache_png, "rb").read(), content_type="image/png")
        return HttpResponse(f"Error: {e}", status=500)