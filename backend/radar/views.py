import os, gzip, tempfile, time
import numpy as np
import xarray as xr
import requests
from PIL import Image
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_GET

# Assume 'utils' is available in the same directory
from .utils import find_latest_grib_url

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_BOUNDS = [20.0, -130.0, 55.0, -60.0] 

# --- GRIB DATA PROCESSING UTILITIES ---

def _downsample_data(da: xr.DataArray, factor: int = 8) -> np.ndarray:
    """Downsamples a DataArray using spatial averaging to reduce memory footprint."""
    
    # Use xarray's rolling window to compute the mean, significantly reducing the grid size.
    # We downsample by 'factor' in both dimensions (latitude and longitude).
    downsampled_array = da.coarsen(x=factor, y=factor, boundary='trim').mean()
    
    # This is the point where the smaller array is finally loaded into memory.
    data = downsampled_array.to_numpy(dtype=float)
    
    print(f"✅ Downsampled from {da.shape} to {data.shape} (Factor: {factor})")
    return data

def _reflectivity_to_rgba(data: np.ndarray) -> np.ndarray:
    """Maps reflectivity values (dBZ) to a colormap with transparency."""
    
    # These colors provide a standard weather radar palette
    bins = [-999, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]
    colors = [
        (0, 0, 0, 0),        # Below threshold (Transparent)
        (157,160,255,180),   # Light Blue
        (96,122,255,200),    # Blue
        (40,156,255,200),    # Cyan Blue
        (0,219,255,200),     # Light Cyan
        (0,255,170,200),     # Light Green
        (0,255,80,200),      # Green
        (132,255,0,200),     # Yellow Green
        (231,255,0,200),     # Yellow
        (255,238,0,200),     # Orange Yellow
        (255,206,0,200),     # Orange
        (255,150,0,200),     # Dark Orange
        (255,100,0,200),     # Reddish Orange
        (255,40,40,200),     # Red
        (204,0,0,200),       # Dark Red
        (143,0,0,200),       # Deep Red
        (90,0,0,200),        # Purple Red
        (50,0,0,200)         # Max Intensity
    ]

    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

    # Filter out extreme outliers, setting them to NaN (will be transparent)
    data = np.where((data < -50) | (data > 80), np.nan, data)

    # Apply color mapping based on bins
    for i in range(len(bins)-1):
        mask = (data >= bins[i]) & (data < bins[i+1])
        rgba[mask] = colors[i]

    # Set NaN values (no data / extreme outliers) to transparent
    rgba[np.isnan(data)] = (0,0,0,0)
    return rgba


def _open_grib_with_fallback(filepath):
    """Attempts to open the GRIB file using various common variable names."""
    
    filters = [
        {"shortName": "REFL"},  # Reflectivity
        {"shortName": "DZ"},    # Reflectivity
        {"shortName": "RALA"},  # Reflectivity at Lowest Altitude (common for MRMS)
        {"typeOfLevel": "surface"},
        {} 
    ]

    for f in filters:
        try:
            # Use 'cache=False' to potentially reduce memory overhead 
            ds = xr.open_dataset(
                filepath,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": f, "indexpath": "", "cache": False}
            )
            print(f"✅ Opened with filter: {f}")
            if len(ds.data_vars) > 0:
                return ds
        except Exception as e:
            print(f"❌ Failed filter {f}: {e}")
    return None

# --- DJANGO VIEWS ---

@require_GET
def metadata(request):
    """Returns metadata about the latest radar image."""
    meta_file = os.path.join(CACHE_DIR, "latest_meta.txt")
    last_updated = open(meta_file).read().strip() if os.path.exists(meta_file) else None
    
    response = JsonResponse({
        "image_url": request.build_absolute_uri("/api/radar/latest.png"),
        "bounds": DEFAULT_BOUNDS,
        "last_updated": last_updated
    })
    # FIX: Add CORS header
    response["Access-Control-Allow-Origin"] = "*"
    return response


@require_GET
def latest_png(request):
    """Downloads, processes, and serves the latest radar image as a PNG."""
    cache_png = os.path.join(CACHE_DIR, "latest.png")
    tmp_grib_path = None # Initialize to ensure cleanup

    try:
        grib_url = find_latest_grib_url()
        if not grib_url:
            return HttpResponse("No MRMS file found", status=502)

        basename = os.path.basename(grib_url)
        cache_gz = os.path.join(CACHE_DIR, basename)
        cache_meta = os.path.join(CACHE_DIR, "latest_meta.txt")

        # 1. DOWNLOAD (STREAMING)
        if not os.path.exists(cache_gz):
            print(f"⬇️ Downloading GRIB file: {grib_url}")
            r = requests.get(grib_url, stream=True, timeout=60) # Increased timeout
            r.raise_for_status()
            with open(cache_gz, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # 2. DECOMPRESS (STREAMING)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
        tmp_grib_path = tmp.name
        tmp.close()
        
        print("🗜️ Decompressing GRIB file to temporary file.")
        with gzip.open(cache_gz, "rb") as gz_in:
            with open(tmp_grib_path, "wb") as f_out:
                while True:
                    chunk = gz_in.read(8192 * 4) # Read larger chunk for efficiency
                    if not chunk:
                        break
                    f_out.write(chunk)

        # 3. OPEN GRIB FILE
        ds = _open_grib_with_fallback(tmp_grib_path)
        if ds is None:
            raise ValueError("Failed to open GRIB file with any filter.")

        vars_list = list(ds.data_vars.keys())
        if not vars_list:
            raise ValueError("No data variables found in dataset.")

        var_name = vars_list[0]
        da = ds[var_name]
        
        # 4. MEMORY OPTIMIZED PROCESSING (DOWNSAMPLING)
        try:
            data = _downsample_data(da, factor=8)
            
            if data.ndim > 2:
                print(f"⚠️ Flattening {data.ndim}D array -> 2D")
                data = data.reshape(data.shape[-2], data.shape[-1])
            
            print(f"✅ Final data shape: {data.shape}")

        except Exception as e:
            # THIS IS THE CATCH BLOCK THAT PREVIOUSLY CONTAINED THE BUGGY ECCODES FALLBACK
            # We now rely solely on the memory-optimized downsampling method above.
            raise ValueError(f"Failed to extract or downsample reflectivity data: {e}")

        # 5. CONVERT TO RGBA AND SAVE PNG
        rgba = _reflectivity_to_rgba(data)
        Image.fromarray(rgba, "RGBA").save(cache_png)
        
        # 6. UPDATE METADATA
        with open(cache_meta, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

        # 7. SERVE IMAGE
        response = HttpResponse(open(cache_png, "rb").read(), content_type="image/png")
        # FIX: Add CORS header
        response["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:
        print("❌ ERROR:", e)
        
        # If an error occurred, attempt to serve the cached image if available
        if os.path.exists(cache_png):
            print("⚠️ Serving stale cached image due to error.")
            response = HttpResponse(open(cache_png, "rb").read(), content_type="image/png")
            response["Access-Control-Allow-Origin"] = "*"
            return response
            
        return HttpResponse(f"Error: {e}", status=500)
        
    finally:
        # 8. CLEANUP: Ensure temporary files are deleted
        try:
            if 'ds' in locals() and ds is not None:
                ds.close()
            if tmp_grib_path and os.path.exists(tmp_grib_path):
                os.unlink(tmp_grib_path)
        except Exception as cleanup_e:
            print(f"Cleanup error: {cleanup_e}")
