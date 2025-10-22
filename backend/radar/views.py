import os, gzip, tempfile, time
import numpy as np
import xarray as xr
import requests
from PIL import Image
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_GET
from cfgrib.messages import Message
import eccodes 
from .utils import find_latest_grib_url

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_BOUNDS = [20.0, -130.0, 55.0, -60.0]


def _downsample_data(data_array, factor=8): 
    if data_array.ndim != 2:
        if data_array.ndim > 2:
             print(f"⚠️ Flattening {data_array.ndim}D array -> 2D")
             data_array = data_array.reshape(data_array.shape[-2], data_array.shape[-1])
        else:
             print(f"⚠️ Unexpected data array dimension: {data_array.ndim}. Returning as is.")
             return data_array


    ny, nx = data_array.shape


    ny_new = ny // factor
    nx_new = nx // factor


    data_truncated = data_array[:ny_new * factor, :nx_new * factor]


    downsampled_array = data_truncated.reshape(
        ny_new, factor, nx_new, factor
    ).mean(axis=(1, 3))

    print(f"📉 Downsampling from ({ny}, {nx}) to ({ny_new}, {nx_new}) (Factor: {factor})")
    return downsampled_array 


def _get_grib_data_low_memory(filepath):
    fhandle = None
    try:
        fhandle = eccodes.codes_open_file(filepath, 'r')
        message = eccodes.codes_grib_find_next(fhandle)
        if message is None:
            raise ValueError("No GRIB messages found in file.")

        values = eccodes.codes_get_values(message)
        eccodes.codes_release(message)
        

        Ni = eccodes.codes_get(message, 'Ni')
        Nj = eccodes.codes_get(message, 'Nj')
        
        if Ni * Nj != values.size:
             raise ValueError(f"Dimensions Ni*Nj ({Ni*Nj}) do not match value size ({values.size})")
        
        data = values.reshape((Nj, Ni))
        return data

    finally:
        if fhandle:
            eccodes.codes_close_file(fhandle)

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


@require_GET
def metadata(request):
    meta_file = os.path.join(CACHE_DIR, "latest_meta.txt")
    last_updated = open(meta_file).read().strip() if os.path.exists(meta_file) else None
    return JsonResponse({
        "image_url": request.build_absolute_uri("/api/radar/latest.png"),
        "bounds": DEFAULT_BOUNDS,
        "last_updated": last_updated
    }, headers={"Access-Control-Allow-Origin": "*"}) 


@require_GET
def latest_png(request):
    tmp_file_path = None
    
    try:
        grib_url = find_latest_grib_url()
        if not grib_url:
            return HttpResponse("No MRMS file found", status=502, headers={"Access-Control-Allow-Origin": "*"})

        basename = os.path.basename(grib_url)
        cache_gz = os.path.join(CACHE_DIR, basename)
        cache_png = os.path.join(CACHE_DIR, "latest.png")
        cache_meta = os.path.join(CACHE_DIR, "latest_meta.txt")

        if not os.path.exists(cache_gz):
            print(f"⬇️ Downloading GRIB file: {grib_url}")
            r = requests.get(grib_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(cache_gz, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
        tmp_file_path = tmp.name
        tmp.close() 

        print("🗜️ Decompressing GRIB file to temporary file.")
        with gzip.open(cache_gz, "rb") as gz_in:
             with open(tmp_file_path, "wb") as f_out:
                 while True:
                     chunk = gz_in.read(8192)
                     if not chunk:
                         break
                     f_out.write(chunk)

        data_raw = _get_grib_data_low_memory(tmp_file_path)
        
        if data_raw is None or data_raw.size == 0:
            raise ValueError("Failed to extract raw data using eccodes.")
        
        print("✅ Successfully extracted raw GRIB data")
        
        data = _downsample_data(data_raw, factor=8) 

        print(f"✅ Final data shape: {data.shape}, min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}")

        rgba = _reflectivity_to_rgba(data)
        Image.fromarray(rgba, "RGBA").save(cache_png)
        
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path) 

        with open(cache_meta, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

        return HttpResponse(open(cache_png, "rb").read(), content_type="image/png", headers={"Access-Control-Allow-Origin": "*"}) # ADDED CORS HEADER

    except Exception as e:
        print("❌ ERROR:", e)
        
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass
        
        if os.path.exists(cache_png):
            return HttpResponse(open(cache_png, "rb").read(), content_type="image/png", headers={"Access-Control-Allow-Origin": "*"}) # ADDED CORS HEADER
        return HttpResponse(f"Error: {e}", status=500, headers={"Access-Control-Allow-Origin": "*"}) # ADDED CORS HEADER
