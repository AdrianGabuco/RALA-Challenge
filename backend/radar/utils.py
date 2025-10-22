import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

MRMS_INDEX = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/"

def find_latest_grib_url():
    r = requests.get(MRMS_INDEX, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    pattern = re.compile(r"MRMS_ReflectivityAtLowestAltitude.*\.grib2\.gz")
    candidates = [urljoin(MRMS_INDEX, a["href"]) for a in soup.find_all("a", href=True) if pattern.search(a["href"])]
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]