"""Fetch and prepare the public data for the Cobb County siting experiment.

Outputs (in data/):
  tracts.csv      186 Cobb County census tracts: GEOID, lat, lon, households (+MOE),
                  median household income (+MOE). ACS 2020-2024 5-year, table-based
                  summary files (no API key needed); centroids from the 2024 Gazetteer.
  candidates.csv  Publix- and Kroger-anchored grocery sites inside Cobb County, from
                  OpenStreetMap (Overpass API), filtered by the TIGER cartographic
                  county boundary.
  cobb_outline.json  simplified county outline (lon/lat) for the figures.

Run:  uv run --with numpy --with pyshp --with shapely python fetch_data.py
Raw downloads are cached in RAW_DIR (default ./raw, git-ignored).
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile

import shapefile  # pyshp
from shapely.geometry import Point, shape

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("RAW_DIR", os.path.join(HERE, "raw"))
OUT = os.path.join(HERE, "data")
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

ACS = "https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData"
GAZ = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_gaz_tracts_13.txt"
COUNTY_SHP = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_500k.zip"
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_Q = (
    '[out:json][timeout:90];nwr["shop"="supermarket"](33.70,-84.76,34.10,-84.35);out center tags;'
)
COBB = "13067"


def get(url: str, name: str, data: bytes | None = None) -> str:
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        for attempt in range(6):
            try:
                req = urllib.request.Request(
                    url, data=data, headers={"User-Agent": "or-theses-applied/1.0"}
                )
                with urllib.request.urlopen(req, timeout=180) as r, open(path, "wb") as f:
                    f.write(r.read())
                break
            except Exception as e:  # Overpass is often busy: back off and retry
                print("retry", name, e)
                time.sleep(10 * 2**attempt)
    return path


def acs(table: str) -> dict[str, dict[str, str]]:
    path = get(f"{ACS}/acsdt5y2024-{table}.dat", f"acsdt5y2024-{table}.dat")
    with open(path) as f:
        rows = list(csv.reader(f, delimiter="|"))
    head = rows[0]
    return {
        r[0][9:]: dict(zip(head, r)) for r in rows[1:] if r[0].startswith(f"1400000US{COBB}")
    }


def main() -> None:
    hh = acs("b11001")
    inc = acs("b19013")
    with open(get(GAZ, "2024_gaz_tracts_13.txt")) as f:
        gaz = {
            r[1]: (float(r[6]), float(r[7].strip()))
            for r in csv.reader(f, delimiter="\t")
            if r[1].startswith(COBB)
        }
    tracts = []
    for geoid in sorted(hh):
        lat, lon = gaz[geoid]
        tracts.append(
            dict(
                geoid=geoid,
                lat=lat,
                lon=lon,
                households=int(hh[geoid]["B11001_E001"]),
                households_moe=int(hh[geoid]["B11001_M001"]),
                income=int(inc[geoid]["B19013_E001"]),
                income_moe=int(inc[geoid]["B19013_M001"]),
            )
        )
    with open(os.path.join(OUT, "tracts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tracts[0]))
        w.writeheader()
        w.writerows(tracts)

    # County boundary (TIGER cartographic, 1:500k)
    zpath = get(COUNTY_SHP, "cb_2024_us_county_500k.zip")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        stem = [n for n in names if n.endswith(".shp")][0][:-4]
        sf = shapefile.Reader(
            shp=io.BytesIO(z.read(stem + ".shp")),
            dbf=io.BytesIO(z.read(stem + ".dbf")),
            shx=io.BytesIO(z.read(stem + ".shx")),
        )
        fields = [fl[0] for fl in sf.fields[1:]]
        cobb = None
        for sr in sf.iterShapeRecords():
            rec = dict(zip(fields, sr.record))
            if rec["GEOID"] == COBB:
                cobb = shape(sr.shape.__geo_interface__)
    outline = cobb.simplify(0.002)
    with open(os.path.join(OUT, "cobb_outline.json"), "w") as f:
        json.dump([[round(x, 5), round(y, 5)] for x, y in outline.exterior.coords], f)

    # Grocery anchors from OpenStreetMap
    opath = get(OVERPASS, "osm_supermarkets.json", urllib.parse.urlencode({"data": OVERPASS_Q}).encode())
    elements = json.load(open(opath))["elements"]
    cands = []
    for e in elements:
        name = e.get("tags", {}).get("name", "")
        if name not in ("Publix", "Kroger"):
            continue
        lat = e.get("lat", e.get("center", {}).get("lat"))
        lon = e.get("lon", e.get("center", {}).get("lon"))
        if lat is None or not cobb.contains(Point(lon, lat)):
            continue
        t = e.get("tags", {})
        street = " ".join(x for x in (t.get("addr:housenumber"), t.get("addr:street")) if x)
        cands.append(
            dict(
                osm=f"{e['type']}/{e['id']}",
                anchor=name,
                lat=round(lat, 6),
                lon=round(lon, 6),
                street=street,
                city=t.get("addr:city", ""),
            )
        )
    # drop duplicates mapped twice (node + building within 150 m)
    keep = []
    for c in sorted(cands, key=lambda c: c["osm"]):
        if all(
            ((c["lat"] - k["lat"]) * 111) ** 2 + ((c["lon"] - k["lon"]) * 92) ** 2 > 0.15**2
            for k in keep
        ):
            keep.append(c)
    keep.sort(key=lambda c: (c["lat"], c["lon"]))
    for i, c in enumerate(keep):
        c["id"] = f"S{i + 1:02d}"
    with open(os.path.join(OUT, "candidates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "anchor", "lat", "lon", "street", "city", "osm"])
        w.writeheader()
        w.writerows(keep)
    # Driving distances (miles) from every tract centroid to every candidate, public OSRM
    # server, in chunks of 50 tracts (one-off; cached in raw/).
    cstr = ";".join(f"{c['lon']},{c['lat']}" for c in keep)
    matrix = []
    for k0 in range(0, len(tracts), 50):
        chunk = tracts[k0 : k0 + 50]
        tstr = ";".join(f"{t['lon']},{t['lat']}" for t in chunk)
        src = ";".join(str(i) for i in range(len(chunk)))
        dst = ";".join(str(len(chunk) + i) for i in range(len(keep)))
        url = (
            f"https://router.project-osrm.org/table/v1/driving/{tstr};{cstr}"
            f"?annotations=distance&sources={src}&destinations={dst}"
        )
        res = json.load(open(get(url, f"osrm_{k0}.json")))
        assert res["code"] == "Ok", res
        matrix += [[round(d / 1609.344, 3) for d in row] for row in res["distances"]]
        time.sleep(1)
    with open(os.path.join(OUT, "drive_miles.json"), "w") as f:
        json.dump({"tracts": [t["geoid"] for t in tracts], "sites": [c["id"] for c in keep],
                   "miles": matrix}, f)

    print(len(tracts), "tracts;", sum(t["households"] for t in tracts), "households;", len(keep), "candidates")


if __name__ == "__main__":
    main()
