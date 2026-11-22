"""Téléchargement de RDD2022 depuis figshare (dépôt officiel, licence CC BY 4.0) et extraction sélective.

Usage :
    python -m roadshift.data.download --out /tmp/rdd_raw --countries Japan India Czech United_States

Le serveur S3 du challenge renvoie 403 depuis 2026 : figshare est la source officielle restante.
Le zip (≈ 13 Go) est vérifié par MD5, puis seuls les pays demandés sont extraits, quelle que soit
l'organisation de l'archive (dossiers par pays, zips par pays, zip global imbriqué).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

FIGSHARE_ARTICLE = 21431547
ALL_COUNTRIES = ["Japan", "India", "Czech", "Norway", "United_States", "China_MotorBike", "China_Drone"]


def _norm(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_")


def country_of(name: str) -> str | None:
    """Pays RDD2022 nommé dans un chemin d'archive (dossier 'Japan' ou zip 'RDD2022_Japan.zip'), sinon None."""
    stems = [_norm(p).removesuffix(".zip") for p in Path(name).parts]
    for c in sorted(ALL_COUNTRIES, key=len, reverse=True):
        cn = _norm(c)
        if any(s == cn or s.endswith("_" + cn) for s in stems):
            return c
    return None


def extract_all(zip_path: Path, dest: Path) -> None:
    """Décompresse un zip puis, récursivement, les zips qu'il contenait ; supprime les zips."""
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    Path(zip_path).unlink()
    while inner := sorted(Path(dest).rglob("*.zip")):
        for zp in inner:
            extract_all(zp, zp.parent)


def extract_countries(zip_path: Path, dest: Path, keep: set[str]) -> None:
    """N'extrait que les pays de `keep` ; les autres ne sont jamais décompressés."""
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            c = country_of(m.filename)
            is_zip = m.filename.lower().endswith(".zip")
            if c is None and is_zip:  # archive englobante : on descend dedans
                tmp_dir = Path(tempfile.mkdtemp(dir=dest, prefix="_nested_"))
                extract_countries(Path(z.extract(m, tmp_dir)), dest, keep)
                shutil.rmtree(tmp_dir)
            elif c in keep and is_zip:
                extract_all(Path(z.extract(m, dest / c)), dest / c)
            elif c in keep:
                z.extract(m, dest)


def md5sum(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def count_xml(raw: Path) -> dict[str, int]:
    n: dict[str, int] = {}
    for x in Path(raw).rglob("*.xml"):
        c = country_of(str(x.relative_to(raw)))
        n[c] = n.get(c, 0) + 1
    return n


def download(out: Path, countries: list[str], tmp_dir: Path = Path("/tmp")) -> dict:
    """Idempotent : ne refait rien si l'extraction de ces pays est déjà marquée comme faite."""
    out.mkdir(parents=True, exist_ok=True)
    marker = out / ("done_" + "-".join(sorted(countries)))
    if not marker.exists():
        meta = json.load(urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}", timeout=60))
        f = max(meta["files"], key=lambda x: x["size"])
        zp = Path(tmp_dir) / f["name"]
        print(f"Téléchargement {f['name']} ({f['size'] / 1e9:.2f} Go), licence {meta['license']['name']}", flush=True)
        t0 = time.time()
        subprocess.run(["wget", "-c", "-nv", "-O", str(zp), f["download_url"]], check=True)  # -c : reprise
        print(f"  {time.time() - t0:.0f} s", flush=True)
        expected = f.get("computed_md5") or f.get("supplied_md5")
        if expected:
            got = md5sum(zp)
            if got != expected:
                zp.unlink()
                raise RuntimeError(f"MD5 différent ({got} != {expected}) : zip supprimé, relancer")
            print("  MD5 vérifié :", got, flush=True)
        extract_countries(zp, out, set(countries))
        zp.unlink()
        marker.touch()
    n = count_xml(out)
    print("XML par pays :", n, flush=True)
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/tmp/rdd_raw"))
    ap.add_argument("--countries", nargs="+", default=["Japan", "India", "Czech", "United_States"])
    args = ap.parse_args(argv)
    download(args.out, args.countries)


if __name__ == "__main__":
    main()
