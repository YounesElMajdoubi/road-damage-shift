"""Extraction sélective : testée sur 4 organisations d'archive possibles (sans réseau)."""

import hashlib
import io
import zipfile

import pytest

from roadshift.data.convert import discover_countries
from roadshift.data.download import count_xml, country_of, extract_countries, md5sum

ALL = ["Japan", "India", "Czech", "Norway", "United_States", "China_MotorBike", "China_Drone"]
KEEP = {"Japan", "India", "Czech", "United_States"}


def _zbytes(files: dict) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return b.getvalue()


def _country(c):
    return {
        f"{c}/train/images/{c}_000001.jpg": "img",
        f"{c}/train/annotations/xmls/{c}_000001.xml": "<a/>",
        f"{c}/test/images/{c}_009999.jpg": "t",
    }


LAYOUTS = {
    "zips_par_pays": {f"RDD2022_all_countries/RDD2022_{c}.zip": _zbytes(_country(c)) for c in ALL},
    "zip_global_imbrique": {"RDD2022.zip": _zbytes({f"RDD2022/{c}.zip": _zbytes(_country(c)) for c in ALL})},
    "dossiers": {f"RDD2022/{k}": v for c in ALL for k, v in _country(c).items()},
    "double_imbrication": {f"x/RDD2022_{c}.zip": _zbytes({f"{c}.zip": _zbytes(_country(c))}) for c in ALL},
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_extract_only_requested_countries(tmp_path, layout):
    zp = tmp_path / "big.zip"
    zp.write_bytes(_zbytes(LAYOUTS[layout]))
    dest = tmp_path / "raw"
    dest.mkdir()
    extract_countries(zp, dest, KEEP)
    assert set(count_xml(dest)) == KEEP  # Norvège et Chine jamais extraites
    assert not list(dest.rglob("*.zip")) and not list(dest.glob("_nested_*"))
    assert set(discover_countries(dest, [*KEEP, "Norway"])) == KEEP  # le convertisseur les retrouve


def test_country_of():
    assert country_of("RDD2022_China_MotorBike.zip") == "China_MotorBike"
    assert country_of("RDD2022/China_Drone/train/x.xml") == "China_Drone"
    assert country_of("a/United_States/train/x.xml") == "United_States"
    assert country_of("README.txt") is None


def test_md5sum(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"road-damage" * 1000)
    assert md5sum(p, chunk=7) == hashlib.md5(b"road-damage" * 1000).hexdigest()
