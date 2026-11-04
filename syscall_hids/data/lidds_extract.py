"""
LID-DS Step 0: Batch Extraction & Caching

Tiap recording LID-DS 2021 tersimpan sebagai .zip yang berisi
{.sc, .json, .pcap, .res}. Script ini mengekstrak HANYA .sc (syscall trace)
dan .json (metadata) ke direktori cache, mengabaikan .pcap/.res untuk
menghemat ruang. Idempotent: recording yang sudah diekstrak dilewati.

Struktur sumber:
  LID-DS/<scenario>/<scenario>/{training,validation,test/normal,test/normal_and_attack}/*.zip

Struktur output:
  LID-DS_extracted/<scenario>/<partition>/<recording>.{sc,json}

Penggunaan:
  python -m syscall_hids.data.lidds_extract                 # ekstrak skenario default (3)
  python -m syscall_hids.data.lidds_extract --all           # ekstrak seluruh 7 skenario
  python -m syscall_hids.data.lidds_extract --scenarios CVE-2014-0160 Juice-Shop
"""

import argparse
import sys
import zipfile
from pathlib import Path

SOURCE_ROOT = Path("LID-DS")
OUTPUT_ROOT = Path("LID-DS_extracted")

# Default: 3 skenario representatif untuk evaluasi lintas dataset.
DEFAULT_SCENARIOS = [
    "CVE-2014-0160",        # Heartbleed (information leak)
    "Bruteforce_CWE-307",   # credential brute-force
    "CWE-89-SQL-injection", # SQL injection
]
ALL_SCENARIOS = [
    "Bruteforce_CWE-307", "CVE-2014-0160", "CVE-2017-12635_6",
    "CWE-89-SQL-injection", "EPS_CWE-434", "Juice-Shop", "PHP_CWE-434",
]

# Partisi relatif terhadap folder dalam skenario
PARTITIONS = [
    "training",
    "validation",
    "test/normal",
    "test/normal_and_attack",
]

KEEP_EXT = {".sc", ".json"}


def find_inner_dir(scenario_dir: Path) -> Path | None:
    """LID-DS membungkus tiap skenario dua kali: <scn>/<scn>/. Cari folder
    dalam yang berisi partisi, abaikan __MACOSX."""
    candidates = [
        d for d in scenario_dir.iterdir()
        if d.is_dir() and d.name != "__MACOSX"
    ]
    for c in candidates:
        if (c / "training").exists() or (c / "test").exists():
            return c
    # mungkin partisi langsung di scenario_dir
    if (scenario_dir / "training").exists() or (scenario_dir / "test").exists():
        return scenario_dir
    return None


def extract_recording(zip_path: Path, out_dir: Path) -> bool:
    """Ekstrak .sc + .json dari satu zip. Return True jika ada file diekstrak."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted_any = False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                ext = Path(member).suffix.lower()
                if ext in KEEP_EXT:
                    target = out_dir / Path(member).name
                    if target.exists() and target.stat().st_size > 0:
                        extracted_any = True
                        continue  # idempotent
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted_any = True
    except zipfile.BadZipFile:
        print(f"    [WARN] bad zip: {zip_path.name}")
        return False
    return extracted_any


def process_scenario(scenario: str) -> dict:
    scenario_dir = SOURCE_ROOT / scenario
    if not scenario_dir.exists():
        print(f"  [SKIP] {scenario}: folder tidak ditemukan")
        return {}
    inner = find_inner_dir(scenario_dir)
    if inner is None:
        print(f"  [SKIP] {scenario}: struktur partisi tidak dikenali")
        return {}

    counts = {}
    for part in PARTITIONS:
        src_part = inner / part
        if not src_part.exists():
            counts[part] = 0
            continue
        zips = sorted(src_part.glob("*.zip"))
        out_part = OUTPUT_ROOT / scenario / part
        n_ok = 0
        for zp in zips:
            rec_out = out_part / zp.stem  # satu folder per recording
            if extract_recording(zp, rec_out):
                n_ok += 1
        counts[part] = n_ok
        print(f"    {part:28s}: {n_ok}/{len(zips)} recording")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="ekstrak seluruh 7 skenario")
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help="daftar skenario spesifik")
    args = ap.parse_args()

    if args.scenarios:
        scenarios = args.scenarios
    elif args.all:
        scenarios = ALL_SCENARIOS
    else:
        scenarios = DEFAULT_SCENARIOS

    print("=" * 64)
    print("LID-DS Step 0: Extraction & Caching")
    print(f"Skenario: {scenarios}")
    print(f"Output:   {OUTPUT_ROOT}/")
    print("=" * 64)

    total = {}
    for scn in scenarios:
        print(f"\n[{scn}]")
        c = process_scenario(scn)
        total[scn] = c

    print("\n" + "=" * 64)
    print("Ringkasan ekstraksi:")
    grand = 0
    for scn, c in total.items():
        s = sum(c.values())
        grand += s
        print(f"  {scn:24s}: {s} recording  {c}")
    print(f"\n  TOTAL: {grand} recording diekstrak ke {OUTPUT_ROOT}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
