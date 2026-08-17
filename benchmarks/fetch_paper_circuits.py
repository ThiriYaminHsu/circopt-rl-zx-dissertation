"""Download the paper's benchmark circuits for evaluation.

Migrated from Section 11 of the dissertation notebook. Pulls the Original/ QASM/qc
files from:
https://github.com/qilimanjaro-tech/Circopt-RL-ZXCalc/tree/main/rl-zx/results/circuits/Original

Downloaded files land in benchmarks/Original/ (gitignored -- re-run this script
rather than committing the circuit files). Run directly to (re-)download; import
`load_all` from evaluation/plotting code to load whatever is already on disk.
"""

import urllib.request
from pathlib import Path
from typing import Dict

import pyzx as zx

PAPER_BENCHMARK_BASE_URL = "https://raw.githubusercontent.com/qilimanjaro-tech/Circopt-RL-ZXCalc/main/rl-zx/results/circuits/Original/"

PAPER_BENCHMARK_NAMES = [
    "Adder8", "QFT16_before", "QFT32_before", "QFT8_before", "QFTAdd8", "adder_8",
    "barenco_tof_10", "barenco_tof_3", "barenco_tof_4", "barenco_tof_5",
    "csla_mux_3_original", "csum_mux_9_corrected", "gf2^4_mult", "gf2^5_mult",
    "gf2^6_mult", "gf2^7_mult", "gf2^8_mult", "grover_5.qc", "ham15-low.qc",
    "hwb6.qc", "mod5_4", "mod_mult_55", "mod_red_21", "qcla_adder_10", "qcla_com_7",
    "qcla_mod_7", "qft_4.qasm", "rc_adder_6", "tof_10", "tof_3.qasm", "tof_4",
    "tof_5", "vbe_adder_3",
]

LOCAL_DIR = Path(__file__).parent / "Original"


def download_all(dest_dir: Path = LOCAL_DIR) -> Dict[str, Path]:
    """Download every benchmark circuit to `dest_dir`, skipping any already present.
    Returns {name: local_path} for the ones that succeeded."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name in PAPER_BENCHMARK_NAMES:
        local_path = dest_dir / name
        if local_path.exists():
            saved[name] = local_path
            continue
        try:
            text = urllib.request.urlopen(PAPER_BENCHMARK_BASE_URL + name).read().decode("utf-8")
            local_path.write_text(text)
            saved[name] = local_path
        except Exception as e:
            print("failed to download", name, ":", e)
    return saved


def load_all(dest_dir: Path = LOCAL_DIR, download_missing: bool = True) -> Dict[str, "zx.Circuit"]:
    """Load every benchmark circuit that is present in `dest_dir` (downloading
    first if `download_missing`). Returns {f"paper_{name}": zx.Circuit}."""
    if download_missing:
        download_all(dest_dir)

    circuits = {}
    for name in PAPER_BENCHMARK_NAMES:
        local_path = dest_dir / name
        if not local_path.exists():
            continue
        try:
            circuits[f"paper_{name}"] = zx.Circuit.load(str(local_path))
        except Exception as e:
            print("failed to load", name, ":", e)
    return circuits


if __name__ == "__main__":
    saved = download_all()
    print(f"Downloaded/found {len(saved)}/{len(PAPER_BENCHMARK_NAMES)} paper benchmark circuits in {LOCAL_DIR}")
