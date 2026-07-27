"""Build a synthetic test archive: 24 3-band uint16 GeoTIFF chips,
windfarm/ (5) + background/ (19) -> deliberate ~21/79 class imbalance.

Usage: python scripts/make_synthetic_zip.py [out.zip]
"""

import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tifffile


def make_synthetic_zip(out: Path) -> Path:
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        specs = [("windfarm", 5), ("background", 19)]
        files = []
        for cls, count in specs:
            d = tmp / cls
            d.mkdir()
            for i in range(count):
                # S2-like DN values: mean ~1500, uint16
                chip = rng.normal(1500, 400, size=(64, 64, 3)).clip(0, 8000).astype(np.uint16)
                if cls == "windfarm":
                    chip[20:44, 30:34] = 6000  # bright "turbine" streak
                p = d / f"chip_{i:03d}.tif"
                tifffile.imwrite(p, chip)
                files.append(p)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in files:
                zf.write(p, p.relative_to(tmp))
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/synthetic_windfarm.zip")
    print(make_synthetic_zip(dest))
