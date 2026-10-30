import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FX = ROOT / "fixtures"


def make_clip(path, secs=2.0, w=360, h=640, audio=False, pattern="testsrc"):
    """A dummy clip from an ffmpeg test pattern (no AI generation)."""
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"{pattern}=size={w}x{h}:rate=24:duration={secs}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={secs}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac", "-shortest"] if audio else []) + [str(path)]
    subprocess.run(cmd, check=True)
    return Path(path)


@pytest.fixture(scope="session")
def fixtures():
    """The real failure takes (gitignored, local only). Skips when they are not on this machine."""
    if not (FX / "fixtures.json").exists():
        pytest.skip("real fixtures not present (fixtures/ is local only)")
    import shotguard_core as sg
    index = json.load(open(FX / "fixtures.json"))
    return {k: (str(FX / v["file"]), sg.spec.load(str(FX / "specs" / f"{k}.json"), base_dir=str(FX)), v["must_flag"])
            for k, v in index.items()}


@pytest.fixture(scope="session")
def comfy_path():
    p = Path(os.environ.get("COMFYUI_DIR", Path.home() / "ComfyUI"))
    if not (p / "comfy_api").exists():
        pytest.skip("ComfyUI not found (set COMFYUI_DIR)")
    sys.path.insert(0, str(p))
    return p
