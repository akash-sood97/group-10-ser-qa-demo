"""Build the standalone repo folder for the QA demo.

Copies this folder plus ``shared/audio_utils.py`` into
``~/Downloads/group-10-ser-qa-demo/`` and writes ``.gitignore`` and a
repo ``README.md``. Run it again to rebuild; it refuses to touch a
target that has a git remote configured.

Run:  python make_repo.py [--target PATH]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED_FILE = HERE.parent / "shared" / "audio_utils.py"
DEFAULT_TARGET = Path.home() / "Downloads" / "group-10-ser-qa-demo"
GITIGNORE = "__pycache__/\n.venv/\n*.pyc\n.DS_Store\n"
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def has_remote(target: Path) -> bool:
    if not (target / ".git").exists():
        return False
    out = subprocess.run(["git", "-C", str(target), "remote"],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    target = ap.parse_args().target.expanduser().resolve()

    if not SHARED_FILE.exists():
        sys.exit(f"Missing {SHARED_FILE}")
    if has_remote(target):
        sys.exit(f"{target} has a git remote; refusing to overwrite it.")

    target.mkdir(parents=True, exist_ok=True)
    for item in target.iterdir():          # clear everything but .git
        if item.name == ".git":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()

    for item in HERE.iterdir():
        if item.name in {"__pycache__", ".DS_Store"}:
            continue
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest, ignore=SKIP)
        else:
            shutil.copy2(item, dest)

    (target / "shared").mkdir()
    shutil.copy2(SHARED_FILE, target / "shared" / "audio_utils.py")
    (target / ".gitignore").write_text(GITIGNORE)
    shutil.copy2(HERE / "README.md", target / "README.md")
    print(f"Repo folder ready: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
