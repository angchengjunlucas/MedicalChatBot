from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    path = Path("medical-multiagent-chatbot/data/chroma")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"Reset Chroma folder: {path}")


if __name__ == "__main__":
    main()
