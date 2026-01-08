from __future__ import annotations

import os
import sys

# Support running both as a module (python -m simpleviewer)
# and directly as a script (python simpleviewer/main.py)
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from simpleviewer.viewer_app import run_app
else:
    from .viewer_app import run_app


def main() -> None:
    try:
        run_app()
    except Exception as exc:
        print(f"Application encountered an error: {exc}")


if __name__ == "__main__":
    main()
