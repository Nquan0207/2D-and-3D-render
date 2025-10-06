from __future__ import annotations

import sys

from .viewer_app import run_app


def main() -> None:
    try:
        run_app()
    except Exception as exc:
        print(f"Application encountered an error: {exc}")


if __name__ == "__main__":
    main()
