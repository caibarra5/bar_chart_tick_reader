#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys


def run_step(script_name: str) -> None:
    script_path = Path(__file__).resolve().parent / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")

    print(f"\n[full_pipeline] Running: {script_name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"Step failed: {script_name} (exit code {result.returncode})")


def main() -> None:
    # Run PNG generation first, then the AIF pipeline.
    run_step("two_bar_png_file_generation.py")
    run_step("run.py")
    print("\n[full_pipeline] Completed successfully.")


if __name__ == "__main__":
    main()
