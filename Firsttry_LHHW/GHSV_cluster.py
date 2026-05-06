from pathlib import Path
import runpy


# Compatibility entry point for the reproducible 12000 GHSV baseline.
# The kinetic model and output workflow live in 12000GHSV.py.
if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name("12000GHSV.py"), run_name="__main__")
