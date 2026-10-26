import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from cropwarn.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
