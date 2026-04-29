"""SPMiner_learn source package root.

This module applies process-level runtime guards that must run before heavy
numeric libraries (for example, PyTorch + MKL users on Windows).
"""

from __future__ import annotations

import os
import platform

__all__: list[str] = []


def _configure_openmp_runtime() -> None:
    """Avoid OpenMP duplicate-runtime aborts on Windows.

    In some Conda stacks, importing packages like PyTorch, NumPy/Scipy, and
    related dependencies can load multiple OpenMP runtimes and trigger
    ``OMP: Error #15``. We default to a compatibility mode so CLI entry points
    launched via ``python -m src...`` do not crash at shutdown.

    Set ``SPMINER_STRICT_OPENMP=1`` to disable this workaround.
    """

    if platform.system().lower() != "windows":
        return
    if os.environ.get("SPMINER_STRICT_OPENMP", "0") == "1":
        return
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


_configure_openmp_runtime()


