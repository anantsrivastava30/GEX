"""Quant analysis application package.

This package is dual-mode:

- **Streamlit mode** (``app.py`` / ``ui/dashboard.py``): real Streamlit is
  installed; everything works as before.
- **Headless mode** (``cli_trader`` / OpenClaw, scripts, notebooks): real
  Streamlit is not installed. To keep modules that do
  ``import streamlit as st`` at top level importable, we install a
  no-op shim into ``sys.modules`` *before* any submodule is imported.

The shim is idempotent and a no-op when real Streamlit is present, so
existing UI code is unaffected. See ``_st_shim.py`` for the contract.
"""

from . import _st_shim as _st_shim  # noqa: F401

_st_shim.install()
