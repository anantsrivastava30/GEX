"""Streamlit shim for headless use.

Several modules in :mod:`quant_analysis` (notably ``services.market_data``,
``services.ai_analysis`` and ``storage.db``) do ``import streamlit as st``
at module top level. When this package is imported from a process that
does **not** have the ``streamlit`` package installed (for example the
``cli_trader`` virtualenv used by OpenClaw), those imports fail and the
whole package becomes unimportable, which defeats Sprint 1.1's goal of
making ``quant_analysis`` usable as a headless library.

This module provides a no-op shim. When ``import streamlit`` fails, we
synthesise a lightweight stand-in module and register it under
``sys.modules['streamlit']`` so that any subsequent
``import streamlit as st`` finds the shim instead of raising
``ModuleNotFoundError``.

Only the names actually used by the importable core are stubbed:

- ``st.cache_data`` / ``st.cache_resource`` — decorator factories that
  return the wrapped function unchanged.
- ``st.secrets`` — a dict-like object that resolves to environment
  variables (so ``st.secrets["TRADIER_TOKEN"]`` keeps working when the
  variable is in ``.env``).
- ``st.error`` / ``st.warning`` / ``st.info`` / ``st.write`` — no-ops
  that route to the standard ``logging`` module so the messages are not
  silently lost.
- ``st.session_state`` — empty dict-like store (rarely touched in the
  importable core, but cheap to provide).

Anything not stubbed will raise ``AttributeError`` on access, which is
the correct behaviour: it tells us a previously-untouched UI hook has
crept into the headless path and should be moved or shimmed
deliberately.

The shim does **nothing** when real Streamlit is installed; ``app.py``
keeps working unchanged.

Reference: ``docs/openclaw/03_EXISTING_CODEBASE_MAP.md`` Sprint 1.1
deliverable; ``docs/openclaw/13_SPRINT_PLAN.md`` 1.1.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from typing import Any, Callable, Mapping


_LOGGER = logging.getLogger("quant_analysis._st_shim")


class _SecretsShim(Mapping):
    """Dict-like accessor that falls back to ``os.environ``.

    Streamlit's real ``st.secrets`` raises ``StreamlitSecretNotFoundError``
    on missing keys; we mirror the surface area used in the codebase
    (``__getitem__``, ``get``, ``in``) and route lookups to environment
    variables. ``credentials.py`` is the canonical source for the
    OpenClaw side; this is the back-compat door for legacy callers that
    still write ``st.secrets["TRADIER_TOKEN"]``.
    """

    def __getitem__(self, key: str) -> str:
        value = os.environ.get(key)
        if value is None:
            raise KeyError(
                f"st.secrets[{key!r}] is not available in headless mode "
                f"and {key} is not set in the environment."
            )
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return os.environ.get(key, default)

    def __iter__(self):
        return iter(os.environ)

    def __len__(self) -> int:
        return len(os.environ)

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return isinstance(key, str) and key in os.environ


def _identity_decorator(*dargs: Any, **dkwargs: Any) -> Callable[..., Any]:
    """Stand-in for ``@st.cache_data`` / ``@st.cache_resource``.

    Streamlit's cache decorators can be used either as bare decorators
    (``@st.cache_data``) or with arguments (``@st.cache_data(ttl=60)``).
    This implementation supports both forms and returns the function
    unchanged. No caching is performed in headless mode; callers that
    need caching should add an explicit on-disk cache.
    """

    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        # Used as bare decorator: @st.cache_data
        return dargs[0]

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return _decorator


def _make_stub() -> types.ModuleType:
    """Build the synthetic ``streamlit`` module."""

    module = types.ModuleType("streamlit")
    module.__doc__ = (
        "Headless no-op shim installed by quant_analysis._st_shim. "
        "Real Streamlit was not importable when quant_analysis was loaded."
    )

    # Caching decorators
    module.cache_data = _identity_decorator
    module.cache_resource = _identity_decorator

    # Secrets accessor
    module.secrets = _SecretsShim()

    # Session state: a plain dict is close enough for the importable core
    module.session_state = {}

    # Logging-style sinks. Route to standard logging so messages do not
    # vanish silently. We use lambdas that ignore any kwargs Streamlit
    # accepts (e.g. icon=, unsafe_allow_html=).
    module.error = lambda msg, *a, **k: _LOGGER.error(str(msg))
    module.warning = lambda msg, *a, **k: _LOGGER.warning(str(msg))
    module.info = lambda msg, *a, **k: _LOGGER.info(str(msg))
    module.success = lambda msg, *a, **k: _LOGGER.info(str(msg))
    module.write = lambda *a, **k: None
    module.markdown = lambda *a, **k: None
    module.caption = lambda *a, **k: None

    # Misc no-ops occasionally encountered at import time
    module.set_page_config = lambda *a, **k: None
    module.stop = lambda *a, **k: None

    # Marker so callers can detect they have the shim, not the real
    # package. Keeps debugging cheap.
    module.__is_shim__ = True

    return module


def install() -> bool:
    """Install the shim if real Streamlit is unavailable.

    Returns ``True`` if a shim was installed in this call, ``False`` if
    real Streamlit was already importable (or if a shim was already
    installed by a previous call).
    """

    if "streamlit" in sys.modules:
        # Either real Streamlit or a previously-installed shim. Leave
        # things alone.
        return False

    try:
        import streamlit  # noqa: F401  (real package present)
        return False
    except ImportError:
        pass

    sys.modules["streamlit"] = _make_stub()
    _LOGGER.debug("Installed Streamlit shim (headless mode).")
    return True


def is_active() -> bool:
    """``True`` iff a shim (not the real Streamlit) is in ``sys.modules``."""

    mod = sys.modules.get("streamlit")
    return bool(mod) and getattr(mod, "__is_shim__", False)
