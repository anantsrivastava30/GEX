"""Tradier credential loading for headless `quant_analysis`.

This module replaces the historical pattern of pulling
``TRADIER_TOKEN`` from ``streamlit.secrets`` (see ``app.py``). It reads
credentials from ``.env`` files via ``python-dotenv`` and from the
process environment, with **no Streamlit dependency**.

Supported environment-variable schemas (resolved in this order):

1. **OpenClaw schema** (preferred — matches ``cli_trader/.env``)::

       TRADIER_ENV=sandbox           # or "production"
       SANDBOX_TRADIER_API_KEY=...
       SANDBOX_TRADIER_ACCOUNT_ID=...
       PRODUCTION_TRADIER_API_KEY=...
       PRODUCTION_TRADIER_ACCOUNT_ID=...

   ``get_tradier_credentials()`` returns the key/account/base-url tuple
   matching ``TRADIER_ENV``.

2. **Legacy quant schema** (fallback — matches the Streamlit dashboard)::

       TRADIER_TOKEN=...
       TRADIER_ACCOUNT_ID=...        # optional
       TRADIER_BASE_URL=...          # optional

   Used only when the OpenClaw schema is not present. The base URL
   defaults to the production endpoint, matching the behaviour of the
   historical Streamlit code path.

``.env`` discovery order (first hit wins, but later files do **not**
override variables already present in ``os.environ``):

1. ``$QUANT_ANALYSIS_DOTENV`` if set.
2. ``./.env`` resolved from the current working directory.
3. ``~/asriva20/quant/.env`` (legacy quant location).
4. ``~/asriva20/cli_trader/.env`` (OpenClaw location).

Per ``AGENTS.md`` §0.8 the token is **never** logged or echoed; the
``mask`` helper is provided for diagnostic output.

Reference: ``docs/openclaw/13_SPRINT_PLAN.md`` §1.1;
``docs/openclaw/03_EXISTING_CODEBASE_MAP.md`` Sprint 1.1 deliverable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv


PRODUCTION_BASE_URL = "https://api.tradier.com/v1"
SANDBOX_BASE_URL = "https://sandbox.tradier.com/v1"


class TradierCredentialsError(RuntimeError):
    """Raised when no usable Tradier credentials can be located."""


@dataclass(frozen=True)
class TradierCredentials:
    """Resolved Tradier credentials.

    ``environment`` is one of ``"sandbox"`` / ``"production"`` /
    ``"legacy"``. ``"legacy"`` indicates the old single-token quant
    schema where the environment is implicit in the chosen base URL.
    """

    api_key: str
    base_url: str
    environment: str
    account_id: Optional[str] = None

    def headers(self) -> dict:
        """Auth headers ready for ``requests``."""

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def masked(self) -> str:
        """Diagnostic representation that never reveals the token."""

        return (
            f"TradierCredentials(env={self.environment}, "
            f"base_url={self.base_url}, "
            f"api_key={mask(self.api_key)}, "
            f"account_id={self.account_id or '<unset>'})"
        )


def _candidate_dotenv_paths() -> Iterable[Path]:
    explicit = os.environ.get("QUANT_ANALYSIS_DOTENV")
    if explicit:
        yield Path(explicit).expanduser()

    yield Path.cwd() / ".env"
    yield Path.home() / "asriva20" / "quant" / ".env"
    yield Path.home() / "asriva20" / "cli_trader" / ".env"


def load_dotenv_files() -> list[Path]:
    """Load all candidate ``.env`` files. Returns the list actually used.

    ``override`` is ``False`` so that already-set environment variables
    (typically set by the operator's shell) take precedence over any
    file. This matches the convention used elsewhere in OpenClaw.
    """

    loaded: list[Path] = []
    seen: set[Path] = set()
    for path in _candidate_dotenv_paths():
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            load_dotenv(resolved, override=False)
            loaded.append(resolved)
    return loaded


def mask(secret: Optional[str]) -> str:
    """Return a masked representation of a secret for log output.

    Shows only the last 4 characters preceded by ``****``. Empty / None
    secrets render as ``<unset>`` and never expose anything.
    """

    if not secret:
        return "<unset>"
    if len(secret) <= 4:
        return "****"
    return f"****{secret[-4:]}"


def _resolve_environment(explicit: Optional[str]) -> Optional[str]:
    """Normalise an environment hint to ``sandbox`` or ``production``."""

    if explicit is None:
        explicit = os.environ.get("TRADIER_ENV")
    if not explicit:
        return None
    value = explicit.strip().lower()
    if value in {"sandbox", "sand", "paper"}:
        return "sandbox"
    if value in {"production", "prod", "live"}:
        return "production"
    raise TradierCredentialsError(
        f"Unrecognised TRADIER_ENV value: {explicit!r}. "
        "Expected one of: sandbox, production."
    )


def get_tradier_credentials(
    environment: Optional[str] = None,
    *,
    load_files: bool = True,
) -> TradierCredentials:
    """Resolve Tradier credentials.

    Parameters
    ----------
    environment:
        Override for ``TRADIER_ENV``. One of ``"sandbox"`` /
        ``"production"``. When ``None``, the value of ``TRADIER_ENV`` in
        the environment is used. When that is also unset, the OpenClaw
        schema requires an explicit choice and falls through to the
        legacy schema; the legacy schema defaults to production.
    load_files:
        When ``True`` (the default) call :func:`load_dotenv_files` first.
        Set to ``False`` in tests to keep the call pure.

    Returns
    -------
    TradierCredentials
        Resolved credentials. The token is never logged.

    Raises
    ------
    TradierCredentialsError
        When no schema yields a usable token.
    """

    if load_files:
        load_dotenv_files()

    env_choice = _resolve_environment(environment)

    # ---- OpenClaw schema --------------------------------------------------
    sandbox_key = os.environ.get("SANDBOX_TRADIER_API_KEY")
    production_key = os.environ.get("PRODUCTION_TRADIER_API_KEY")
    sandbox_account = os.environ.get("SANDBOX_TRADIER_ACCOUNT_ID")
    production_account = os.environ.get("PRODUCTION_TRADIER_ACCOUNT_ID")

    has_openclaw_schema = bool(sandbox_key or production_key)
    if has_openclaw_schema:
        # If env_choice is None but only one schema is populated, pick it.
        if env_choice is None:
            if sandbox_key and not production_key:
                env_choice = "sandbox"
            elif production_key and not sandbox_key:
                env_choice = "production"
            else:
                raise TradierCredentialsError(
                    "Both SANDBOX and PRODUCTION Tradier keys are set but "
                    "TRADIER_ENV is unset. Set TRADIER_ENV to 'sandbox' or "
                    "'production' (or pass --env to the CLI)."
                )

        if env_choice == "sandbox":
            if not sandbox_key:
                raise TradierCredentialsError(
                    "TRADIER_ENV=sandbox but SANDBOX_TRADIER_API_KEY is not set."
                )
            return TradierCredentials(
                api_key=sandbox_key,
                base_url=SANDBOX_BASE_URL,
                environment="sandbox",
                account_id=sandbox_account,
            )

        if env_choice == "production":
            if not production_key:
                raise TradierCredentialsError(
                    "TRADIER_ENV=production but PRODUCTION_TRADIER_API_KEY is not set."
                )
            return TradierCredentials(
                api_key=production_key,
                base_url=PRODUCTION_BASE_URL,
                environment="production",
                account_id=production_account,
            )

    # ---- Legacy quant schema ---------------------------------------------
    legacy_token = os.environ.get("TRADIER_TOKEN")
    if legacy_token:
        legacy_base = os.environ.get("TRADIER_BASE_URL")
        if legacy_base:
            base_url = legacy_base
        elif env_choice == "sandbox":
            base_url = SANDBOX_BASE_URL
        else:
            base_url = PRODUCTION_BASE_URL
        return TradierCredentials(
            api_key=legacy_token,
            base_url=base_url,
            environment=env_choice or "legacy",
            account_id=os.environ.get("TRADIER_ACCOUNT_ID"),
        )

    raise TradierCredentialsError(
        "No Tradier credentials found. Set one of:\n"
        "  - SANDBOX_TRADIER_API_KEY / PRODUCTION_TRADIER_API_KEY "
        "(plus TRADIER_ENV), or\n"
        "  - TRADIER_TOKEN (legacy single-token schema).\n"
        "Either export them or place them in one of the .env files "
        "searched by quant_analysis.credentials.load_dotenv_files()."
    )
