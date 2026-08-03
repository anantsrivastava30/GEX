"""Snapshot-backed alert rules, in-app events, and optional delivery."""

from __future__ import annotations

import hashlib
import json
import logging
import smtplib
from email.message import EmailMessage
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

from backend.app import storage
from backend.app.config import get_settings
from backend.app.schemas import AlertRuleMutation, CustomScreenerRequest
from backend.app.services_screener import run_custom_screener
from quant_analysis.storage.snapshots import last_trading_date

logger = logging.getLogger(__name__)
_MAX_RULES = 50
_MAX_EVENTS_PER_RULE = 25
_DISCORD_HOSTS = {"discord.com", "discordapp.com"}


def _email_configured() -> bool:
    settings = get_settings()
    return bool(settings.smtp_host and settings.smtp_from and settings.alert_email_to)


def get_alert_status() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "workspace": "shared",
        "scheduler_enabled": settings.scheduler_enabled,
        "pin_required": bool(settings.workspace_pin or settings.ai_pin),
        "discord_configured": bool(settings.alert_discord_webhook_url),
        "email_configured": _email_configured(),
        "evaluation_interval_minutes": 30,
    }


def _validated_rule(request: AlertRuleMutation) -> Dict[str, Any]:
    if request.watchlist_id is not None and not storage.get_watchlist(
        request.watchlist_id
    ):
        raise ValueError("Watchlist not found")
    if request.notify_discord and not get_settings().alert_discord_webhook_url:
        raise ValueError("Discord notifications are not configured on the server")
    if request.notify_email and not _email_configured():
        raise ValueError("Email notifications are not configured on the server")

    query = request.query.model_copy(deep=True)
    query.symbols = None
    if not query.conditions:
        raise ValueError("An alert rule requires at least one condition")
    # Validation stays in the same whitelist/filter implementation used by
    # evaluation. It performs persisted reads only, never a live chain scan.
    run_custom_screener(query)
    return {
        "name": request.name.strip(),
        "enabled": request.enabled,
        "watchlist_id": request.watchlist_id,
        "query": query.model_dump(mode="json"),
        "notify_discord": request.notify_discord,
        "notify_email": request.notify_email,
    }


def list_alert_rules() -> List[Dict[str, Any]]:
    return storage.list_alert_rules()


def create_alert_rule(request: AlertRuleMutation) -> Dict[str, Any]:
    if len(storage.list_alert_rules()) >= _MAX_RULES:
        raise ValueError(f"A maximum of {_MAX_RULES} alert rules is allowed")
    return storage.create_alert_rule(_validated_rule(request))


def update_alert_rule(rule_id: int, request: AlertRuleMutation) -> Dict[str, Any]:
    rule = storage.update_alert_rule(rule_id, _validated_rule(request))
    if rule is None:
        raise KeyError(rule_id)
    return rule


def delete_alert_rule(rule_id: int) -> None:
    if storage.alert_rule_has_events(rule_id):
        raise RuntimeError("Disable this rule instead; its retained event history prevents deletion")
    if not storage.delete_alert_rule(rule_id):
        raise KeyError(rule_id)


def _event_message(rule: Dict[str, Any], row: Dict[str, Any]) -> str:
    values = row["values"]
    if row["scope"] == "contract":
        return (
            f"{values.get('expiration_date', 'unknown expiry')} "
            f"{values.get('strike', 'unknown strike')} "
            f"{str(values.get('option_type', 'option')).upper()}: "
            f"Vol/OI {values.get('volume_oi', 'n/a')}, volume "
            f"{values.get('volume', 'n/a')}, OI {values.get('open_interest', 'n/a')}."
        )
    for window in (50, 200):
        cross = values.get(f"sma_{window}_cross")
        if cross in {"above", "below"}:
            return (
                f"Spot {values.get('spot', 'n/a')} crossed {cross} the "
                f"{window}-day SMA at {values.get(f'sma_{window}', 'n/a')}."
            )
    return (
        f"Gamma-gap score {values.get('gamma_gap_score', 'n/a')}, "
        f"magnet {values.get('gamma_magnet_strike', 'n/a')}, "
        f"net GEX {values.get('net_gex_total', 'n/a')}."
    )


def _insert_events(rule: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, Any]]:
    inserted = []
    event_limit = (
        get_settings().snapshot_max_symbols
        if result["scope"] == "ticker"
        else _MAX_EVENTS_PER_RULE
    )
    for row in result["rows"][:event_limit]:
        fingerprint_data = [
            rule["id"],
            rule["version"],
            row["snapshot_date"],
            row["row_key"],
        ]
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_data, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        title = f"{row['symbol']} matched {rule['name']}"
        event = {
            "alert_rule_id": rule["id"],
            "rule_version": rule["version"],
            "fingerprint": fingerprint,
            "row_key": row["row_key"],
            "snapshot_date": row["snapshot_date"],
            "symbol": row["symbol"],
            "scope": result["scope"],
            "title": title,
            "message": _event_message(rule, {**row, "scope": result["scope"]}),
            "payload": row,
            "discord_status": "pending" if rule["notify_discord"] else "not_requested",
            "email_status": "pending" if rule["notify_email"] else "not_requested",
        }
        event_id = storage.insert_alert_event(event)
        if event_id:
            inserted.append({"id": event_id, **event})
    return inserted


def _discord_digest(rule: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    settings = get_settings()
    webhook = settings.alert_discord_webhook_url or ""
    parsed = urlparse(webhook)
    if parsed.scheme != "https" or parsed.hostname not in _DISCORD_HOSTS:
        raise ValueError("Invalid Discord webhook configuration")
    lines = [f"**{rule['name']}** - {len(events)} snapshot match(es)"]
    lines.extend(f"- {event['title']}: {event['message']}" for event in events[:10])
    response = requests.post(
        webhook,
        json={"content": "\n".join(lines)[:1900], "allowed_mentions": {"parse": []}},
        timeout=10,
        allow_redirects=False,
    )
    response.raise_for_status()


def _email_digest(rule: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    settings = get_settings()
    sender = settings.smtp_from or ""
    recipient = settings.alert_email_to or ""
    if any(char in sender + recipient for char in "\r\n"):
        raise ValueError("Invalid email address configuration")

    message = EmailMessage()
    message["Subject"] = f"GEX alert: {rule['name']}"
    message["From"] = sender
    message["To"] = recipient
    body = [f"{len(events)} persisted snapshot match(es) for {rule['name']}:\n"]
    body.extend(f"- {event['title']}: {event['message']}" for event in events)
    message.set_content("\n".join(body))

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as client:
        if settings.smtp_starttls:
            client.starttls()
        if settings.smtp_username:
            client.login(settings.smtp_username, settings.smtp_password or "")
        client.send_message(message)


def _deliver_channel(
    rule: Dict[str, Any], events: List[Dict[str, Any]], channel: str
) -> None:
    event_ids = [event["id"] for event in events]
    if channel == "discord":
        try:
            _discord_digest(rule, events)
        except Exception as exc:
            logger.warning("Discord alert delivery failed: %s", type(exc).__name__)
            storage.update_delivery(event_ids, "discord", "failed", "Delivery failed")
        else:
            storage.update_delivery(event_ids, "discord", "sent")
    elif channel == "email":
        try:
            _email_digest(rule, events)
        except Exception as exc:
            logger.warning("Email alert delivery failed: %s", type(exc).__name__)
            storage.update_delivery(event_ids, "email", "failed", "Delivery failed")
        else:
            storage.update_delivery(event_ids, "email", "sent")


def _deliver(rule: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
    if rule["notify_discord"]:
        _deliver_channel(rule, events, "discord")
    if rule["notify_email"]:
        _deliver_channel(rule, events, "email")


def _retry_failed_deliveries() -> None:
    for channel in ("discord", "email"):
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for event in storage.list_pending_deliveries(channel):
            grouped.setdefault(event["alert_rule_id"], []).append(event)
        for rule_id, events in grouped.items():
            rule = storage.get_alert_rule(rule_id)
            if rule and rule.get(f"notify_{channel}"):
                _deliver_channel(rule, events, channel)
            else:
                storage.update_delivery(
                    [event["id"] for event in events], channel, "skipped"
                )


def evaluate_enabled_alerts() -> Dict[str, int]:
    """Evaluate enabled rules against today's latest persisted snapshots."""

    _retry_failed_deliveries()
    evaluated = 0
    emitted = 0
    current_date = last_trading_date()
    for rule in storage.list_alert_rules(enabled_only=True):
        try:
            query = CustomScreenerRequest.model_validate(rule["query"])
            if rule["watchlist_id"] is None:
                query.symbols = None
            else:
                watchlist = storage.get_watchlist(rule["watchlist_id"])
                if not watchlist:
                    storage.update_rule_evaluation(rule["id"], "Watchlist not found")
                    continue
                query.symbols = watchlist["symbols"]
            if query.scope == "ticker":
                query.limit = min(
                    max(query.limit, get_settings().snapshot_max_symbols), 100
                )
            else:
                query.limit = min(query.limit, 100)
            result = run_custom_screener(query)
        except ValueError as exc:
            storage.update_rule_evaluation(rule["id"], f"Invalid rule: {exc}")
            continue
        except Exception:
            logger.exception("Alert evaluation failed for rule %s", rule["id"])
            storage.update_rule_evaluation(rule["id"], "Evaluation failed")
            continue

        evaluated += 1
        if result["stale"] or result["as_of"] != current_date:
            storage.update_rule_evaluation(
                rule["id"], "Skipped: snapshots are stale or incomplete"
            )
            continue
        events = _insert_events(rule, result)
        if events:
            _deliver(rule, events)
            emitted += len(events)
        storage.update_rule_evaluation(rule["id"], None)
    return {"evaluated": evaluated, "emitted": emitted}


def list_alert_events(unread_only: bool, limit: int) -> Dict[str, Any]:
    return storage.list_alert_events(unread_only, limit)


def mark_event_read(event_id: int) -> None:
    if not storage.mark_alert_event_read(event_id):
        raise KeyError(event_id)


def mark_all_events_read() -> int:
    return storage.mark_all_alert_events_read()
