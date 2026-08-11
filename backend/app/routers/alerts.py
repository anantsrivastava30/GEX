"""Shared-workspace alert rules and in-app event inbox."""

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from backend.app.schemas import (
    AlertEventReadRequest,
    AlertEventsResponse,
    AlertRule,
    AlertRuleMutation,
    AlertStatus,
)
from backend.app.security import require_workspace_pin
from backend.app.services_alerts import (
    create_alert_rule,
    delete_alert_rule,
    get_alert_status,
    list_alert_events,
    list_alert_rules,
    mark_all_events_read,
    mark_event_read,
    mark_events_read,
    update_alert_rule,
)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/status", response_model=AlertStatus)
def alert_status(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return get_alert_status()


@router.get("/rules", response_model=list[AlertRule])
def rules(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return list_alert_rules()


@router.post("/rules", response_model=AlertRule, status_code=status.HTTP_201_CREATED)
def add_rule(
    request: AlertRuleMutation, _: None = Depends(require_workspace_pin)
):
    try:
        return create_alert_rule(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.put("/rules/{rule_id}", response_model=AlertRule)
def edit_rule(
    rule_id: int,
    request: AlertRuleMutation,
    _: None = Depends(require_workspace_pin),
):
    try:
        return update_alert_rule(rule_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Alert rule not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_rule(rule_id: int, _: None = Depends(require_workspace_pin)):
    try:
        delete_alert_rule(rule_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Alert rule not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/events", response_model=AlertEventsResponse)
def events(
    response: Response,
    unread_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
):
    response.headers["Cache-Control"] = "no-store"
    return list_alert_events(unread_only, limit)


@router.post("/events/{event_id}/read", response_model=dict)
def read_event(event_id: int, _: None = Depends(require_workspace_pin)):
    try:
        mark_event_read(event_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Alert event not found") from exc
    return {"status": "ok"}


@router.post("/events/read", response_model=dict)
def read_events(
    request: AlertEventReadRequest, _: None = Depends(require_workspace_pin)
):
    """Mark a selected batch of events read in one call (thread triage)."""

    return {"updated": mark_events_read(request.ids)}


@router.post("/events/read-all", response_model=dict)
def read_all_events(_: None = Depends(require_workspace_pin)):
    return {"updated": mark_all_events_read()}
