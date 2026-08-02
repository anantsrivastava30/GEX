"""Shared-workspace watchlist endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend.app.schemas import Watchlist, WatchlistListResponse, WatchlistMutation
from backend.app.services_watchlists import (
    create_watchlist,
    delete_watchlist,
    list_watchlists,
    update_watchlist,
)
from backend.app.security import require_workspace_pin

router = APIRouter(prefix="/api/watchlists", tags=["watchlists"])


@router.get("", response_model=WatchlistListResponse)
def watchlists(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return list_watchlists()


@router.post("", response_model=Watchlist, status_code=status.HTTP_201_CREATED)
def add_watchlist(
    request: WatchlistMutation,
    response: Response,
    _: None = Depends(require_workspace_pin),
):
    response.headers["Cache-Control"] = "no-store"
    try:
        return create_watchlist(request.name, request.symbols)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.put("/{watchlist_id}", response_model=Watchlist)
def edit_watchlist(
    watchlist_id: int,
    request: WatchlistMutation,
    response: Response,
    _: None = Depends(require_workspace_pin),
):
    response.headers["Cache-Control"] = "no-store"
    try:
        return update_watchlist(watchlist_id, request.name, request.symbols)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Watchlist not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/{watchlist_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_watchlist(
    watchlist_id: int, _: None = Depends(require_workspace_pin)
):
    try:
        delete_watchlist(watchlist_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Watchlist not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
