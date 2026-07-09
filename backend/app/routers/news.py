from typing import List

from fastapi import APIRouter

from backend.app import services
from backend.app.schemas import NewsItem

router = APIRouter(prefix="/api/news", tags=["news"])


@router.get("", response_model=List[NewsItem])
def news():
    return services.get_news()
