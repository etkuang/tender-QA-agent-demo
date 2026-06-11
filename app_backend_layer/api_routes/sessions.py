# coding: utf-8
# @Author: Wang Qingkang

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from common.logger import get_logger
from app_backend_layer.history.history_db import HistoryManager
from app_backend_layer.schemas import SessionMeta, StoredMessage

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = get_logger("backend.sessions")


def get_db(request: Request) -> HistoryManager:
    return request.app.state.db


@router.get("/get_list", response_model=list[SessionMeta])
async def get_history_list(db: HistoryManager = Depends(get_db)):
    try:
        return await db.get_all_metadata()
    except Exception:
        logger.exception("sessions.get_list failed")
        raise HTTPException(status_code=500, detail="Database query failure.")


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, db: HistoryManager = Depends(get_db)):
    try:
        await db.delete_session(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception:
        logger.exception("sessions.delete failed | session_id=%s", session_id)
        raise HTTPException(status_code=500, detail="Database deletion failure.")


@router.get("/{session_id}/messages", response_model=list[StoredMessage])
async def get_session_messages(session_id: str, db: HistoryManager = Depends(get_db)):
    try:
        return await db.load_messages(session_id)
    except Exception:
        logger.exception("sessions.messages failed | session_id=%s", session_id)
        raise HTTPException(status_code=500, detail="Database retrieval failure.")
