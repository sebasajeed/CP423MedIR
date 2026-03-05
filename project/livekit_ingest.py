import base64
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

try:
    from .clinical_ir import ClinicalIRSystem
except ImportError:
    from clinical_ir import ClinicalIRSystem


class SegmentPayload(BaseModel):
    content: str = Field(..., min_length=1)
    speaker_role: str = Field(..., min_length=1)
    session_id: str | None = None
    participant_id: str | None = None
    start: float | None = None
    end: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioPayload(BaseModel):
    audio_b64: str = Field(..., min_length=1)
    speaker_role: str = Field(..., min_length=1)
    filename: str = "livekit_chunk.wav"
    session_id: str | None = None
    participant_id: str | None = None
    start_offset: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


load_dotenv()
app = FastAPI(title="Clinical IR LiveKit Ingest Service")
bot: ClinicalIRSystem | None = None
ingest_token = os.getenv("LIVEKIT_INGEST_TOKEN", "").strip()


def _authorize(x_api_key: str | None) -> None:
    if ingest_token and x_api_key != ingest_token:
        raise HTTPException(status_code=401, detail="Unauthorized ingest request")


def _get_bot() -> ClinicalIRSystem:
    global bot
    if bot is None:
        bot = ClinicalIRSystem()
    return bot


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/livekit/segment")
def ingest_segment(payload: SegmentPayload, x_api_key: str | None = Header(default=None)) -> dict:
    _authorize(x_api_key)
    runtime_bot = _get_bot()

    role = payload.speaker_role.strip().upper()
    md = dict(payload.metadata or {})
    if payload.session_id:
        md["session_id"] = payload.session_id
    if payload.participant_id:
        md["participant_id"] = payload.participant_id
    if payload.start is not None:
        md["start"] = payload.start
    if payload.end is not None:
        md["end"] = payload.end
    md["source"] = "livekit"

    record = runtime_bot.index_segment(content=payload.content, speaker_role=role, metadata=md)
    return {"status": "indexed", "record_id": record.get("id"), "speaker_role": role}


@app.post("/livekit/audio")
def ingest_audio(payload: AudioPayload, x_api_key: str | None = Header(default=None)) -> dict:
    _authorize(x_api_key)
    runtime_bot = _get_bot()

    try:
        audio_bytes = base64.b64decode(payload.audio_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio payload: {exc}") from exc

    role = payload.speaker_role.strip().upper()
    transcribed_segments = runtime_bot.transcribe_audio_bytes(audio_bytes, filename=payload.filename)

    indexed_count = 0
    for seg in transcribed_segments:
        md = dict(payload.metadata or {})
        md["source"] = "livekit"
        if payload.session_id:
            md["session_id"] = payload.session_id
        if payload.participant_id:
            md["participant_id"] = payload.participant_id
        md["start"] = payload.start_offset + float(seg["start"])
        md["end"] = payload.start_offset + float(seg["end"])

        runtime_bot.index_segment(content=seg["text"], speaker_role=role, metadata=md)
        indexed_count += 1

    return {
        "status": "indexed",
        "speaker_role": role,
        "segments_indexed": indexed_count,
    }
