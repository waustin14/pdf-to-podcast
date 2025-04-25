from fastapi import FastAPI, HTTPException, Response
from shared.api_types import ServiceType
from shared.job import JobStatusManager
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from kokoro import KPipeline
from pydub import AudioSegment
import io
import os
from shared.otel import OpenTelemetryInstrumentation, OpenTelemetryConfig
from opentelemetry.trace.status import StatusCode
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KOKORO_VOICES = {
    # American English – Female (af_)
    "af_heart": "Heart",
    "af_alloy": "Alloy",
    "af_aoede": "Aoede",
    "af_bella": "Bella",
    "af_jessica": "Jessica",
    "af_kore": "Kore",
    "af_nicole": "Nicole",
    "af_nova": "Nova",
    "af_river": "River",
    "af_sarah": "Sarah",
    "af_sky": "Sky",

    # American English – Male (am_)
    "am_adam": "Adam",
    "am_echo": "Echo",
    "am_eric": "Eric",
    "am_fenrir": "Fenrir",
    "am_liam": "Liam",
    "am_michael": "Michael",
    "am_onyx": "Onyx",
    "am_puck": "Puck",
    "am_santa": "Santa",

    # British English – Female (bf_)
    "bf_alice": "Alice",
    "bf_emma": "Emma",
    "bf_isabella": "Isabella",
    "bf_lily": "Lily",

    # British English – Male (bm_)
    "bm_daniel": "Daniel",
    "bm_fable": "Fable",
    "bm_george": "George",
    "bm_lewis": "Lewis"
}

class TTSProviderKokoro:
    def __init__(self):
        self.pipelineAmerican = KPipeline('a')  # American English
        self.pipelineBritish = KPipeline('b')  # British English

    def synthesize(self, text: str, voice: str) -> torch.Tensor:
        audio = torch.Tensor()
        if voice.startswith("af_") or voice.startswith("am_"):
            pipeline = self.pipelineAmerican
        elif voice.startswith("bf_") or voice.startswith("bm_"):
            pipeline = self.pipelineBritish
        else:
            raise ValueError(f"Unsupported voice: {voice}")
        for chunk in pipeline(text, voice=voice, speed=1, split_pattern=r'\n+'):
            audio = torch.cat((audio, chunk[2]))
        return audio

app = FastAPI(title="Kokoro TTS Provider", debug=True)
DEFAULT_VOICE_1 = os.getenv("DEFAULT_VOICE_1", "am_puck")
DEFAULT_VOICE_2 = os.getenv("DEFAULT_VOICE_2", "af_heart")
DEFAULT_VOICE_MAPPING = {"speaker-1": DEFAULT_VOICE_1, "speaker-2": DEFAULT_VOICE_2}

telemetry = OpenTelemetryInstrumentation()
config = OpenTelemetryConfig(
    service_name="tts-provider",
    otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://jaeger:4317"),
    enable_redis=True,
    enable_requests=True,
)
telemetry.initialize(config, app)

job_manager = JobStatusManager(ServiceType.TTS_PROVIDER, telemetry=telemetry)

# Mirrors OpenAI’s Speech v1 API request schema
class TTSProviderRequest(BaseModel):
    model: str = "kokoro-82m" # e.g. "kokoro-82m"
    input: str                # the text to speak
    voice: str = "af_heart"

# Mirrors the “data” array in OpenAI’s response
class AudioChunk(BaseModel):
    audio: str # the base64 encoded audio data

tts_provider = TTSProviderKokoro()

@app.post("/v1/audio/speech")
def synthesize(req: TTSProviderRequest):
    """Synthesize speech from text"""
    try:
        # Check if the voice is supported
        if req.voice not in KOKORO_VOICES:
            raise HTTPException(400, f"Unsupported voice: {req.voice}")
        # Check if the input text is empty
        if not req.input.strip():
            raise HTTPException(400, "Input text cannot be empty")
        # Check if the model is supported
        if req.model != "kokoro-82m":
            raise HTTPException(400, f"Unsupported model: {req.model}")
        audio = tts_provider.synthesize(req.input, req.voice)
        audio_pcm16 = (audio.numpy() * 32767).astype(np.int16)
        # Convert raw audio to MP3 format
        audio_segment = AudioSegment(
            audio_pcm16.tobytes(),
            sample_width=2,
            frame_rate=24000,
            channels=1
        )
        audio_buffer = io.BytesIO()
        audio_segment.export(audio_buffer, format="mp3")
        audio_buffer.seek(0)
    except Exception as e:
        raise HTTPException(500, f"TTS generation failed: {e}")

    return Response(
        content=audio_buffer.getvalue(),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": 'attachment; filename="speech.mp3"',
            "X-Content-Type-Options": "nosniff",
        }
    )

@app.get("/v1/voices")
async def get_voices():
    """Get available voices"""
    return KOKORO_VOICES

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
    }
