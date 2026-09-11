"""Pipecat pipeline wiring for the Twilio voice call: STT -> Claude -> TTS."""

import os

from anthropic import AsyncAnthropic
from fastapi import WebSocket
from loguru import logger

from habidos.prompts import RECEPTIONIST_SYSTEM_PROMPT
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 500

# How long the caller must stay silent (after Silero VAD first detects
# quiet) before we treat their turn as finished and let Claude respond.
# Higher = more room for natural pauses but slower to respond; lower = snappier
# but more likely to cut the caller off mid-thought. 0.6s is pipecat's own
# default; 0.8s gives a little more breathing room for phone conversation.
USER_SPEECH_TIMEOUT_SECS = 0.8

# Conversation history per call, keyed by Twilio call_sid. In-memory only —
# lost on process restart, and not shared across multiple app instances.
_conversation_histories: dict[str, list[dict[str, str]]] = {}


class ClaudeResponder(FrameProcessor):
    """Sends each final transcript to Claude and speaks back its reply."""

    def __init__(self, call_sid: str, client: AsyncAnthropic, **kwargs):
        super().__init__(**kwargs)
        self._call_sid = call_sid
        self._client = client

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                await self._reply_to(text, direction)
        else:
            await self.push_frame(frame, direction)

    async def _reply_to(self, text: str, direction: FrameDirection) -> None:
        history = _conversation_histories.setdefault(self._call_sid, [])
        history.append({"role": "user", "content": text})

        try:
            response = await self._client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=RECEPTIONIST_SYSTEM_PROMPT,
                messages=history,
            )
            reply = next(
                (block.text for block in response.content if block.type == "text"), ""
            )
        except Exception as e:
            logger.error(f"Claude call failed for call {self._call_sid}: {e}")
            reply = "I'm sorry, I'm having trouble right now. I'll have a technician follow up with you."

        history.append({"role": "assistant", "content": reply})

        logger.info(f"Claude reply: {reply}")
        await self.push_frame(TTSSpeakFrame(text=reply), direction)


async def run_bot(websocket: WebSocket, stream_sid: str, call_sid: str) -> None:
    """Build and run the pipeline for a single Twilio call until it ends."""
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        settings=ElevenLabsTTSService.Settings(voice=os.getenv("ELEVENLABS_VOICE_ID")),
    )

    claude_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams()))

    # Turn start uses pipecat's own defaults (VAD, with transcription as a
    # fallback for speech too soft for VAD to catch). Turn stop uses a plain
    # silence timeout instead of the heavier ML turn-analyzer default — see
    # USER_SPEECH_TIMEOUT_SECS above for the tunable knob.
    user_turn = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            stop=[
                SpeechTimeoutUserTurnStopStrategy(
                    user_speech_timeout=USER_SPEECH_TIMEOUT_SECS
                )
            ]
        )
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            user_turn,
            ClaudeResponder(call_sid=call_sid, client=claude_client),
            tts,
            transport.output(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
        enable_rtvi=False,
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info(f"Twilio call {call_sid} disconnected")
        await worker.cancel()

    # A signal handler per call would stomp on the ones from the previous
    # call still on this process, so let uvicorn own SIGINT/SIGTERM instead.
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    await runner.run()
