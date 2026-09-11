"""Pipecat pipeline wiring for the Twilio voice call: STT -> echo -> TTS."""

import os

from fastapi import WebSocket
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner


class EchoResponder(FrameProcessor):
    """Placeholder "brain": speaks the final transcript back, no LLM involved."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                logger.info(f"Echoing back: {text}")
                await self.push_frame(
                    TTSSpeakFrame(text=f"You said: {text}"), direction
                )
        else:
            await self.push_frame(frame, direction)


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

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            EchoResponder(),
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
