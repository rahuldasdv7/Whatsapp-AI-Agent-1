"""FastAPI app that bridges a Twilio phone call into a Pipecat voice pipeline."""

import json

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from loguru import logger

from habidos.pipeline import run_bot

load_dotenv()

app = FastAPI()


@app.post("/incoming-call")
async def incoming_call(request: Request) -> Response:
    """Answer a Twilio call and instruct it to stream audio to our WebSocket."""
    host = request.url.hostname
    stream_url = f"wss://{host}/ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Receive Twilio Media Streams audio and run it through the voice pipeline."""
    await websocket.accept()

    messages = websocket.iter_text()
    await messages.__anext__()  # "connected" event, nothing to read from it
    start_message = json.loads(await messages.__anext__())  # "start" event

    stream_sid = start_message["start"]["streamSid"]
    call_sid = start_message["start"]["callSid"]
    logger.info(f"Twilio call started: call_sid={call_sid} stream_sid={stream_sid}")

    await run_bot(websocket, stream_sid, call_sid)
