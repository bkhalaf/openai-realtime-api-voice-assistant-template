import os
import json
import base64
import asyncio
import logging

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, Response
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OpenAI API key. Please set it in the .env file.")

# Initialize FastAPI
app = FastAPI()

# Constants
SYSTEM_MESSAGE = """You are an AI receptionist for Ameer Eyes Clinic.

When a caller connects, greet them in the following four languages in this exact order:
1. Arabic: "مرحباً بكم في عيادة أمير للعيون"
2. French: "Bienvenue à la clinique ophtalmologique Ameer"
3. Dutch: "Welkom bij Ameer oogkliniek"
4. English: "Welcome to Ameer Eyes Clinic. How can I help you today?"

After the greeting, continue the conversation in whichever language the caller responds in.

Your job is to book appointments. Ask the caller for:
1. Their full name
2. The reason for the visit (e.g. routine checkup, vision problems, surgery consultation, follow-up)
3. Their preferred date and time

Ask one question at a time. Do not ask for other contact information. Assume the clinic has availability and confirm the appointment once all details are collected. Keep the conversation friendly, professional, and concise."""

VOICE = "alloy"
PORT = int(os.getenv("PORT", "5050"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# OpenAI Realtime model
OPENAI_REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"

# Session management
sessions: dict[str, dict] = {}

# List of Event Types to log to the console
LOG_EVENT_TYPES = [
    "response.content.done",
    "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started",
    "session.created",
    "response.text.done",
    "conversation.item.input_audio_transcription.completed",
]

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)


# Root Route
@app.get("/")
async def root():
    return JSONResponse({"message": "Twilio Media Stream Server is running!"})


# Route for Twilio to handle incoming and outgoing calls
@app.api_route("/incoming-call", methods=["GET", "POST"])
async def incoming_call(request: Request):
    logger.info("Incoming call")

    host = request.headers.get("host", "localhost")
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="ar-SA">مرحباً بكم في عيادة أمير للعيون</Say>
    <Say language="fr-FR">Bienvenue à la clinique ophtalmologique Ameer</Say>
    <Say language="nl-NL">Welkom bij Ameer oogkliniek</Say>
    <Say language="en-US">Welcome to Ameer Eyes Clinic</Say>
    <Connect>
        <Stream url="wss://{host}/media-stream" />
    </Connect>
</Response>"""

    return Response(content=twiml_response, media_type="text/xml")


# WebSocket route for media-stream
@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    session_id = f"session_{id(ws)}"
    session = {"transcript": "", "stream_sid": None}
    sessions[session_id] = session

    openai_ws_url = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
    extra_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        async with websockets.connect(
            openai_ws_url, additional_headers=extra_headers
        ) as openai_ws:
            logger.info("Connected to the OpenAI Realtime API")

            # Send session update after a brief delay
            await asyncio.sleep(0.25)
            await _send_session_update(openai_ws)

            # Run both listeners concurrently
            await asyncio.gather(
                _receive_from_openai(openai_ws, ws, session, session_id),
                _receive_from_twilio(openai_ws, ws, session, session_id),
            )
    except websockets.exceptions.ConnectionClosed:
        logger.info("OpenAI WebSocket connection closed")
    except WebSocketDisconnect:
        logger.info(f"Client disconnected ({session_id})")
    except Exception as e:
        logger.error(f"Error in media stream: {e}")
    finally:
        logger.info(f"Client disconnected ({session_id}).")
        logger.info("Full Transcript:")
        logger.info(session["transcript"])

        await process_transcript_and_send(session["transcript"], session_id)

        # Clean up the session
        sessions.pop(session_id, None)


async def _send_session_update(openai_ws):
    """Send session configuration to OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
            "input_audio_transcription": {
                "model": "whisper-1",
            },
        },
    }

    logger.info(f"Sending session update: {json.dumps(session_update)}")
    await openai_ws.send(json.dumps(session_update))


async def _receive_from_openai(openai_ws, twilio_ws: WebSocket, session: dict, session_id: str):
    """Listen for messages from OpenAI and forward audio to Twilio."""
    try:
        async for message in openai_ws:
            try:
                response = json.loads(message)

                if response.get("type") in LOG_EVENT_TYPES:
                    logger.info(f"Received event: {response['type']}")

                # User message transcription handling
                if response.get("type") == "conversation.item.input_audio_transcription.completed":
                    user_message = response.get("transcript", "").strip()
                    session["transcript"] += f"User: {user_message}\n"
                    logger.info(f"User ({session_id}): {user_message}")

                # Agent message handling
                if response.get("type") == "response.done":
                    agent_message = "Agent message not found"
                    output = response.get("response", {}).get("output", [])
                    if output:
                        content_list = output[0].get("content", [])
                        for content in content_list:
                            if content.get("transcript"):
                                agent_message = content["transcript"]
                                break

                    session["transcript"] += f"Agent: {agent_message}\n"
                    logger.info(f"Agent ({session_id}): {agent_message}")

                    if response.get("response", {}).get("status") == "failed":
                        logger.error(
                            f"OpenAI response failed: "
                            f"{json.dumps(response['response'].get('status_details'), indent=2)}"
                        )

                if response.get("type") == "session.updated":
                    logger.info(f"Session updated successfully: {response}")

                # Send audio back to Twilio
                if response.get("type") == "response.audio.delta" and response.get("delta"):
                    audio_delta = {
                        "event": "media",
                        "streamSid": session["stream_sid"],
                        "media": {
                            "payload": base64.b64encode(
                                base64.b64decode(response["delta"])
                            ).decode("ascii"),
                        },
                    }
                    await twilio_ws.send_json(audio_delta)

            except Exception as e:
                logger.error(f"Error processing OpenAI message: {e}, Raw message: {message}")
    except websockets.exceptions.ConnectionClosed:
        logger.info("Disconnected from the OpenAI Realtime API")


async def _receive_from_twilio(openai_ws, twilio_ws: WebSocket, session: dict, session_id: str):
    """Listen for messages from Twilio and forward audio to OpenAI."""
    try:
        while True:
            message = await twilio_ws.receive_text()
            try:
                data = json.loads(message)

                if data.get("event") == "media":
                    if openai_ws.open:
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data["media"]["payload"],
                        }
                        await openai_ws.send(json.dumps(audio_append))

                elif data.get("event") == "start":
                    session["stream_sid"] = data["start"]["streamSid"]
                    logger.info(f"Incoming stream has started {session['stream_sid']}")

                elif data.get("event") == "stop":
                    logger.info("Received stop event from Twilio")
                    if openai_ws.open:
                        await openai_ws.send(
                            json.dumps({"type": "input_audio_buffer.commit"})
                        )
                        await openai_ws.send(
                            json.dumps({"type": "response.create"})
                        )

                else:
                    logger.info(f"Received non-media event: {data.get('event')}")

            except Exception as e:
                logger.error(f"Error parsing message: {e}, Message: {message}")
    except WebSocketDisconnect:
        pass


async def make_chatgpt_completion(transcript: str) -> dict:
    """Make ChatGPT API completion call with structured outputs."""
    logger.info("Starting ChatGPT API call...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-2024-08-06",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Extract appointment details from the transcript: customer name, reason for visit, and preferred date/time.",
                        },
                        {"role": "user", "content": transcript},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "appointment_details_extraction",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "customerName": {"type": "string"},
                                    "reasonForVisit": {"type": "string"},
                                    "preferredDateTime": {"type": "string"},
                                    "language": {"type": "string"},
                                    "specialNotes": {"type": "string"},
                                },
                                "required": [
                                    "customerName",
                                    "reasonForVisit",
                                    "preferredDateTime",
                                    "language",
                                    "specialNotes",
                                ],
                            },
                        },
                    },
                },
                timeout=30.0,
            )

        logger.info(f"ChatGPT API response status: {response.status_code}")
        data = response.json()
        logger.info(f"Full ChatGPT API response: {json.dumps(data, indent=2)}")
        return data

    except Exception as e:
        logger.error(f"Error making ChatGPT completion call: {e}")
        raise


async def send_to_webhook(payload: dict):
    """Send data to webhook."""
    if not WEBHOOK_URL:
        logger.info("No webhook URL configured, skipping webhook send.")
        return

    logger.info(f"Sending data to webhook: {json.dumps(payload, indent=2)}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                WEBHOOK_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30.0,
            )

        logger.info(f"Webhook response status: {response.status_code}")
        if response.is_success:
            logger.info("Data successfully sent to webhook.")
        else:
            logger.error(f"Failed to send data to webhook: {response.reason_phrase}")

    except Exception as e:
        logger.error(f"Error sending data to webhook: {e}")


async def process_transcript_and_send(transcript: str, session_id: str = None):
    """Main function to extract and send appointment details."""
    logger.info(f"Starting transcript processing for session {session_id}...")
    try:
        result = await make_chatgpt_completion(transcript)

        logger.info(f"Raw result from ChatGPT: {json.dumps(result, indent=2)}")

        choices = result.get("choices", [])
        if choices and choices[0].get("message", {}).get("content"):
            try:
                parsed_content = json.loads(choices[0]["message"]["content"])
                logger.info(f"Parsed content: {json.dumps(parsed_content, indent=2)}")

                if parsed_content:
                    await send_to_webhook(parsed_content)
                    logger.info(f"Extracted and sent appointment details: {parsed_content}")
                else:
                    logger.error("Unexpected JSON structure in ChatGPT response")

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from ChatGPT response: {e}")
        else:
            logger.error("Unexpected response structure from ChatGPT API")

    except Exception as e:
        logger.error(f"Error in process_transcript_and_send: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
