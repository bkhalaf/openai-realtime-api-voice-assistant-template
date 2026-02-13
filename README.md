# OpenAI Realtime API Voice Assistant

This project implements an AI-powered inbound call agent for Ameer Eyes Clinic. It uses OpenAI's realtime API and integrates with Twilio to handle incoming phone calls, book appointments, and extract customer details.

## Features

- Handles incoming calls using Twilio's voice services
- Utilizes OpenAI's realtime API for natural language processing
- Transcribes user speech and generates AI responses in real-time
- Multilingual support (Arabic, French, Dutch, English)
- Extracts appointment details (name, reason for visit, preferred date/time) from the conversation
- Sends extracted information to a webhook for further processing

## Technologies Used

- Python 3.10+
- FastAPI (web framework)
- WebSockets (for real-time communication)
- OpenAI GPT-4 Realtime API
- Twilio (for telephony services)
- httpx (async HTTP client)
- python-dotenv (for environment variable management)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/bkhalaf/openai-realtime-api-voice-assistant-template.git
   cd openai-realtime-api-voice-assistant-template
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   WEBHOOK_URL=https://your-webhook-url.com
   ```

5. Start the server:
   ```
   python main.py
   ```

## Usage

Once the server is running, it will handle incoming Twilio calls. The AI agent will engage with callers, transcribe their speech, generate appropriate responses, and extract relevant appointment information from the conversation.

## Note

This project is a demonstration and should be adapted for production use, including proper error handling, security measures, and compliance with relevant regulations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
