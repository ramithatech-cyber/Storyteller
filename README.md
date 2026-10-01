# Chronicle AI: Interactive Story Studio

An interactive AI storyteller built with Streamlit. Pick a story world, steer the plot in a chat, and listen to each chapter read aloud by a narrator voice.

## Features

- **Five story worlds:** Cyberpunk Noir, Dark Fantasy, Space Opera, Post-Apocalyptic, Steampunk Mystery
- **Conversational storytelling:** a LangChain LCEL chain (prompt → OpenAI → output parser) remembers the conversation and ends each reply with a hook or a choice
- **Voice narration:**
  - ElevenLabs free (premade) voices, with automatic retry on other free voices and models
  - Falls back to free Google Text-to-Speech (gTTS) if ElevenLabs is unavailable or no key is set
- **Story memory controls:** chat history is kept short to save tokens, and you can reset it from the sidebar

## Tech stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| LLM orchestration | LangChain (`langchain-core`, `langchain-openai`) |
| Model | OpenAI (`gpt-4o-mini` by default) |
| Narration | ElevenLabs, gTTS |

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/ramithatech-cyber/Storyteller.git
cd Storyteller
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `OPENAI_API_KEY` | yes | none | The app stops with an error without it |
| `ELEVENLABS_API_KEY` | no | none | Without it, narration uses Google TTS |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | |
| `ELEVENLABS_MODEL` | no | `eleven_multilingual_v2` | |

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501, choose a story world in the sidebar, and type in the chat box to direct the story.

## Project structure

```
Storyteller/
├── app.py             # Streamlit app: UI, LangChain story chain, narration
├── requirements.txt
├── .env.example       # template for API keys
├── .gitignore
├── LICENSE
└── README.md
```

## License

[MIT](LICENSE)
