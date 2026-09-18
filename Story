import io
import os
import re

import streamlit as st
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 1. Page Configuration
st.set_page_config(
    page_title="Chronicle AI - Story Studio",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Load API Keys from .env
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
MAX_TTS_CHARS = 2500      # protects the free-plan monthly character quota
MAX_HISTORY_MESSAGES = 20  # keeps the prompt small and cheap

# 3. ElevenLabs default ("premade") voices -- these work on the FREE plan.
#    Voice Library / community voices need a paid plan (that caused the 402).
FALLBACK_VOICES = {
    "Sarah": "EXAVITQu4vr4xnJyEWIi",
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Adam": "pNInz6obpgDQGcFmaJgB",
    "Rachel": "21m00Tcm4TzveoiLJk3M",
}
DEFAULT_VOICE_ID = FALLBACK_VOICES["Sarah"]

# 4. High-Contrast High-Gloss CSS Engine
st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1c233d 0%, #0a0c14 80%) !important;
        color: #FFFFFF !important;
    }
    .glowing-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #FF2A6D, #05D9E8, #FFFFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
        margin-bottom: 0px;
        text-shadow: 0 0 30px rgba(5, 217, 232, 0.4);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(20, 24, 40, 0.95) 0%, rgba(10, 12, 20, 0.98) 100%) !important;
        border-right: 1px solid rgba(5, 217, 232, 0.3) !important;
        box-shadow: 5px 0 25px rgba(0,0,0,0.7) !important;
    }
    [data-testid="stChatMessage"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%) !important;
        border: 1px solid rgba(5, 217, 232, 0.6) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5),
                    inset 0 1px 1px rgba(255, 255, 255, 0.8),
                    0 0 15px rgba(5, 217, 232, 0.2) !important;
        margin-bottom: 20px !important;
        padding: 18px !important;
    }
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] div {
        color: #FFFFFF !important;
        font-size: 1.08rem !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.8) !important;
    }
    [data-testid="stChatInput"] {
        background: linear-gradient(135deg, rgba(22, 28, 48, 0.95) 0%, rgba(12, 16, 28, 0.98) 100%) !important;
        border: 2px solid #05D9E8 !important;
        border-radius: 20px !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.6),
                    0 0 25px rgba(5, 217, 232, 0.4) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #A0AAB2 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.02) 100%) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.5) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF2A6D 0%, #9a0036 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.8), 0 4px 15px rgba(255, 42, 109, 0.4) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# 5. Helper Functions
def _api_error_message(err: ApiError) -> str:
    """Pull a readable message out of an ElevenLabs ApiError."""
    body = err.body
    if isinstance(body, dict):
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            return detail.get("message", str(detail))
        return str(detail)
    return str(body)


@st.cache_resource
def get_eleven_client():
    """One shared ElevenLabs client (None if the key is missing)."""
    api_key = (os.getenv("ELEVENLABS_API_KEY") or os.getenv("elevenlabs_api_key") or "").strip()
    return ElevenLabs(api_key=api_key) if api_key else None


@st.cache_data(ttl=3600, show_spinner=False)
def get_free_voices() -> dict:
    """Voices this account can use for free (premade), with a safe fallback."""
    client = get_eleven_client()
    if client is None:
        return FALLBACK_VOICES
    try:
        response = client.voices.get_all()
        voices = {
            v.name: v.voice_id
            for v in response.voices
            if getattr(v, "category", None) == "premade"
        }
        return dict(sorted(voices.items())) or FALLBACK_VOICES
    except Exception:
        return FALLBACK_VOICES


def clean_for_speech(text: str) -> str:
    """Remove markdown symbols so the narrator doesn't read them aloud."""
    text = re.sub(r"[*_#`>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_TTS_CHARS:
        cut = text[:MAX_TTS_CHARS]
        text = cut[: cut.rfind(".") + 1] or cut
    return text


def google_tts(speech_text):
    """100% free narration (no API key) using Google Text-to-Speech."""
    try:
        from gtts import gTTS
    except ImportError:
        st.warning("Free Google voice needs one install: run  venv\\Scripts\\pip install gTTS  then restart.")
        return None
    try:
        buf = io.BytesIO()
        gTTS(text=speech_text, lang="en", tld="co.uk").write_to_fp(buf)
        return buf.getvalue()
    except Exception as err:
        st.warning(f"🔇 Google voice failed: {err}")
        return None


def elevenlabs_tts(speech_text, voice_id):
    """Try the chosen free ElevenLabs voice, then other free voices/models.
    Returns (audio_bytes or None, last_error_message)."""
    client = get_eleven_client()
    if client is None:
        return None, "ELEVENLABS_API_KEY not found in .env"

    voice_ids = [voice_id] + [v for v in FALLBACK_VOICES.values() if v != voice_id]
    last_error = ""
    for model in (TTS_MODEL, "eleven_flash_v2_5"):
        for vid in voice_ids:
            try:
                stream = client.text_to_speech.convert(
                    text=speech_text,
                    voice_id=vid,
                    model_id=model,
                    output_format="mp3_44100_128",
                )
                return b"".join(stream), ""
            except ApiError as err:
                last_error = f"{err.status_code}: {_api_error_message(err)}"
                # Only voice problems are worth retrying with another voice
                if err.status_code not in (400, 402, 404, 422):
                    return None, last_error
            except Exception as err:
                return None, str(err)
    return None, last_error


def text_to_speech(text, voice_id, engine):
    """Returns MP3 bytes, or None. Never crashes the app."""
    speech_text = clean_for_speech(text)
    if not speech_text:
        return None

    if engine.startswith("ElevenLabs"):
        audio, error = elevenlabs_tts(speech_text, voice_id)
        if audio:
            return audio
        st.info(f"ElevenLabs unavailable ({error}). Switched to the free Google voice.")

    return google_tts(speech_text)


@st.cache_resource
def get_chain():
    """Initializes the LangChain LCEL pipeline."""
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.8)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are 'Chronicle', a high-concept interactive storyteller. The"
            " story is set in a {genre} world. Craft vivid, dramatic narratives"
            " with rich atmosphere. Use expressive punctuation like exclamation"
            " points, ellipses (...), and vivid descriptions. Keep each reply"
            " under 250 words and end with a hook or a choice for the reader.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


def welcome_message(genre_name):
    return {
        "role": "assistant",
        "content": f"Welcome to the **{genre_name}** universe. What storyline shall we forge today?",
    }


# 6. API key check for OpenAI
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Missing OPENAI_API_KEY! Add it to your .env file and restart the app.")
    st.stop()

chain = get_chain()
VOICE_MAP = get_free_voices()

# 7. Sidebar Controls
with st.sidebar:
    st.markdown("### 🎛️ Story Engine Settings")

    genre = st.selectbox(
        "Story World Theme",
        ["Cyberpunk Noir", "Dark Fantasy", "Space Opera", "Post-Apocalyptic", "Steampunk Mystery"],
    )

    tts_engine = st.radio(
        "Narration Engine",
        ["ElevenLabs (free voices)", "Google TTS (free, no key)"],
        help="If ElevenLabs fails, the app automatically falls back to Google TTS.",
    )
    narrator_voice = st.selectbox(
        "Narrator Voice 🎙️",
        list(VOICE_MAP.keys()),
        disabled=not tts_engine.startswith("ElevenLabs"),
    )
    narration_enabled = st.toggle("Enable Voice Narration 🔊", value=True)

    st.markdown("---")
    st.markdown(
        """
    <div class="metric-card">
        <h4 style="margin:0; color:#05D9E8;">🌌 Chronicle AI v3.0</h4>
        <p style="font-size:0.85rem; color:#E0E6ED; margin-top:5px;">
            Powered by LangChain LCEL, OpenAI, ElevenLabs, and Streamlit.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("🗑️ Reset Story Memory", use_container_width=True):
        st.session_state["messages"] = [welcome_message(genre)]
        st.rerun()

# 8. Main Title Header
st.markdown('<h1 class="glowing-title">🌌 CHRONICLE AI</h1>', unsafe_allow_html=True)
st.caption(f"Currently Exploring: **{genre}** World | Voice: **{narrator_voice}**")
st.markdown("---")

# 9. Session Memory State Initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = [welcome_message(genre)]

# 10. Render Chat Messages and Audio Players
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio") and narration_enabled:
            st.audio(msg["audio"], format="audio/mp3")

# 11. User Prompt Execution
if user_input := st.chat_input("Direct the story..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    history_messages = [
        (msg["role"], msg["content"])
        for msg in st.session_state.messages[:-1][-MAX_HISTORY_MESSAGES:]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Weaving the narrative and recording audio..."):
            try:
                response_text = chain.invoke(
                    {"input": user_input, "history": history_messages, "genre": genre}
                )
            except Exception as err:
                st.error(f"⚠️ Story engine error: {err}")
                st.session_state.messages.pop()  # drop the unanswered prompt
                st.stop()

            st.markdown(response_text)

            audio_bytes = None
            if narration_enabled:
                voice_id = VOICE_MAP.get(narrator_voice, DEFAULT_VOICE_ID)
                audio_bytes = text_to_speech(response_text, voice_id, tts_engine)

            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

    msg_payload = {"role": "assistant", "content": response_text}
    if audio_bytes:
        msg_payload["audio"] = audio_bytes  # raw bytes replay reliably on rerun
    st.session_state.messages.append(msg_payload)
