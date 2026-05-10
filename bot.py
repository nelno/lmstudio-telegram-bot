import logging
import sqlite3
import requests
import time
import os
import base64
import tempfile
import ast
import cv2  # pip install opencv-python
import asyncio
import telegram.error
import re
from datetime import datetime
from telegram.constants import ChatAction
from dotenv import load_dotenv
from telegram import Update, User
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Bot & LM Studio Config
# ------------------------------------------------------------------------------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set. Please set it in the .env file.")
  
DB_FILE = "conversations.db"

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"   # ← Change to your PC's LAN IP if needed (e.g. http://192.168.1.45:1234/v1)
LM_STUDIO_CHAT_COMPLETIONS_URL = f"{LM_STUDIO_BASE_URL}/chat/completions"

DEFAULT_MODEL = "llama-3.2-3b-instruct"   # ← Change this to your actual vision model name
TOKEN_THRESHOLD = 7975

# increase timeout if needed
LLM_TIMEOUT = 90  # seconds per stage
LLM_MSG_TIMEOUT = 30  # seconds per message

# Video mode setting (per user)
# False = send all frames at once (recommended)
# True  = experimental: send frame pairs with timestamps
DEFAULT_USE_PARTIAL_VIDEO = False

MAX_HISTORY_TURNS=32

# Main conversation parameters (extend as needed)
conversation_params = {
    "max_tokens": 300,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 40,
    "stop": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": None,
    "logit_bias": {},
    "seed": None,
}

# ------------------------------------------------------------------------------
# Global flag to skip images only on the very first message after bot restart
# ------------------------------------------------------------------------------
skip_images_on_first_message = True

# ------------------------------------------------------------------------------
# Database Initialization & CRUD
# ------------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    with conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                last_seen INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                default_model TEXT,
                active_conversation_id INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_conversations (
                conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                conversation_name TEXT NOT NULL,
                model TEXT,
                system_prompt TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                summary TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                default_model TEXT,
                active_conversation_id INTEGER,
                use_partial_video INTEGER DEFAULT 0   -- 0 = False, 1 = True
            )
        """)
    conn.close()

def migrate_user_settings_table():
    """Automatically add missing columns to user_settings table"""
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        try:
            # Add use_partial_video column if it doesn't exist
            c.execute("ALTER TABLE user_settings ADD COLUMN use_partial_video INTEGER DEFAULT 0")
            print("🔧 Added missing column: use_partial_video")
        except sqlite3.OperationalError:
            pass  # Column already exists

        conn.commit()


def ensure_user_settings_row(user_id: int):
    """Ensure user exists while preserving use_partial_video setting"""
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO user_settings 
                (user_id, default_model, active_conversation_id, use_partial_video)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                default_model = COALESCE(excluded.default_model, default_model)
                -- Do NOT touch use_partial_video on conflict (preserve user choice)
        """, (user_id, DEFAULT_MODEL, None, 0))
        conn.commit()

def upsert_user(telegram_user: User):
    user_id = telegram_user.id
    username = telegram_user.username or ""
    first_name = telegram_user.first_name or ""
    last_name = telegram_user.last_name or ""
    last_seen = int(time.time())
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO users (user_id, username, first_name, last_name, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username=excluded.username,
                first_name=excluded.first_name,
                last_name=excluded.last_name,
                last_seen=excluded.last_seen
        """, (user_id, username, first_name, last_name, last_seen))
        c.execute("""
            INSERT INTO user_settings (user_id, default_model, active_conversation_id)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO NOTHING
        """, (user_id, DEFAULT_MODEL, None))

def get_user_settings(user_id: int) -> dict:
    """Get user settings with automatic migration if needed"""
    # Run migration if necessary (safe to call every time)
    migrate_user_settings_table()

    # Ensure the user has a row
    ensure_user_settings_row(user_id)

    # Now safely fetch the data
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT default_model, active_conversation_id, use_partial_video 
            FROM user_settings WHERE user_id = ?
        """, (user_id,))
        row = c.fetchone()

        if row:
            return {
                "default_model": row[0],
                "active_conversation_id": row[1],
                "use_partial_video": bool(row[2]) if row[2] is not None else DEFAULT_USE_PARTIAL_VIDEO
            }

    # Fallback (should rarely happen)
    return {
        "default_model": DEFAULT_MODEL,
        "active_conversation_id": None,
        "use_partial_video": DEFAULT_USE_PARTIAL_VIDEO
    }

def set_user_setting(user_id: int, field: str, value):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        query = f"UPDATE user_settings SET {field} = ? WHERE user_id = ?"
        c.execute(query, (value, user_id))

def create_conversation(user_id: int, name: str, model: str = None) -> int:
    if not model:
        model = get_user_settings(user_id)["default_model"]
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO user_conversations (user_id, conversation_name, model, system_prompt)
            VALUES (?, ?, ?, ?)
        """, (user_id, name, model, ""))
        return c.lastrowid

def get_user_conversations(user_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT conversation_id, conversation_name, model, system_prompt
            FROM user_conversations
            WHERE user_id = ?
            ORDER BY conversation_id ASC
        """, (user_id,))
        rows = c.fetchall()
    return [
        {
            "conversation_id": r[0],
            "conversation_name": r[1],
            "model": r[2],
            "system_prompt": r[3]
        }
        for r in rows
    ]

def switch_conversation(user_id: int, conversation_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT conversation_id FROM user_conversations
            WHERE conversation_id = ? AND user_id = ?
        """, (conversation_id, user_id))
        if c.fetchone():
            set_user_setting(user_id, "active_conversation_id", conversation_id)
            return True
    return False

def update_conversation_model(conversation_id: int, model: str):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("UPDATE user_conversations SET model = ? WHERE conversation_id = ?", (model, conversation_id))

def update_conversation_system_prompt(conversation_id: int, prompt: str):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("UPDATE user_conversations SET system_prompt = ? WHERE conversation_id = ?", (prompt, conversation_id))

def get_messages(conversation_id: int) -> list:
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (conversation_id,))
        rows = c.fetchall()
    
    messages = []
    for role, content_str in rows:
        try:
            content = ast.literal_eval(content_str)
        except:
            content = content_str
        messages.append({"role": role, "content": content})
    return messages

def get_clean_history(cid: int, max_text_turns: int = MAX_HISTORY_TURNS) -> list:
    """Return last N messages for LLM context.
    - Normal text messages are kept as-is.
    - OLD images/videos are replaced with clean text summaries using their ACTUAL description.
    - Only the VERY LAST (current) image or video keeps the full structured content for the opinion stage.
    """
    all_msgs = get_messages(cid)
    cleaned = []
    
    for i, m in enumerate(all_msgs):
        role = m.get("role")
        content = m.get("content")
        
        if role in ("video_desc", "image_desc", "system"):
            continue
            
        # === Handle IMAGE messages ===
        if role == "user" and isinstance(content, list):
            has_image = any(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in content
            )
            if has_image:
                is_current = (i == len(all_msgs) - 1)  # last message in history
                
                if is_current:
                    # Current image → keep full content so opinion stage works
                    cleaned.append({"role": role, "content": content})
                else:
                    # Find the matching image_desc that comes AFTER this image
                    desc_text = "unknown image"
                    for later_msg in all_msgs[i+1:]:
                        if later_msg.get("role") == "image_desc":
                            desc = later_msg.get("content", "")
                            # Extract the actual description
                            if isinstance(desc, str):
                                if "description:" in desc:
                                    desc_text = desc.split("description:", 1)[1].strip()
                                else:
                                    desc_text = desc.strip()
                            break
                    
                    cleaned.append({
                        "role": role,
                        "content": f"[an image described as: {desc_text}]"
                    })
                continue
                
        # === Handle VIDEO-related opinion prompts ===
        if role == "user" and isinstance(content, str):
            # Check if this looks like an opinion prompt (for video or thumbnail-as-video)
            if ("Current segment description:" in content or 
                "Here is a detailed description of an image:" in content or
                "Video ID:" in content):
                
                is_current = (i == len(all_msgs) - 1)
                
                if is_current:
                    # Current video → keep the full opinion prompt
                    cleaned.append({"role": role, "content": content})
                else:
                    # Find the associated video_desc or image_desc
                    desc_text = "unknown video"
                    for later_msg in all_msgs[i+1:]:
                        if later_msg.get("role") in ("video_desc", "image_desc"):
                            desc = later_msg.get("content", "")
                            if isinstance(desc, str):
                                if "description:" in desc:
                                    desc_text = desc.split("description:", 1)[1].strip()
                                else:
                                    desc_text = desc.strip()
                            break
                    
                    # For videos we can make it clearer
                    if "Video ID" in content or any("Segment" in str(d) for d in all_msgs):
                        cleaned.append({
                            "role": role,
                            "content": f"[a video described as: {desc_text}]"
                        })
                    else:
                        cleaned.append({
                            "role": role,
                            "content": f"[an image described as: {desc_text}]"
                        })
                continue
                
        # === Normal text messages ===
        if role in ("user", "assistant"):
            if isinstance(content, str) and len(content) < 800:  # avoid old base64 or huge strings
                cleaned.append({"role": role, "content": content})
    
    # Limit to recent turns
    return cleaned[-max_text_turns:]

def debug_log_messages(msgs: list, label: str = "Sending to LLM"):
    """Full debug output - no truncation"""
    logger.info(f"=== {label} ===")
    logger.info(f"Total messages: {len(msgs)}")
    
    for i, msg in enumerate(msgs):
        role = msg.get("role", "UNKNOWN")
        content = msg.get("content", "")
        
        logger.info(f"[{i}] {role.upper()}:")
        
        if isinstance(content, list):
            image_count = sum(1 for item in content 
                            if isinstance(item, dict) and item.get("type") == "image_url")
            logger.info(f"    → {image_count} image(s)")
            
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    logger.info(f"    Text: {item.get('text', '')}")
        else:
            logger.info(f"    {content}")
    
    logger.info("=" * 100)

def append_message(conversation_id: int, role: str, content):
    if isinstance(content, (dict, list)):
        content_str = str(content)
    else:
        content_str = content
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content_str, int(time.time()))
        )

def clear_conversation_messages(conversation_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))

def append_summary(conversation_id: int, summary: str):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO conversation_summary (conversation_id, summary, timestamp)
            VALUES (?, ?, ?)
        """, (conversation_id, summary, int(time.time())))

def get_summaries(conversation_id: int) -> list:
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT summary FROM conversation_summary
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (conversation_id,))
        return [r[0] for r in c.fetchall()]

def set_partial_video_mode(user_id: int, enabled: bool):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("UPDATE user_settings SET use_partial_video = ? WHERE user_id = ?", (1 if enabled else 0, user_id))

# ------------------------------------------------------------------------------
# Video Staged Processing Helpers
# ------------------------------------------------------------------------------

def generate_video_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_video_description(text: str) -> str | None:
    """Extract content between <video_desc> and </video_desc>"""
    match = re.search(r'<video_desc>(.*?)</video_desc>', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def store_segment_description(cid: int, video_id: str, segment_id: int, description: str):
    """Store the raw VL description"""
    content = f"Video {video_id} | Segment {segment_id}: {description}"
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (cid, "video_desc", content, int(time.time()))
        )

def get_segment_descriptions(cid: int, video_id: str) -> list:
    """Return all segment descriptions for this video"""
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT content FROM messages 
            WHERE conversation_id = ? AND role = 'video_desc' 
            AND content LIKE ? 
            ORDER BY timestamp ASC
        """, (cid, f"Video {video_id}%"))
        return [row[0] for row in c.fetchall()]

#-------------------------------------------------------------------------------
# prompt builders
#-------------------------------------------------------------------------------

def build_vision_prompt(pair: list, video_id: str, segment_id: int, previous_descriptions: list) -> list:
    """Build prompt for the Vision Stage (2 frames)"""
    content = [
        {"type": "text", "text": f"These are frames from Video ID: {video_id}, Segment {segment_id}."}
    ]

    for b64, ts in pair:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        content.append({"type": "text", "text": f"Frame at {ts} seconds."})

    content.append({
        "type": "text",
        "text": "Describe the scene, actions, objects, and any important details in explicit detail. "
                "Respond with ONLY the description inside <video_desc> tags. Do not add any other text."
    })

    return content


def build_opinion_prompt(video_id: str, all_descriptions: list, current_desc: str) -> str:
    """Build prompt for Opinion Stage - only previous + current"""
    if len(all_descriptions) <= 1:
        prev_text = "This is the first segment of the video."
    else:
        prev_text = f"Previous segment: {all_descriptions[-2]}"   # use actual previous

    prompt = f"""Video ID: {video_id}

{prev_text}

Current segment description:
{current_desc}

Give your honest opinion and commentary on what is happening in the video based on these two segments. 
Speak naturally and in character. Be concise but insightful."""

    return prompt


def build_final_summary_prompt(video_id: str, all_descriptions: list) -> str:
    """Final summary after all segments"""
    desc_list = "\n\n".join([f"Segment {i+1}: {d}" for i, d in enumerate(all_descriptions)])

    return f"""Video ID: {video_id} - Complete Analysis

Here are all segment descriptions from the video:

{desc_list}

Provide a coherent overall summary of the entire video, commenting on one or two of the scenes and then providing your final thoughts/commentary. 
Speak naturally in character."""

def extract_image_description(text: str) -> str | None:
    """Extract content between <image_desc> and </image_desc>"""
    match = re.search(r'<image_desc>(.*?)</image_desc>', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def build_image_description_prompt() -> list:
    """Prompt for the first stage: pure description of the image"""
    return [
        {"type": "text", "text": "Describe the image in explicit detail. "
                                  "Include scene, objects, people, actions, text, colors, mood, and any important details. "
                                  "Respond with ONLY the description inside <image_desc> tags. Do not add any other text."}
    ]


def build_image_opinion_prompt(image_desc: str) -> str:
    """Second stage: ask for opinion/commentary in character"""
    return f"""Here is a detailed description of an image:

{image_desc}

Give your honest opinion and commentary on what you see in this image. 
Speak naturally and in character. Be insightful, concise, and engaging. 
What stands out? What does it make you think or feel?"""

# ------------------------------------------------------------------------------
# Telegram API Helpers
# ------------------------------------------------------------------------------
async def send_chat_action_safe(update: Update, context: ContextTypes.DEFAULT_TYPE, action: ChatAction):
    """
    Safe generic wrapper for sending chat actions.
    Prevents the bot from crashing on Telegram timeouts or network issues.
    """
    if not update or not update.effective_chat:
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=action,
            timeout=15
        )
    except telegram.error.TimedOut:
        logger.warning(f"send_chat_action ({action}) timed out - continuing anyway")
    except Exception as e:
        logger.warning(f"Failed to send chat action '{action}': {type(e).__name__}: {e}")

# ------------------------------------------------------------------------------
# LM Studio API Calls
# ------------------------------------------------------------------------------
def call_lm_studio_chat(messages: list, model: str) -> dict:
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": conversation_params["max_tokens"],
            "temperature": conversation_params["temperature"],
            "top_p": conversation_params["top_p"],
            "top_k": conversation_params["top_k"],
            "presence_penalty": conversation_params["presence_penalty"],
            "frequency_penalty": conversation_params["frequency_penalty"],
            "logit_bias": conversation_params["logit_bias"],
        }
        # ... (keep the if conditions for stop/repeat_penalty/seed)

        resp = requests.post(LM_STUDIO_CHAT_COMPLETIONS_URL, 
                           headers=headers, 
                           json=payload, 
                           timeout=300)   # Increased from 120 to 300 seconds
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"Error calling chat completions: {e}")
        return {"error": str(e)}

# ------------------------------------------------------------------------------
# Image Handling
# ------------------------------------------------------------------------------
async def download_photo_to_base64(photo) -> str:
    """Download the highest resolution photo and convert it to base64"""
    file = await photo.get_file()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            img_bytes = f.read()
        return base64.b64encode(img_bytes).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ------------------------------------------------------------------------------
# Video Handling - Extract frames for Qwen3-VL
# ------------------------------------------------------------------------------
async def extract_frames_from_video(video_file_obj) -> list:
    """Extract frames dynamically based on video duration with better error handling"""
    frames = []  
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        await video_file_obj.download_to_drive(tmp.name)
        video_path = tmp.name

    try:
        logger.info(f"Starting frame extraction for video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video with cv2.VideoCapture")
            raise ValueError("Could not open video file - possibly unsupported format")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0

        logger.info(f"Video info - Duration: {duration_seconds:.1f}s, Total frames: {total_frames}, FPS: {fps:.2f}")

        # Dynamic frame count
        if duration_seconds <= 15:
            num_frames = 2
        elif duration_seconds <= 30:
            num_frames = 4
        elif duration_seconds <= 60:
            num_frames = 5
        elif duration_seconds <= 120:
            num_frames = 10
        else:
            num_frames = 12

        if total_frames <= 0:
            logger.warning("Video has 0 frames")
            cap.release()
            return frames

        step = max(1, total_frames // num_frames)
        
        for i in range(0, total_frames, step):
            if len(frames) >= num_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame at position {i}")
                break

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer).decode('utf-8')
            timestamp_sec = round(i / fps)
            frames.append((b64, timestamp_sec))
        
        cap.release()
        logger.info(f"Successfully extracted {len(frames)} frames from {duration_seconds:.1f}s video")
        return frames

    except Exception as e:
        logger.error(f"Error in extract_frames_from_video: {e}")
        raise
    finally:
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp video file: {e}")

# ------------------------------------------------------------------------------
# Photo Handling
# ------------------------------------------------------------------------------
async def call_vision_stage_for_image(cid: int, update: Update, model: str) -> str:
    """Stage 1 for images: get raw description"""
    try:
        msgs = []
        # Add system prompt if set
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT system_prompt FROM user_conversations WHERE conversation_id = ?", (cid,))
            row = c.fetchone()
            if row and row[0]:
                msgs.append({"role": "system", "content": row[0]})

        # Only the latest user message (which contains the image + instruction)
        last_msg = get_messages(cid)[-1]
        msgs.append({"role": "user", "content": last_msg["content"]})

        debug_log_messages(msgs, label="IMAGE DESCRIPTION STAGE")

        data = await asyncio.wait_for(
            asyncio.to_thread(call_lm_studio_chat, msgs, model),
            timeout=LLM_TIMEOUT
        )

        if "error" in data:
            return f"Error: {data['error']}"
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")

    except asyncio.TimeoutError:
        return "[Image description stage timed out]"
    except Exception as e:
        logger.error(f"Image description stage error: {e}")
        return f"[Error during image description: {str(e)[:100]}]"


async def call_opinion_stage_for_image(cid: int, update: Update, model: str) -> str:
    """Stage 2 for images: get opinion based on description"""
    try:
        msgs = []
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT system_prompt FROM user_conversations WHERE conversation_id = ?", (cid,))
            row = c.fetchone()
            if row and row[0]:
                msgs.append({"role": "system", "content": row[0]})

        # Clean history (ignores previous images and image_desc)
        for m in get_clean_history(cid):
            msgs.append(m)

        # Add the opinion prompt (last user message)
        last_msg = get_messages(cid)[-1]
        msgs.append({"role": "user", "content": last_msg["content"]})

        debug_log_messages(msgs, label="IMAGE OPINION STAGE")

        data = await asyncio.wait_for(
            asyncio.to_thread(call_lm_studio_chat, msgs, model),
            timeout=LLM_TIMEOUT
        )

        if "error" in data:
            return f"Error: {data['error']}"
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")

    except asyncio.TimeoutError:
        return "[Image opinion stage timed out]"
    except Exception as e:
        logger.error(f"Image opinion stage error: {e}")
        return f"[Error during image opinion: {str(e)[:100]}]"
    
async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE, cid: int, caption: str = "") -> list:
    """Process photo with optional caption"""
    try:
        photo = update.message.photo[-1] if update.message.photo else None
        if not photo and update.message.video and update.message.video.thumbnail:
            photo = update.message.video.thumbnail

        if not photo:
            return []

        base64_img = await download_photo_to_base64(photo)

        # Build content with caption if present
        image_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
        ]

        if caption.strip():
            image_content.append({"type": "text", "text": f"Caption: {caption}"})

        image_content.extend(build_image_description_prompt())

        append_message(cid, "user", image_content)

        await send_chat_action_safe(update, context, ChatAction.TYPING)

        model = get_user_settings(update.effective_user.id)["default_model"]

        desc_response = await call_vision_stage_for_image(cid, update, model)
        raw_desc = extract_image_description(desc_response) or desc_response[:800]

        append_message(cid, "image_desc", f"Image description: {raw_desc}")

        # === STAGE 2: Opinion (caption is already in history) ===
        await send_chat_action_safe(update, context, ChatAction.TYPING)

        opinion_prompt = build_image_opinion_prompt(raw_desc)
        if caption.strip():
            opinion_prompt = f"Caption: {caption}\n\n{opinion_prompt}"

        append_message(cid, "user", opinion_prompt)

        opinion_response = await call_opinion_stage_for_image(cid, update, model)

        await update.message.reply_text(opinion_response)
        return []

    except Exception as e:
        logger.error(f"Photo handling error: {e}")
        await update.message.reply_text("❌ Sorry, I had trouble analyzing that image.")
        return []

async def handle_photo_message_from_thumbnail(thumbnail, update: Update, context: ContextTypes.DEFAULT_TYPE, cid: int, caption: str = "") -> list:
    """Treat video thumbnail as a tiny 2-frame video with optional caption support."""
    saved_path = None
    try:
        if not thumbnail:
            await update.message.reply_text("❌ No thumbnail available.")
            return []

        # === DEBUG: Save thumbnail to working directory ===
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"thumbnail_{timestamp}.jpg"
            saved_path = os.path.join(os.getcwd(), filename)

            base64_img = await download_photo_to_base64(thumbnail)
            img_data = base64.b64decode(base64_img)
            with open(saved_path, "wb") as f:
                f.write(img_data)

            logger.info(f"✅ Thumbnail saved for debugging: {saved_path}")
        except Exception as save_err:
            logger.warning(f"Failed to save thumbnail: {save_err}")

        # Get model
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT model FROM user_conversations WHERE conversation_id = ?", (cid,))
            row = c.fetchone()
        model = row[0] if row and row[0] else DEFAULT_MODEL

        # === Pretend this is a 2-frame video ===
        video_id = generate_video_id()

        # Use the SAME thumbnail as both frames with fake timestamps
        base64_img = await download_photo_to_base64(thumbnail)
        fake_frames = [
            (base64_img, 0),   # First frame at 0 seconds
            (base64_img, 2)    # Second frame at 2 seconds
        ]

        caption.strip()

        # Now reuse video processing logic
        segment_descriptions = []

        # Process the single "segment" (pair of frames)
        pair = fake_frames
        segment_id = 1

        await send_chat_action_safe(update, context, ChatAction.TYPING)

        # 1. Vision Stage - Get description of the "video"
        vision_content = build_vision_prompt(pair, video_id, segment_id, segment_descriptions)
        if caption.strip():
            vision_content.insert(1, {"type": "text", "text": f"User caption: {caption}"})

        append_message(cid, "user", vision_content)

        vision_response = await call_vision_stage(cid, update, model)
        desc = extract_video_description(vision_response) or vision_response[:600]

        store_segment_description(cid, video_id, segment_id, desc)
        segment_descriptions.append(desc)

        # 2. Opinion Stage for this segment
        opinion_text = build_opinion_prompt(video_id, segment_descriptions, desc)
        if caption.strip():
            opinion_text = f"Caption: {caption}\n\n{opinion_text}"

        append_message(cid, "user", opinion_text)

        opinion_response = await call_opinion_stage(cid, update, model)

        logger.info("=" * 80)
        logger.info("LLM THUMBNAIL-AS-VIDEO OPINION RESPONSE")
        logger.info("-" * 80)
        logger.info(opinion_response)
        logger.info("=" * 80)

        await update.message.reply_text(opinion_response)

        # 3. Final Summary (since it's a "complete" tiny video)
        await send_chat_action_safe(update, context, ChatAction.TYPING)

        final_text = build_final_summary_prompt(video_id, segment_descriptions)
        if caption.strip():
            final_text = f"Caption: {caption}\n\n{final_text}"

        append_message(cid, "user", final_text)

        final_response = await call_opinion_stage(cid, update, model)

        logger.info("=" * 80)
        logger.info("LLM THUMBNAIL-AS-VIDEO FINAL SUMMARY")
        logger.info("-" * 80)
        logger.info(final_response)
        logger.info("=" * 80)

        await update.message.reply_text(final_response)

        return []   # Prevent main chat() from sending another reply

    except Exception as e:
        logger.error(f"Thumbnail-as-video handling error: {e}")
        await update.message.reply_text("❌ Sorry, I had trouble analyzing the video thumbnail.")
        return []

# ------------------------------------------------------------------------------
# Video Handling with Incremental Progress
# ------------------------------------------------------------------------------
async def call_vision_stage(cid: int, update: Update, model: str) -> str:
    try:
        """Call for raw video description (vision stage)"""
        msgs = []
        # Add system prompt if exists
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT system_prompt FROM user_conversations WHERE conversation_id = ?", (cid,))
            row = c.fetchone()
            if row and row[0]:
                msgs.append({"role": "system", "content": row[0]})

        # Add only the latest user message (the one with 2 images)
        last_msg = get_messages(cid)[-1]
        if isinstance(last_msg["content"], list):
            msgs.append({"role": "user", "content": last_msg["content"]})
        else:
            msgs.append({"role": last_msg["role"], "content": last_msg["content"]})

        debug_log_messages(msgs, label="VISION STAGE PROMPT")

        # Run with timeout
        data = await asyncio.wait_for(
            asyncio.to_thread(call_lm_studio_chat, msgs, model),
            timeout=LLM_TIMEOUT
        )

        if "error" in data:
            return f"Error: {data['error']}"
        
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    
    except asyncio.TimeoutError:
        logger.error("Vision stage timed out")
        return "[Vision stage timed out - try again with shorter video]"
    except Exception as e:
        logger.error(f"Vision stage error: {e}")
        return f"[Error during vision analysis: {str(e)[:100]}]"

async def call_opinion_stage(cid: int, update: Update, model: str) -> str:
    """Call for opinion/commentary stage"""
    try:
        msgs = []
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT system_prompt FROM user_conversations WHERE conversation_id = ?", (cid,))
            row = c.fetchone()
            if row and row[0]:
                msgs.append({"role": "system", "content": row[0]})

        # Recent clean text history
        for m in get_clean_history(cid):
            msgs.append(m)

        # Add the latest opinion prompt
        last_msg = get_messages(cid)[-1]
        msgs.append({"role": last_msg["role"], "content": last_msg["content"]})

        debug_log_messages(msgs, label="OPINION STAGE PROMPT")

        # Run with timeout
        data = await asyncio.wait_for(
            asyncio.to_thread(call_lm_studio_chat, msgs, model),
            timeout=LLM_TIMEOUT
        )

        if "error" in data:
            return f"Error: {data['error']}"
        
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    
    except asyncio.TimeoutError:
        logger.error("Opinion stage timed out")
        return "[Opinion stage timed out]"
    except Exception as e:
        logger.error(f"Opinion stage error: {e}")
        return f"[Error during opinion stage: {str(e)[:100]}]"

async def handle_video_message(update: Update, context: ContextTypes.DEFAULT_TYPE, cid: int, caption: str = "") -> list:
    """Handles both regular videos and GIF animations with optional caption"""
    video_id = generate_video_id()
    is_gif = bool(update.message.animation)

    try:
        if is_gif:
            media = update.message.animation
            media_type = "GIF"
        else:
            media = update.message.video
            media_type = "video"

        if not media:
            return []

        caption.strip()

        # Get model
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT model FROM user_conversations WHERE conversation_id = ?", (cid,))
            row = c.fetchone()
        model = row[0] if row and row[0] else DEFAULT_MODEL

        try:
            file = await media.get_file()
            frames = await extract_frames_from_video(file)

        except Exception as e:
            # ... (existing too-big / thumbnail fallback logic stays the same) ...
            error_str = str(e).lower()
            if "too big" in error_str or "file is too big" in error_str:
                await update.message.reply_text(
                    "❌ File too large for full processing.\nAnalyzing thumbnail instead..."
                )
                if media.thumbnail:
                    return await handle_photo_message_from_thumbnail(
                        media.thumbnail, update, context, cid, caption
                    )
                else:
                    await update.message.reply_text("❌ No thumbnail available.")
                    return []
            else:
                await update.message.reply_text("❌ Sorry, I had trouble processing that file.")
                return []

        # === Normal processing ===
        total_pairs = (len(frames) + 1) // 2
        segment_descriptions = []

        for idx in range(0, len(frames), 2):
            pair = frames[idx:idx + 2]
            segment_id = (idx // 2) + 1

            await send_chat_action_safe(update, context, ChatAction.TYPING)

            # Vision prompt with caption
            vision_content = build_vision_prompt(pair, video_id, segment_id, segment_descriptions)
            if caption.strip():
                # Insert caption at the beginning
                vision_content.insert(1, {"type": "text", "text": f"User caption: {caption}"})

            append_message(cid, "user", vision_content)

            vision_response = await call_vision_stage(cid, update, model)
            desc = extract_video_description(vision_response) or vision_response[:600]

            store_segment_description(cid, video_id, segment_id, desc)
            segment_descriptions.append(desc)

            # Opinion stage with caption context
            opinion_text = build_opinion_prompt(video_id, segment_descriptions, desc)
            if caption.strip():
                opinion_text = f"Caption: {caption}\n\n{opinion_text}"

            append_message(cid, "user", opinion_text)

            opinion_response = await call_opinion_stage(cid, update, model)
            await update.message.reply_text(opinion_response)
            await asyncio.sleep(1.2)

        # Final summary
        await send_chat_action_safe(update, context, ChatAction.TYPING)

        final_text = build_final_summary_prompt(video_id, segment_descriptions)
        if caption.strip():
            final_text = f"Caption: {caption}\n\n{final_text}"

        append_message(cid, "user", final_text)

        final_response = await call_opinion_stage(cid, update, model)
        await update.message.reply_text(final_response)

    except Exception as e:
        logger.error(f"Video/GIF error: {e}")
        await update.message.reply_text(f"❌ Sorry, I had trouble processing that {media_type}.")

    return []

# ------------------------------------------------------------------------------
# Unified Chat Handler (Text + Photo + Video) with progress messages
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Unified Chat Handler (Text + Photo + Video + GIF) with caption support
# ------------------------------------------------------------------------------
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    upsert_user(user)

    s = get_user_settings(user.id)
    cid = s["active_conversation_id"]
    if not cid:
        cid = create_conversation(user.id, "Default Thread")
        set_user_setting(user.id, "active_conversation_id", cid)

    await send_chat_action_safe(update, context, ChatAction.TYPING)

    user_content = []
    caption = update.message.caption or ""  # Capture any text caption/description

    try:
        if update.message.text and not any([
            update.message.photo,
            update.message.video,
            update.message.animation
        ]):
            user_content.append({"type": "text", "text": update.message.text})

        elif update.message.photo:
            user_content = await handle_photo_message(update, context, cid, caption)

        elif update.message.video or update.message.animation:
            user_content = await handle_video_message(update, context, cid, caption)

    except Exception as e:
        logger.error(f"Chat handler error: {e}")
        await update.message.reply_text("❌ Something went wrong while processing your message.")
        return

    if not user_content:
        return

    # Store the user message
    # For photos and videos we store the full content (including images + caption)
    # For pure text we store only the text string
    if isinstance(user_content, list) and len(user_content) == 1 and user_content[0].get("type") == "text":
        content_to_store = user_content[0]["text"]
    else:
        content_to_store = user_content

    append_message(cid, "user", content_to_store)

    # Prepare clean history for LM Studio
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT model, system_prompt FROM user_conversations WHERE conversation_id = ?", (cid,))
        row = c.fetchone()

    model = row[0] if row and row[0] else s["default_model"]
    system_prompt = row[1] if row and row[1] else ""

    msgs = []
    if system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt})

    # Add recent clean text history (ignores images and video_desc)
    for m in get_clean_history(cid):
        msgs.append(m)

    # Add the CURRENT user message (this is where the new photo/video + caption goes)
    if isinstance(user_content, list):
        msgs.append({"role": "user", "content": user_content})
    else:
        msgs.append({"role": "user", "content": user_content})

    # === DEBUGGING ===
    debug_log_messages(msgs, label="FULL PROMPT SENT TO LLM")

    try:
        # Run the blocking LLM call in a thread with timeout
        data = await asyncio.wait_for(
            asyncio.to_thread(call_lm_studio_chat, msgs, model),
            timeout=LLM_MSG_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.error("LLM response timed out")
        await update.message.reply_text("❌ The model didn't respond in time. Please try again.")
        return
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        await update.message.reply_text("❌ Sorry, I had trouble getting a response from the model.")
        return

    if "error" in data:
        await update.message.reply_text(f"API Error: {data['error']}")
        return

    try:
        assistant_text = data["choices"][0]["message"]["content"]
    except:
        assistant_text = "No content in response."

    append_message(cid, "assistant", assistant_text)

    await update.message.reply_text(assistant_text, parse_mode=ParseMode.MARKDOWN)

# ------------------------------------------------------------------------------
# Command Handlers
# ------------------------------------------------------------------------------

async def summarize_thread_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args

    # If user typed /summarize_thread <id>, use that ID; else active conversation
    cid = None
    if args:
        try:
            cid = int(args[0])
        except ValueError:
            await update.message.reply_text(
                "*Invalid conversation ID.*", parse_mode=ParseMode.MARKDOWN
            )
            return

    if not cid:
        s = get_user_settings(user_id)
        cid = s["active_conversation_id"]

    if not cid:
        await update.message.reply_text(
            "*No active conversation or invalid ID.*", parse_mode=ParseMode.MARKDOWN
        )
        return

    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT model FROM user_conversations WHERE conversation_id = ?", (cid,))
        row = c.fetchone()

    if not row:
        await update.message.reply_text(
            "*Conversation not found.*", parse_mode=ParseMode.MARKDOWN
        )
        return

    model = row[0] if row[0] else DEFAULT_MODEL

    # Note: Full summarize_conversation function removed for simplicity.
    # You can add it back later if needed.
    await update.message.reply_text(
        f"*Summarize feature is simplified in this version.*\nConversation {cid} selected.",
        parse_mode=ParseMode.MARKDOWN
    )

async def set_parameter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: `/set <param> <value>`", parse_mode=ParseMode.MARKDOWN
        )
        return

    param = args[0].lower()
    val = " ".join(args[1:])
    if param in conversation_params:
        try:
            if param in ["max_tokens", "top_k"]:
                conversation_params[param] = int(val)
            elif param in ["temperature", "top_p", "presence_penalty", "frequency_penalty"]:
                conversation_params[param] = float(val)
            elif param == "stop":
                conversation_params["stop"] = None if val.lower() == "none" else val
            elif param == "repeat_penalty":
                conversation_params["repeat_penalty"] = None if val.lower() == "none" else float(val)
            elif param == "seed":
                conversation_params["seed"] = None if val.lower() == "none" else int(val)
            else:
                conversation_params[param] = val
            await update.message.reply_text(
                f"*Set* `{param}` *to* `{conversation_params[param]}`",
                parse_mode=ParseMode.MARKDOWN
            )
        except:
            await update.message.reply_text(
                f"Invalid value for `{param}`: `{val}`", parse_mode=ParseMode.MARKDOWN
            )
    else:
        await update.message.reply_text(
            f"Unknown parameter: `{param}`", parse_mode=ParseMode.MARKDOWN
        )

async def show_parameters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = [f"`{k}` = `{v}`" for k, v in conversation_params.items()]
    formatted = "\n".join(lines)
    msg = "*Current parameters:*\n" + formatted
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def clear_context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s = get_user_settings(user_id)
    cid = s["active_conversation_id"]
    if not cid:
        await update.message.reply_text(
            "*No active context to clear.*", parse_mode=ParseMode.MARKDOWN
        )
        return
    clear_conversation_messages(cid)
    await update.message.reply_text(
        "*Context cleared.*", parse_mode=ParseMode.MARKDOWN
    )

async def show_summaries_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cid = get_user_settings(user_id)["active_conversation_id"]
    if not cid:
        await update.message.reply_text(
            "*No active conversation.*", parse_mode=ParseMode.MARKDOWN
        )
        return
    sums = get_summaries(cid)
    if sums:
        bullet_sums = "\n\n".join(f" {s}" for s in sums)
        await update.message.reply_text(
            f"*Summaries for Conversation {cid}:*\n\n{bullet_sums}",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text("*No summaries found.*", parse_mode=ParseMode.MARKDOWN)

async def new_conversation_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    name = " ".join(context.args) or "Unnamed"
    conv_id = create_conversation(user_id, name)
    switch_conversation(user_id, conv_id)
    await update.message.reply_text(
        f"New conversation *'{name}'* created.\nSwitched to conversation *ID={conv_id}*",
        parse_mode=ParseMode.MARKDOWN
    )

async def list_threads_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    convs = get_user_conversations(user_id)
    s = get_user_settings(user_id)

    if not convs:
        await update.message.reply_text("*No conversations.*", parse_mode=ParseMode.MARKDOWN)
        return

    lines = []
    for c in convs:
        cid = c["conversation_id"]
        cname_escaped = c["conversation_name"].replace("_", "\\_")
        active_prefix = "**(active)** " if cid == s["active_conversation_id"] else ""
        lines.append(
            f"{active_prefix}**ID {cid}**: [{cname_escaped}] -> Model: `{c['model']}`"
        )

    out_text = "*Your conversations:*\n\n" + "\n".join(lines)
    await update.message.reply_text(out_text, parse_mode=ParseMode.MARKDOWN)

async def switch_thread_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text(
            "Usage: `/switch_thread <id>`", parse_mode=ParseMode.MARKDOWN
        )
        return

    try:
        cid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("*Invalid conversation ID.*", parse_mode=ParseMode.MARKDOWN)
        return

    if switch_conversation(user_id, cid):
        await update.message.reply_text(
            f"Switched to conversation *ID={cid}*", parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text(
            "*Not found or not owned by you.*", parse_mode=ParseMode.MARKDOWN
        )

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cid = get_user_settings(user_id)["active_conversation_id"]
    if not cid:
        await update.message.reply_text(
            "*No active conversation.*", parse_mode=ParseMode.MARKDOWN
        )
        return
    if not context.args:
        await update.message.reply_text(
            "Usage: `/set_model <model_name>`", parse_mode=ParseMode.MARKDOWN
        )
        return
    model = " ".join(context.args)
    update_conversation_model(cid, model)
    await update.message.reply_text(
        f"Conversation *{cid}* model set to `{model}`", parse_mode=ParseMode.MARKDOWN
    )

async def set_system_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cid = get_user_settings(user_id)["active_conversation_id"]
    if not cid:
        await update.message.reply_text("*No active conversation.*", parse_mode=ParseMode.MARKDOWN)
        return
    prompt = " ".join(context.args)
    update_conversation_system_prompt(cid, prompt)
    await update.message.reply_text("*System prompt updated.*", parse_mode=ParseMode.MARKDOWN)

async def show_system_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s = get_user_settings(user_id)
    cid = s["active_conversation_id"]
    if not cid:
        await update.message.reply_text(
            "*No active conversation.*", parse_mode=ParseMode.MARKDOWN
        )
        return

    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT system_prompt FROM user_conversations WHERE conversation_id = ?", (cid,))
        row = c.fetchone()

    if row and row[0]:
        system_prompt = row[0]
        await update.message.reply_text(
            f"*System Prompt for conversation {cid}:*\n\n```{system_prompt}```",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text("*No system prompt set.*", parse_mode=ParseMode.MARKDOWN)

async def list_models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*list_models command requires additional setup (LM_STUDIO_MODELS_URL).*\n"
        "You can add it later if needed.", 
        parse_mode=ParseMode.MARKDOWN
    )

async def completion_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Completion command not fully migrated in this vision-enabled version.*", 
        parse_mode=ParseMode.MARKDOWN
    )

async def embedding_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Embedding command not fully migrated in this vision-enabled version.*", 
        parse_mode=ParseMode.MARKDOWN
    )

async def toggle_video_mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s = get_user_settings(user_id)
    new_mode = not s["use_partial_video"]
    
    set_partial_video_mode(user_id, new_mode)
    
    mode_text = "✅ **Partial Video Mode** (frame pairs with timestamps)" if new_mode else "✅ **Normal Video Mode** (all frames at once)"
    await update.message.reply_text(
        f"Video processing mode switched!\n\nCurrent mode: {mode_text}\n\n"
        f"Use /toggle_video_mode again to switch back.",
        parse_mode=ParseMode.MARKDOWN
    )

async def video_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s = get_user_settings(user_id)
    
    mode = "Partial Video Mode (frame pairs with timestamps)" if s["use_partial_video"] else "Normal Video Mode (all frames at once - recommended)"
    frames = "12 frames (in pairs)" if s["use_partial_video"] else "12 frames (sent together)"
    
    await update.message.reply_text(
        f"🎥 **Video Processing Status**\n\n"
        f"**Current Mode:** {mode}\n"
        f"**Frames extracted:** {frames}\n"
        f"**Timeout:** 5 minutes (300s)\n\n"
        f"Use `/toggle_video_mode` to switch between Normal and Partial mode.\n"
        f"Normal mode usually gives better results and is faster.",
        parse_mode=ParseMode.MARKDOWN
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors and optionally notify the user."""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
    
    # Optional: Try to tell the user something went wrong (safely)
    if update and hasattr(update, "effective_chat") and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ Something went wrong while processing your message. Please try again."
            )
        except Exception:
            pass  # Don't crash the error handler itself

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    init_db()

    # Optional: longer timeouts for the whole bot
    request = telegram.request.HTTPXRequest(
        connect_timeout=30,
        read_timeout=90,
        write_timeout=90,
        pool_timeout=90,
    )
    
    app = Application.builder().token(BOT_TOKEN).request(request).build()

    # Register the error handler
    app.add_error_handler(error_handler)

    # Commands
    app.add_handler(CommandHandler("set", set_parameter))
    app.add_handler(CommandHandler("show_params", show_parameters))
    app.add_handler(CommandHandler("clear_context", clear_context_command))
    app.add_handler(CommandHandler("show_summaries", show_summaries_command))
    app.add_handler(CommandHandler("new_thread", new_conversation_command))
    app.add_handler(CommandHandler("list_threads", list_threads_command))
    app.add_handler(CommandHandler("switch_thread", switch_thread_command))
    app.add_handler(CommandHandler("set_model", set_model_command))
    app.add_handler(CommandHandler("set_system_prompt", set_system_prompt_command))
    app.add_handler(CommandHandler("show_system_prompt", show_system_prompt_command))
    app.add_handler(CommandHandler("list_models", list_models_command))
    app.add_handler(CommandHandler("summarize_thread", summarize_thread_command))
    app.add_handler(CommandHandler("toggle_video_mode", toggle_video_mode_command))
    app.add_handler(CommandHandler("video_status", video_status_command))
    app.add_handler(MessageHandler(filters.PHOTO, chat))
    app.add_handler(MessageHandler(filters.VIDEO, chat))
    app.add_handler(MessageHandler(filters.VIDEO | filters.ANIMATION, chat))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    print("✅ LM Studio Telegram Bot with **Vision/Image support** is running!")
    print("Send photos or text+photos to test vision capabilities.")
    app.run_polling()

if __name__ == "__main__":
    main()