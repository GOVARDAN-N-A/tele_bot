# advanced_telegram_groq_bot_enhanced.py
import asyncio
import logging
import os
import re
import time
import html
import json
import aiofiles
import hashlib
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import httpx
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
import contextlib
from functools import wraps
import pickle
import redis.asyncio as redis
from cryptography.fernet import Fernet

# ========================== CONFIG / SETUP ==========================
load_dotenv()

class UserRole(Enum):
    USER = "user"
    ADMIN = "admin"
    PREMIUM = "premium"
    BANNED = "banned"

@dataclass
class Config:
    telegram_bot_token: str
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    request_timeout: float = 30.0
    max_retries: int = 4
    retry_backoff_base: float = 0.8
    history_max_messages: int = 12
    rate_limit_window_sec: int = 60
    rate_limit_max_calls: int = 20
    enable_group_only_on_mention: bool = False
    admin_user_ids: Set[int] = field(default_factory=set)
    redis_url: Optional[str] = None
    encryption_key: Optional[str] = None
    max_tokens: int = 2000
    enable_analytics: bool = True
    enable_voice_transcription: bool = True
    enable_image_generation: bool = False
    webhook_url: Optional[str] = None
    webhook_port: int = 8443
    use_webhook: bool = False

def load_config() -> Config:
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not tg_token or not groq_key:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or GROQ_API_KEY. "
            "Set them as environment variables or in a .env file."
        )
    
    admin_ids = os.getenv("ADMIN_USER_IDS", "").strip()
    admin_set = {int(id.strip()) for id in admin_ids.split(",") if id.strip()} if admin_ids else set()
    
    return Config(
        telegram_bot_token=tg_token,
        groq_api_key=groq_key,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        request_timeout=float(os.getenv("REQUEST_TIMEOUT", "30")),
        max_retries=int(os.getenv("MAX_RETRIES", "4")),
        retry_backoff_base=float(os.getenv("RETRY_BACKOFF_BASE", "0.8")),
        history_max_messages=int(os.getenv("HISTORY_MAX_MESSAGES", "12")),
        rate_limit_window_sec=int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60")),
        rate_limit_max_calls=int(os.getenv("RATE_LIMIT_MAX_CALLS", "20")),
        enable_group_only_on_mention=os.getenv("GROUP_ONLY_ON_MENTION", "false").lower() == "true",
        admin_user_ids=admin_set,
        redis_url=os.getenv("REDIS_URL"),
        encryption_key=os.getenv("ENCRYPTION_KEY"),
        max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
        enable_analytics=os.getenv("ENABLE_ANALYTICS", "true").lower() == "true",
        enable_voice_transcription=os.getenv("ENABLE_VOICE", "true").lower() == "true",
        enable_image_generation=os.getenv("ENABLE_IMAGE_GEN", "false").lower() == "true",
        webhook_url=os.getenv("WEBHOOK_URL"),
        webhook_port=int(os.getenv("WEBHOOK_PORT", "8443")),
        use_webhook=os.getenv("USE_WEBHOOK", "false").lower() == "true",
    )

# Enhanced Logging with rotation
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger("TheriyadhuBot")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        "bot.log", maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ========================== ENHANCED RATE LIMITER ==========================
class EnhancedRateLimiter:
    def __init__(self, window_seconds: int, max_calls: int):
        self.window_seconds = window_seconds
        self.max_calls = max_calls
        self.user_calls: Dict[int, deque] = {}
        self.premium_multiplier = 2
        self.admin_unlimited = True
        
    def allow(self, user_id: int, user_role: UserRole = UserRole.USER) -> Tuple[bool, Optional[int]]:
        if user_role == UserRole.BANNED:
            return False, None
        
        if user_role == UserRole.ADMIN and self.admin_unlimited:
            return True, None
            
        now = time.monotonic()
        q = self.user_calls.setdefault(user_id, deque())
        
        # Purge old entries
        while q and now - q[0] > self.window_seconds:
            q.popleft()
        
        # Adjust limit for premium users
        effective_limit = self.max_calls
        if user_role == UserRole.PREMIUM:
            effective_limit *= self.premium_multiplier
            
        if len(q) < effective_limit:
            q.append(now)
            return True, None
            
        # Calculate retry after
        retry_after = int(self.window_seconds - (now - q[0])) + 1
        return False, max(retry_after, 1)

# ========================== CACHING SYSTEM ==========================
class CacheManager:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client = None
        self.local_cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl = 3600  # 1 hour default
        
    async def connect(self):
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        # Try Redis first
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Fallback to local cache
        if key in self.local_cache:
            value, expiry = self.local_cache[key]
            if time.time() < expiry:
                return value
            else:
                del self.local_cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl = ttl or self.cache_ttl
        
        # Set in Redis
        if self.redis_client:
            try:
                await self.redis_client.set(key, pickle.dumps(value), ex=ttl)
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        # Also set in local cache
        self.local_cache[key] = (value, time.time() + ttl)
    
    async def delete(self, key: str):
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception:
                pass
        if key in self.local_cache:
            del self.local_cache[key]

# ========================== USER MANAGEMENT ==========================
class UserManager:
    def __init__(self, cache_manager: CacheManager, encryption_key: Optional[str] = None):
        self.cache = cache_manager
        self.users: Dict[int, Dict[str, Any]] = {}
        self.encryption_key = encryption_key
        self.fernet = None
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode()[:32].ljust(32, b'0'))
    
    async def get_user(self, user_id: int) -> Dict[str, Any]:
        # Check cache first
        cached = await self.cache.get(f"user:{user_id}")
        if cached:
            return cached
            
        # Get from memory or create new
        if user_id not in self.users:
            self.users[user_id] = {
                "id": user_id,
                "role": UserRole.USER,
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
                "last_message": None,
                "preferences": {},
                "history": [],
                "tokens_used": 0,
                "subscription_expires": None,
            }
        
        user_data = self.users[user_id]
        await self.cache.set(f"user:{user_id}", user_data, ttl=3600)
        return user_data
    
    async def update_user(self, user_id: int, updates: Dict[str, Any]):
        user = await self.get_user(user_id)
        user.update(updates)
        self.users[user_id] = user
        await self.cache.set(f"user:{user_id}", user, ttl=3600)
    
    async def ban_user(self, user_id: int, reason: str = ""):
        await self.update_user(user_id, {
            "role": UserRole.BANNED,
            "ban_reason": reason,
            "banned_at": datetime.now().isoformat()
        })
    
    async def upgrade_to_premium(self, user_id: int, days: int = 30):
        expiry = datetime.now() + timedelta(days=days)
        await self.update_user(user_id, {
            "role": UserRole.PREMIUM,
            "subscription_expires": expiry.isoformat()
        })

# ========================== ANALYTICS ==========================
class Analytics:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stats = defaultdict(lambda: defaultdict(int))
        self.daily_stats = defaultdict(lambda: defaultdict(int))
        
    async def track_event(self, event_type: str, user_id: int, metadata: Dict[str, Any] = None):
        if not self.enabled:
            return
            
        today = datetime.now().strftime("%Y-%m-%d")
        self.stats[event_type]["total"] += 1
        self.stats[event_type][f"user_{user_id}"] += 1
        self.daily_stats[today][event_type] += 1
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    self.stats[event_type][key] = self.stats[event_type].get(key, 0) + value
    
    async def get_stats(self, period: str = "all") -> Dict[str, Any]:
        if period == "today":
            today = datetime.now().strftime("%Y-%m-%d")
            return dict(self.daily_stats[today])
        elif period == "all":
            return dict(self.stats)
        return {}

# ========================== ENHANCED TEXT FORMATTING ==========================
class TextFormatter:
    CODE_BLOCK_RE = re.compile(r"```(?:\s*(\w+))?\n([\s\S]*?)\n```")
    INLINE_CODE_RE = re.compile(r"`([^`]+)`")
    BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
    ITALIC_STAR_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
    ITALIC_UNDER_RE = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
    # Fixed regex for inline KaTeX expressions
    LINK_RE = re.compile(
    r"""```math\s+([\s\S]+?)```KATEX_INLINE_OPEN([\s\S]+?)KATEX_INLINE_CLOSE""",
    re.MULTILINE
)
    HEADING_RE = re.compile(r"^(#{1,6})\s*(.+)$", re.MULTILINE)
    BULLET_RE = re.compile(r"^\s*[\-\+\*]\s+", re.MULTILINE)
    NUMBERED_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
    BLOCKQUOTE_RE = re.compile(r"^>\s+(.+)$", re.MULTILINE)
    
    ALLOWED_TAGS = [
        "b", "strong", "i", "em", "u", "ins", "s", "strike", 
        "del", "code", "pre", "blockquote", "br", "a", "tg-spoiler"
    ]
    
    @classmethod
    def markdown_to_html(cls, text: str) -> str:
        # Code blocks first (to preserve content)
        def code_block_sub(m: re.Match) -> str:
            lang = m.group(1) or ""
            content = html.escape(m.group(2))
            if lang:
                return f"<pre><code class='language-{html.escape(lang)}'>{content}</code></pre>"
            return f"<pre><code>{content}</code></pre>"
        text = cls.CODE_BLOCK_RE.sub(code_block_sub, text)
        
        # Inline code
        text = cls.INLINE_CODE_RE.sub(lambda m: f"<code>{html.escape(m.group(1))}</code>", text)
        
        # Links
        text = cls.LINK_RE.sub(lambda m: f'<a href="{html.escape(m.group(2))}">{html.escape(m.group(1))}</a>', text)
        
        # Text formatting
        text = cls.BOLD_RE.sub(r"<b>\1</b>", text)
        text = cls.ITALIC_STAR_RE.sub(r"<i>\1</i>", text)
        text = cls.ITALIC_UNDER_RE.sub(r"<i>\1</i>", text)
        
        # Headings
        def heading_sub(m: re.Match) -> str:
            level = len(m.group(1))
            title = m.group(2).strip()
            if level <= 2:
                return f"<b><u>{title}</u></b>"
            return f"<b>{title}</b>"
        text = cls.HEADING_RE.sub(heading_sub, text)
        
        # Lists
        text = cls.BULLET_RE.sub("• ", text)
        text = cls.NUMBERED_RE.sub(lambda m: f"{m.group(0)}", text)
        
        # Blockquotes
        text = cls.BLOCKQUOTE_RE.sub(r"<blockquote>\1</blockquote>", text)
        
        return text
    
    @classmethod
    def sanitize_html(cls, text: str) -> str:
        # Store allowed tags
        placeholders = {}
        counter = 0
        
        def store_tag(tag_content: str) -> str:
            nonlocal counter
            key = f"\x00TAG{counter}\x00"
            placeholders[key] = tag_content
            counter += 1
            return key
        
        # Process each allowed tag
        for tag in cls.ALLOWED_TAGS:
            # Opening tags with attributes
            text = re.sub(
                rf"<\s*{tag}(?:\s+[^>]*)?>",
                lambda m: store_tag(m.group(0)),
                text,
                flags=re.IGNORECASE
            )
            # Closing tags
            text = re.sub(
                rf"<\s*/\s*{tag}\s*>",
                lambda m: store_tag(m.group(0)),
                text,
                flags=re.IGNORECASE
            )
        
        # Escape everything else
        text = html.escape(text)
        
        # Restore allowed tags
        for key, val in placeholders.items():
            text = text.replace(html.escape(key), val).replace(key, val)
        
        return text
    
    @classmethod
    def split_message(cls, text: str, max_len: int = 4096) -> List[str]:
        if len(text) <= max_len:
            return [text]
        
        chunks = []
        # Try to split at natural boundaries
        split_patterns = ["\n\n", "\n", ". ", " "]
        
        while len(text) > max_len:
            split_at = -1
            for pattern in split_patterns:
                pos = text.rfind(pattern, 0, max_len)
                if pos > 0:
                    split_at = pos + len(pattern)
                    break
            
            if split_at == -1:
                split_at = max_len
            
            chunks.append(text[:split_at].strip())
            text = text[split_at:].lstrip()
        
        if text:
            chunks.append(text)
        
        return chunks

# ========================== ENHANCED GROQ CLIENT ==========================
class EnhancedGroqClient:
    MODELS = {
        "llama-3.3-70b": "llama-3.3-70b-versatile",
        "llama-3.1-70b": "llama-3.1-70b-versatile",
        "mixtral": "mixtral-8x7b-32768",
        "gemma2": "gemma2-9b-it",
    }
    
    def __init__(self, api_key: str, model: str, timeout: float, 
                 max_retries: int, backoff_base: float, cache_manager: CacheManager):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.cache = cache_manager
        self._client: Optional[httpx.AsyncClient] = None
        self.request_count = 0
        self.error_count = 0
        
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.groq.com/openai/v1",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
                http2=True,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
        return self._client
    
    async def aclose(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_cache_key(self, messages: List[Dict[str, str]], temperature: float) -> str:
        # Create cache key from messages
        content = json.dumps(messages, sort_keys=True) + str(temperature)
        return f"groq:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                             temperature: float, max_tokens: int = 2000,
                             use_cache: bool = True) -> str:
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(messages, temperature)
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info("Cache hit for Groq request")
                return cached
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "frequency_penalty": 0.1,
        }
        
        client = self._get_client()
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.request_count += 1
                resp = await client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                
                # Cache successful response
                if use_cache:
                    await self.cache.set(cache_key, text, ttl=3600)
                
                return text
                
            except httpx.HTTPStatusError as e:
                self.error_count += 1
                status = e.response.status_code
                last_error = e
                
                if status == 429:  # Rate limit
                    retry_after = int(e.response.headers.get("retry-after", 2 ** attempt))
                    await asyncio.sleep(retry_after)
                elif status in (500, 502, 503, 504) and attempt < self.max_retries:
                    backoff = self.backoff_base * (2 ** attempt)
                    await asyncio.sleep(backoff)
                else:
                    break
                    
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                self.error_count += 1
                last_error = e
                if attempt < self.max_retries:
                    backoff = self.backoff_base * (2 ** attempt)
                    await asyncio.sleep(backoff)
                else:
                    break
        
        raise RuntimeError(f"Groq API failed after {self.max_retries} retries: {last_error}")
    
    async def get_models(self) -> List[str]:
        """Get available models from Groq API"""
        try:
            client = self._get_client()
            resp = await client.get("/models")
            resp.raise_for_status()
            data = resp.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return list(self.MODELS.values())

# ========================== CONVERSATION MANAGER ==========================
class ConversationManager:
    def __init__(self, max_messages: int = 12):
        self.max_messages = max_messages
        self.conversations: Dict[int, List[Dict[str, str]]] = {}
        self.summaries: Dict[int, str] = {}
        
    def get_history(self, user_id: int) -> List[Dict[str, str]]:
        return self.conversations.get(user_id, [])
    
    def add_message(self, user_id: int, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim if needed
        if len(self.conversations[user_id]) > self.max_messages:
            # Keep system message if exists
            system_msg = None
            if self.conversations[user_id][0]["role"] == "system":
                system_msg = self.conversations[user_id][0]
            
            # Keep last max_messages
            self.conversations[user_id] = self.conversations[user_id][-self.max_messages:]
            
            # Restore system message
            if system_msg:
                self.conversations[user_id].insert(0, system_msg)
    
    def clear_history(self, user_id: int):
        if user_id in self.conversations:
            del self.conversations[user_id]
        if user_id in self.summaries:
            del self.summaries[user_id]
    
    async def summarize_conversation(self, user_id: int, groq_client) -> str:
        """Create a summary of the conversation for context"""
        history = self.get_history(user_id)
        if len(history) < 4:
            return ""
        
        # Create summary prompt
        summary_messages = [
            {"role": "system", "content": "Summarize the following conversation in 2-3 sentences."},
            {"role": "user", "content": str(history)}
        ]
        
        try:
            summary = await groq_client.chat_completion(summary_messages, temperature=0.3, max_tokens=200)
            self.summaries[user_id] = summary
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize: {e}")
            return ""

# ========================== ENHANCED TELEGRAM HELPERS ==========================
class TypingAction:
    def __init__(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int):
        self.context = context
        self.chat_id = chat_id
        self._task: Optional[asyncio.Task] = None
        
    async def __aenter__(self):
        async def send_typing():
            try:
                while True:
                    await self.context.bot.send_chat_action(self.chat_id, ChatAction.TYPING)
                    await asyncio.sleep(4)
            except asyncio.CancelledError:
                pass
        
        self._task = asyncio.create_task(send_typing())
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

def admin_only(func):
    """Decorator to restrict commands to admins only"""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        cfg: Config = context.application.bot_data["config"]
        user_id = update.effective_user.id
        
        if user_id not in cfg.admin_user_ids:
            await update.message.reply_text(
                "⛔ This command is restricted to administrators only.",
                parse_mode=ParseMode.HTML
            )
            return
        
        return await func(update, context)
    return wrapper

def track_usage(event_type: str):
    """Decorator to track command usage"""
    def decorator(func):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            analytics: Analytics = context.application.bot_data.get("analytics")
            if analytics:
                await analytics.track_event(
                    event_type,
                    update.effective_user.id,
                    {"command": func.__name__}
                )
            return await func(update, context)
        return wrapper
    return decorator

# ========================== ENHANCED SYSTEM PROMPTS ==========================
SYSTEM_PROMPTS = {
    "default": (
        "You are Theriyadhu Bot, an advanced AI assistant for Telegram. "
        "Format all responses using Telegram HTML tags: "
        "<b>, <i>, <u>, <s>, <code>, <pre>, <a href='...'>, <blockquote>, <tg-spoiler>. "
        "Be helpful, concise, and accurate. Use formatting to enhance readability."
    ),
    "coder": (
        "You are a coding expert assistant. Focus on clean, efficient code with best practices. "
        "Use <pre><code> blocks for all code. Include brief explanations only when necessary. "
        "Format responses with Telegram HTML. Suggest optimizations and alternatives when relevant."
    ),
    "teacher": (
        "You are an educational assistant. Explain concepts step-by-step with clarity. "
        "Use examples, analogies, and structured formatting with Telegram HTML tags. "
        "Break down complex topics into digestible parts. Use bullet points and numbered lists."
    ),
    "creative": (
        "You are a creative writing assistant. Help with storytelling, poetry, and creative content. "
        "Use vivid language and engaging narrative techniques. Format with Telegram HTML for emphasis."
    ),
    "analyst": (
        "You are a data analyst assistant. Focus on logical analysis, statistics, and insights. "
        "Present information clearly with structured formatting using Telegram HTML tags. "
        "Use lists and comparisons to highlight key points."
    ),
    "short": (
        "You are a concise assistant. Provide brief, direct answers without unnecessary elaboration. "
        "Use Telegram HTML formatting sparingly. Get straight to the point."
    ),
}

# ========================== COMMAND HANDLERS ==========================
@track_usage("start")
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_manager: UserManager = context.application.bot_data["user_manager"]
    user_id = update.effective_user.id
    
    # Get or create user
    user = await user_manager.get_user(user_id)
    
    # Create inline keyboard
    keyboard = [
        [
            InlineKeyboardButton("📖 Help", callback_data="help"),
            InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
        ],
        [
            InlineKeyboardButton("🎨 Modes", callback_data="modes"),
            InlineKeyboardButton("📊 Stats", callback_data="stats"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_text = (
        f"<b>🤖 Welcome to Theriyadhu Bot!</b>\n\n"
        f"Hello {html.escape(update.effective_user.first_name)}! "
        f"I'm an advanced AI assistant powered by Groq.\n\n"
        f"<b>Quick Start:</b>\n"
        f"• Send me any message to chat\n"
        f"• Use /help for all commands\n"
        f"• Try different modes with /mode\n"
        f"• Check your usage with /stats\n\n"
        f"<i>Your role: {user['role'].value}</i>"
    )
    
    await update.message.reply_text(
        welcome_text,
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

@track_usage("help")
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg: Config = context.application.bot_data["config"]
    user_id = update.effective_user.id
    is_admin = user_id in cfg.admin_user_ids
    
    help_text = (
        "<b>📖 Command Reference</b>\n\n"
        "<b>Basic Commands:</b>\n"
        "• /start - Welcome message and quick menu\n"
        "• /help - This help message\n"
        "• /mode [type] - Change AI personality\n"
        "• /reset - Clear conversation history\n"
        "• /stats - View your usage statistics\n"
        "• /settings - Configure preferences\n"
        "• /ping - Check bot latency\n\n"
        "<b>Conversation:</b>\n"
        "• /summarize - Get conversation summary\n"
        "• /export - Export chat history\n"
        "• /search [query] - Search in history\n\n"
        "<b>Available Modes:</b>\n"
        "• default - Balanced assistant\n"
        "• coder - Programming expert\n"
        "• teacher - Educational focus\n"
        "• creative - Creative writing\n"
        "• analyst - Data analysis\n"
        "• short - Brief responses\n"
    )
    
    if is_admin:
        help_text += (
            "\n<b>Admin Commands:</b>\n"
            "• /admin - Admin control panel\n"
            "• /broadcast [message] - Send to all users\n"
            "• /ban [user_id] - Ban a user\n"
            "• /unban [user_id] - Unban a user\n"
            "• /upgrade [user_id] [days] - Give premium\n"
            "• /system - System information\n"
            "• /logs - View recent logs\n"
        )
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

@track_usage("mode")
async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    
    if not args:
        current_mode = context.user_data.get("mode", "default")
        
        # Create mode selection keyboard
        keyboard = []
        for i in range(0, len(SYSTEM_PROMPTS), 2):
            row = []
            modes = list(SYSTEM_PROMPTS.keys())[i:i+2]
            for mode in modes:
                emoji = {
                    "default": "🤖",
                    "coder": "💻",
                    "teacher": "📚",
                    "creative": "🎨",
                    "analyst": "📊",
                    "short": "⚡"
                }.get(mode, "🔧")
                row.append(InlineKeyboardButton(
                    f"{emoji} {mode.title()}",
                    callback_data=f"mode_{mode}"
                ))
            keyboard.append(row)
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"<b>🎛 Current mode:</b> {html.escape(current_mode)}\n\n"
            "Select a mode from below:",
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
        return
    
    mode = args[0].lower()
    if mode not in SYSTEM_PROMPTS:
        await update.message.reply_text(
            f"❌ Invalid mode. Available: {', '.join(SYSTEM_PROMPTS.keys())}",
            parse_mode=ParseMode.HTML
        )
        return
    
    context.user_data["mode"] = mode
    context.user_data["system_prompt"] = SYSTEM_PROMPTS[mode]
    
    await update.message.reply_text(
        f"✅ Mode changed to <b>{html.escape(mode)}</b>",
        parse_mode=ParseMode.HTML
    )

@track_usage("reset")
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conv_manager: ConversationManager = context.application.bot_data["conversation_manager"]
    user_id = update.effective_user.id
    
    conv_manager.clear_history(user_id)
    context.user_data.clear()
    
    await update.message.reply_text(
        "🧹 <b>Conversation reset!</b>\n"
        "Your chat history has been cleared.",
        parse_mode=ParseMode.HTML
    )

@track_usage("stats")
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_manager: UserManager = context.application.bot_data["user_manager"]
    analytics: Analytics = context.application.bot_data["analytics"]
    user_id = update.effective_user.id
    
    user = await user_manager.get_user(user_id)
    user_stats = await analytics.get_stats("all")
    
    stats_text = (
        f"<b>📊 Your Statistics</b>\n\n"
        f"<b>Account Info:</b>\n"
        f"• Role: {user['role'].value}\n"
        f"• Member since: {user['created_at'][:10]}\n"
        f"• Messages sent: {user.get('message_count', 0)}\n"
        f"• Tokens used: {user.get('tokens_used', 0):,}\n"
    )
    
    if user['role'] == UserRole.PREMIUM:
        expires = user.get('subscription_expires')
        if expires:
            stats_text += f"• Premium expires: {expires[:10]}\n"
    
    await update.message.reply_text(stats_text, parse_mode=ParseMode.HTML)

@track_usage("ping")
async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_time = time.perf_counter()
    message = await update.message.reply_text("🏓 Pinging...")
    end_time = time.perf_counter()
    
    latency_ms = int((end_time - start_time) * 1000)
    
    # Get system stats
    groq_client = context.application.bot_data["groq_client"]
    
    await message.edit_text(
        f"<b>🏓 Pong!</b>\n\n"
        f"• Latency: <code>{latency_ms}ms</code>\n"
        f"• Groq requests: {groq_client.request_count}\n"
        f"• Errors: {groq_client.error_count}",
        parse_mode=ParseMode.HTML
    )

# ========================== ADMIN COMMANDS ==========================
@admin_only
@track_usage("admin")
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("📊 System Stats", callback_data="admin_stats"),
            InlineKeyboardButton("👥 User List", callback_data="admin_users"),
        ],
        [
            InlineKeyboardButton("📢 Broadcast", callback_data="admin_broadcast"),
            InlineKeyboardButton("🔧 Settings", callback_data="admin_settings"),
        ],
        [
            InlineKeyboardButton("📝 Logs", callback_data="admin_logs"),
            InlineKeyboardButton("🔄 Restart", callback_data="admin_restart"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "<b>🔧 Admin Control Panel</b>\n\n"
        "Select an action:",
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

@admin_only
async def cmd_system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import psutil
    import platform
    
    # Get system info
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    groq_client = context.application.bot_data["groq_client"]
    user_manager = context.application.bot_data["user_manager"]
    
    system_text = (
        f"<b>🖥 System Information</b>\n\n"
        f"<b>Server:</b>\n"
        f"• OS: {platform.system()} {platform.release()}\n"
        f"• Python: {platform.python_version()}\n"
        f"• CPU Usage: {cpu_percent}%\n"
        f"• Memory: {memory.percent}% ({memory.used // (1024**3)}GB/{memory.total // (1024**3)}GB)\n"
        f"• Disk: {disk.percent}% used\n\n"
        f"<b>Bot Stats:</b>\n"
        f"• Active users: {len(user_manager.users)}\n"
        f"• API requests: {groq_client.request_count}\n"
        f"• API errors: {groq_client.error_count}\n"
        f"• Uptime: {get_uptime()}"
    )
    
    await update.message.reply_text(system_text, parse_mode=ParseMode.HTML)

def get_uptime():
    """Calculate bot uptime"""
    if not hasattr(get_uptime, 'start_time'):
        get_uptime.start_time = time.time()
    
    uptime_seconds = int(time.time() - get_uptime.start_time)
    days = uptime_seconds // 86400
    hours = (uptime_seconds % 86400) // 3600
    minutes = (uptime_seconds % 3600) // 60
    
    return f"{days}d {hours}h {minutes}m"

# ========================== CALLBACK HANDLERS ==========================
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "help":
        await cmd_help(update, context)
    elif data == "settings":
        await show_settings(update, context)
    elif data.startswith("mode_"):
        mode = data.replace("mode_", "")
        context.user_data["mode"] = mode
        context.user_data["system_prompt"] = SYSTEM_PROMPTS[mode]
        await query.message.edit_text(
            f"✅ Mode changed to <b>{html.escape(mode)}</b>",
            parse_mode=ParseMode.HTML
        )
    elif data == "stats":
        await cmd_stats(update, context)
    # Add more callback handlers as needed

async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("🎨 Change Mode", callback_data="settings_mode"),
            InlineKeyboardButton("🔔 Notifications", callback_data="settings_notif"),
        ],
        [
            InlineKeyboardButton("🌐 Language", callback_data="settings_lang"),
            InlineKeyboardButton("📊 Data Export", callback_data="settings_export"),
        ],
        [InlineKeyboardButton("🔙 Back", callback_data="main_menu")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    settings_text = (
        "<b>⚙️ Settings</b>\n\n"
        f"• Current mode: {context.user_data.get('mode', 'default')}\n"
        f"• Notifications: {'On' if context.user_data.get('notifications', True) else 'Off'}\n"
        f"• Language: English\n"
    )
    
    if update.callback_query:
        await update.callback_query.message.edit_text(
            settings_text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            settings_text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )

# ========================== MAIN MESSAGE HANDLER ==========================
@track_usage("message")
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    
    cfg: Config = context.application.bot_data["config"]
    groq: EnhancedGroqClient = context.application.bot_data["groq_client"]
    limiter: EnhancedRateLimiter = context.application.bot_data["rate_limiter"]
    user_manager: UserManager = context.application.bot_data["user_manager"]
    conv_manager: ConversationManager = context.application.bot_data["conversation_manager"]
    analytics: Analytics = context.application.bot_data["analytics"]
    
    user_id = update.effective_user.id
    user = await user_manager.get_user(user_id)
    
    # Check if banned
    if user['role'] == UserRole.BANNED:
        await update.message.reply_text(
            "❌ You have been banned from using this bot.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Rate limiting
    allowed, retry_after = limiter.allow(user_id, user['role'])
    if not allowed:
        await update.message.reply_text(
            f"⏳ Rate limit reached. Try again in {retry_after}s",
            parse_mode=ParseMode.HTML
        )
        return
    
    user_text = update.message.text.strip()
    
    # Check for group mention requirement
    if cfg.enable_group_only_on_mention and update.message.chat.type in ("group", "supergroup"):
        bot_username = context.bot.username
        if f"@{bot_username}" not in user_text:
            return
    
    # Log message
    logger.info(f"User {user_id} ({update.effective_user.username}): {user_text[:100]}")
    
    # Update user stats
    await user_manager.update_user(user_id, {
        "message_count": user.get("message_count", 0) + 1,
        "last_message": datetime.now().isoformat()
    })
    
    # Track analytics
    await analytics.track_event("message", user_id, {
        "length": len(user_text),
        "chat_type": update.message.chat.type
    })
    
    # Get conversation history
    history = conv_manager.get_history(user_id)
    
    # Build messages for API
    system_prompt = context.user_data.get("system_prompt", SYSTEM_PROMPTS["default"])
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add summary if conversation is long
    if len(history) > 8:
        summary = await conv_manager.summarize_conversation(user_id, groq)
        if summary:
            messages.append({"role": "system", "content": f"Previous context: {summary}"})
    
    # Add recent history
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current message
    messages.append({"role": "user", "content": user_text})
    
    try:
        # Show typing indicator
        async with TypingAction(context, update.effective_chat.id):
            response = await groq.chat_completion(
                messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens
            )
    except Exception as e:
        logger.error(f"API Error for user {user_id}: {e}")
        await update.message.reply_text(
            "⚠️ <b>Error processing your request</b>\n"
            "Please try again later or use /reset to clear the conversation.",
            parse_mode=ParseMode.HTML
        )
        return
    
    # Format and sanitize response
    formatter = TextFormatter()
    formatted_response = formatter.markdown_to_html(response)
    safe_html = formatter.sanitize_html(formatted_response)
    
    # Split into chunks if needed
    chunks = formatter.split_message(safe_html)
    
    # Send response
    for i, chunk in enumerate(chunks):
        try:
            await update.message.reply_text(
                chunk,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=False
            )
        except Exception as e:
            logger.error(f"Failed to send chunk {i}: {e}")
            # Try sending without formatting as fallback
            await update.message.reply_text(chunk[:4000])
    
    # Update conversation history
    conv_manager.add_message(user_id, "user", user_text)
    conv_manager.add_message(user_id, "assistant", response)
    
    # Update token count (rough estimate)
    token_estimate = (len(user_text) + len(response)) // 4
    await user_manager.update_user(user_id, {
        "tokens_used": user.get("tokens_used", 0) + token_estimate
    })

# ========================== ERROR HANDLER ==========================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Exception while handling update {update}:", exc_info=context.error)
    
    if isinstance(update, Update) and update.effective_message:
        error_message = (
            "💥 <b>An unexpected error occurred</b>\n\n"
            "The error has been logged. Please try again or use /reset if the issue persists."
        )
        
        try:
            await update.effective_message.reply_text(
                error_message,
                parse_mode=ParseMode.HTML
            )
        except Exception:
            pass

# ========================== APPLICATION SETUP ==========================
async def post_init(application: Application) -> None:
    """Initialize bot after startup"""
    bot_commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("help", "Show help message"),
        BotCommand("mode", "Change AI personality"),
        BotCommand("reset", "Clear conversation"),
        BotCommand("stats", "View your statistics"),
        BotCommand("settings", "Configure preferences"),
        BotCommand("ping", "Check bot status"),
    ]
    await application.bot.set_my_commands(bot_commands)
    
    # Initialize cache connection
    cache: CacheManager = application.bot_data["cache_manager"]
    await cache.connect()
    
    logger.info("Bot initialization complete")

async def shutdown(application: Application) -> None:
    """Cleanup on shutdown"""
    # Close Groq client
    groq: EnhancedGroqClient = application.bot_data.get("groq_client")
    if groq:
        await groq.aclose()
    
    # Close cache connection
    cache: CacheManager = application.bot_data.get("cache_manager")
    if cache and cache.redis_client:
        await cache.redis_client.close()
    
    logger.info("Bot shutdown complete")

def build_application(cfg: Config) -> Application:
    """Build and configure the application"""
    builder = ApplicationBuilder().token(cfg.telegram_bot_token)
    
    # Configure for webhook if enabled
    if cfg.use_webhook and cfg.webhook_url:
        builder.url(cfg.webhook_url)
        builder.webhook_url(f"{cfg.webhook_url}/{cfg.telegram_bot_token}")
        builder.port(cfg.webhook_port)
    
    app = builder.build()
    
    # Initialize components
    cache_manager = CacheManager(cfg.redis_url)
    user_manager = UserManager(cache_manager, cfg.encryption_key)
    groq_client = EnhancedGroqClient(
        api_key=cfg.groq_api_key,
        model=cfg.groq_model,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        backoff_base=cfg.retry_backoff_base,
        cache_manager=cache_manager
    )
    
    # Store in bot_data
    app.bot_data["config"] = cfg
    app.bot_data["cache_manager"] = cache_manager
    app.bot_data["user_manager"] = user_manager
    app.bot_data["groq_client"] = groq_client
    app.bot_data["rate_limiter"] = EnhancedRateLimiter(
        cfg.rate_limit_window_sec,
        cfg.rate_limit_max_calls
    )
    app.bot_data["conversation_manager"] = ConversationManager(cfg.history_max_messages)
    app.bot_data["analytics"] = Analytics(cfg.enable_analytics)
    
    # Add handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("admin", cmd_admin))
    app.add_handler(CommandHandler("system", cmd_system))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    # Add initialization and shutdown
    app.post_init = post_init
    app.post_shutdown = shutdown
    
    return app

# ========================== MAIN ENTRY POINT ==========================
def main():
    """Main entry point"""
    try:
        cfg = load_config()
        logger.info(f"🚀 Starting Enhanced Theriyadhu Bot")
        logger.info(f"Model: {cfg.groq_model}")
        logger.info(f"Admin IDs: {cfg.admin_user_ids}")
        
        # Track startup time for uptime calculation
        get_uptime.start_time = time.time()
        
        application = build_application(cfg)
        
        if cfg.use_webhook and cfg.webhook_url:
            logger.info(f"Starting webhook on {cfg.webhook_url}:{cfg.webhook_port}")
            application.run_webhook(
                listen="0.0.0.0",
                port=cfg.webhook_port,
                url_path=cfg.telegram_bot_token,
                webhook_url=f"{cfg.webhook_url}/{cfg.telegram_bot_token}"
            )
        else:
            logger.info("Starting long polling mode")
            application.run_polling()
        
    except Exception as e:
        logger.exception(f"Failed to start bot: {e}")


if __name__ == "__main__":
    main()