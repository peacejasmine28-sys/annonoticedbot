import asyncio
import io
import logging
import os
import zipfile
from datetime import datetime, timezone

import asyncpg
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message, CallbackQuery, BusinessConnection,
    InlineKeyboardMarkup, InlineKeyboardButton,
)
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID"))
BRAND = os.getenv("BRAND_NAME", "Annopow")
DATABASE_URL = os.getenv("DATABASE_URL")
MY_NAME = os.getenv("MY_NAME", "Anno Pow")  # your display name in Telegram exports

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "llama-3.3-70b-versatile"

bot = Bot(BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

logging.basicConfig(level=logging.INFO)

pool: asyncpg.Pool | None = None

# ---------- DATABASE ----------

async def db_init():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with pool.acquire() as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            tg_id BIGINT PRIMARY KEY,
            username TEXT, first_name TEXT, last_name TEXT,
            language TEXT,
            first_contact TIMESTAMPTZ,
            last_contact TIMESTAMPTZ,
            total_messages INT DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            tg_id BIGINT, role TEXT, text TEXT,
            ts TIMESTAMPTZ DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS training_data (
            id SERIAL PRIMARY KEY,
            question TEXT, answer TEXT, source TEXT,
            ts TIMESTAMPTZ DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY, value TEXT
        );
        CREATE TABLE IF NOT EXISTS pending (
            id SERIAL PRIMARY KEY,
            customer_id BIGINT, question TEXT,
            suggestion TEXT, admin_msg_id BIGINT
        );
        INSERT INTO settings (key, value) VALUES ('mode', 'manual')
        ON CONFLICT (key) DO NOTHING;
        """)
        await db.execute(
            "ALTER TABLE pending ADD COLUMN IF NOT EXISTS business_connection_id TEXT")

async def get_mode() -> str:
    async with pool.acquire() as db:
        row = await db.fetchrow("SELECT value FROM settings WHERE key='mode'")
        return row["value"] if row else "manual"

async def set_mode(mode: str):
    async with pool.acquire() as db:
        await db.execute("UPDATE settings SET value=$1 WHERE key='mode'", mode)

async def upsert_customer(m: Message):
    now = datetime.now(timezone.utc)
    u = m.from_user
    async with pool.acquire() as db:
        await db.execute("""
            INSERT INTO customers (tg_id, username, first_name, last_name,
                language, first_contact, last_contact, total_messages)
            VALUES ($1,$2,$3,$4,$5,$6,$6,1)
            ON CONFLICT (tg_id) DO UPDATE SET
                username=EXCLUDED.username,
                first_name=EXCLUDED.first_name,
                last_name=EXCLUDED.last_name,
                last_contact=EXCLUDED.last_contact,
                total_messages=customers.total_messages+1
        """, u.id, u.username, u.first_name, u.last_name,
             u.language_code, now)

async def log_message(tg_id: int, role: str, text: str):
    async with pool.acquire() as db:
        await db.execute(
            "INSERT INTO messages (tg_id, role, text) VALUES ($1,$2,$3)",
            tg_id, role, text)

async def save_training(q: str, a: str, source: str):
    async with pool.acquire() as db:
        await db.execute(
            "INSERT INTO training_data (question, answer, source) VALUES ($1,$2,$3)",
            q, a, source)

async def create_pending(customer_id: int, question: str, suggestion: str,
                         business_connection_id: str | None = None) -> int:
    async with pool.acquire() as db:
        return await db.fetchval(
            "INSERT INTO pending (customer_id, question, suggestion, business_connection_id) "
            "VALUES ($1,$2,$3,$4) RETURNING id",
            customer_id, question, suggestion, business_connection_id)

async def set_pending_admin_msg(pid: int, admin_msg_id: int):
    async with pool.acquire() as db:
        await db.execute("UPDATE pending SET admin_msg_id=$1 WHERE id=$2",
                         admin_msg_id, pid)

async def find_similar_training(question: str, limit: int = 6):
    words = [w.lower() for w in question.split() if len(w) > 3]
    async with pool.acquire() as db:
        rows = await db.fetch(
            "SELECT question, answer FROM training_data ORDER BY id DESC LIMIT 500")
    scored = []
    for r in rows:
        score = sum(1 for w in words if w in r["question"].lower())
        scored.append((score, r["question"], r["answer"]))
    scored.sort(reverse=True, key=lambda x: x[0])
    hits = [(q, a) for s, q, a in scored[:limit] if s > 0]
    return hits or [(q, a) for _, q, a in scored[:3]]

async def recent_history(tg_id: int, limit: int = 6):
    async with pool.acquire() as db:
        rows = await db.fetch(
            "SELECT role, text FROM messages WHERE tg_id=$1 ORDER BY id DESC LIMIT $2",
            tg_id, limit)
    return [(r["role"], r["text"]) for r in reversed(rows)]

# ---------- AI ----------

async def generate_reply(tg_id: int, question: str) -> str:
    examples = await find_similar_training(question)
    history = await recent_history(tg_id)

    example_block = "\n\n".join(
        f"Customer: {q}\nYou: {a}" for q, a in examples) or "None yet."
    history_block = "\n".join(f"{r}: {t}" for r, t in history)

    system = (
        f"You are the customer support agent for {BRAND}. "
        "Reply in the same style, tone and language as the example replies below.\n\n"
        "RULES:\n"
        "1. For questions about prices, delivery, refunds, policies or anything "
        "specific to the business: ONLY use facts found in the examples. If the "
        "examples don't cover it, say you'll confirm with the team - never guess.\n"
        "2. For general questions (how things work, product advice, greetings, "
        "small talk, technical explanations): use your own knowledge freely and "
        "answer helpfully.\n"
        "3. Be concise, human and friendly. Match the customer's language.\n\n"
        f"EXAMPLE PAST REPLIES:\n{example_block}"
    )
    user = f"Recent conversation:\n{history_block}\n\nCustomer's new message: {question}"

    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=400,
        temperature=0.6,
    )
    return resp.choices[0].message.content.strip()

# ---------- HELPERS ----------

def suggestion_kb(pending_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Send", callback_data=f"send:{pending_id}"),
        InlineKeyboardButton(text="🔄 Regenerate", callback_data=f"regen:{pending_id}"),
    ], [
        InlineKeyboardButton(text="✍️ I'll reply myself (reply to this msg)",
                             callback_data=f"noop:{pending_id}"),
    ]])

async def send_to_customer(customer_id: int, text: str,
                           business_connection_id: str | None):
    if business_connection_id:
        await bot.send_message(customer_id, text,
                               business_connection_id=business_connection_id)
    else:
        await bot.send_message(customer_id, text)

async def notify_with_suggestion(header: str, customer_id: int, question: str,
                                 business_connection_id: str | None):
    try:
        suggestion = await generate_reply(customer_id, question)
    except Exception as e:
        logging.error(f"AI error: {e}")
        await bot.send_message(ADMIN_ID, header + "\n\n⚠️ AI failed — reply manually.")
        return
    pid = await create_pending(customer_id, question, suggestion,
                               business_connection_id)
    sent = await bot.send_message(
        ADMIN_ID,
        header + f"\n\n🤖 Suggested reply:\n{suggestion}",
        reply_markup=suggestion_kb(pid))
    await set_pending_admin_msg(pid, sent.message_id)

# ---------- CHAT EXPORT IMPORT (send a ZIP of the 'chats' folder) ----------

def extract_pairs_from_html(html: str, my_name: str):
    """Yield (question, answer) pairs from one messages.html content."""
    soup = BeautifulSoup(html, "html.parser")
    last_sender = None
    last_q = None
    for msg in soup.select("div.message.default"):
        name_el = msg.select_one("div.from_name")
        if name_el:
            last_sender = name_el.get_text(strip=True)
        text_el = msg.select_one("div.text")
        if not text_el or not last_sender:
            continue
        text = text_el.get_text(" ", strip=True)
        if not text:
            continue
        if last_sender != my_name:
            last_q = text
        elif last_q:
            yield last_q[:1000], text[:1000]
            last_q = None

@router.message(F.from_user.id == ADMIN_ID, F.document)
async def on_admin_document(m: Message):
    doc = m.document
    if not doc.file_name or not doc.file_name.lower().endswith(".zip"):
        return await m.answer("Send me a ZIP file of your export's 'chats' folder.")
    if doc.file_size and doc.file_size > 19 * 1024 * 1024:
        return await m.answer(
            "⚠️ File too big — Telegram only lets bots download up to ~20 MB.\n"
            "Re-export WITHOUT media (untick photos/videos/voice/stickers), "
            "or split the chats folder into smaller ZIPs.")
    status = await m.answer("⏳ Downloading ZIP...")
    try:
        buf = io.BytesIO()
        await bot.download(doc, destination=buf)
        buf.seek(0)
    except Exception as e:
        logging.error(f"Download failed: {e}")
        return await status.edit_text(f"⚠️ Download failed: {type(e).__name__}")

    await status.edit_text("⏳ Importing conversations...")
    pairs = 0
    files_done = 0
    try:
        with zipfile.ZipFile(buf) as zf:
            html_names = [n for n in zf.namelist()
                          if n.lower().endswith(".html") and "messages" in n.lower()]
            for name in html_names:
                try:
                    html = zf.read(name).decode("utf-8", errors="ignore")
                    for q, a in extract_pairs_from_html(html, MY_NAME):
                        await save_training(q, a, "zip_import")
                        pairs += 1
                except Exception as e:
                    logging.error(f"Skipped {name}: {e}")
                files_done += 1
                if files_done % 25 == 0:
                    try:
                        await status.edit_text(
                            f"⏳ Importing... {files_done}/{len(html_names)} files, "
                            f"{pairs} pairs so far")
                    except Exception:
                        pass
    except zipfile.BadZipFile:
        return await status.edit_text("⚠️ That file isn't a valid ZIP.")

    await status.edit_text(
        f"✅ Import complete!\nFiles processed: {files_done}\n"
        f"Q&A pairs learned: {pairs}\n\nCheck /stats to confirm.")

# ---------- BUSINESS CONNECTION DIAGNOSTIC ----------

@router.business_connection()
async def on_business_connection(bc: BusinessConnection):
    rights = getattr(bc, "rights", None)
    if rights is not None:
        rights_text = "\n".join(
            f"• {k}: {'✅' if v else '❌'}"
            for k, v in rights.model_dump().items() if v is not None)
    else:
        can_reply = getattr(bc, "can_reply", None)
        rights_text = f"• can_reply: {'✅' if can_reply else '❌'}"
    status = "🟢 ENABLED" if bc.is_enabled else "🔴 DISABLED"
    info = (f"🔌 Business connection update\n"
            f"Status: {status}\n"
            f"Connection ID: {bc.id}\n"
            f"Account: {bc.user.first_name} (ID {bc.user_id})\n\n"
            f"Granted rights:\n{rights_text}")
    logging.info(info)
    try:
        await bot.send_message(ADMIN_ID, info)
    except Exception as e:
        logging.error(f"Couldn't notify admin about connection: {e}")

# ---------- ADMIN COMMANDS ----------

def is_admin(m: Message) -> bool:
    return m.from_user.id == ADMIN_ID

@router.message(Command("auto"))
async def cmd_auto(m: Message):
    if not is_admin(m): return
    await set_mode("auto")
    await m.answer("🤖 Auto mode ON — bot replies to customers automatically.")

@router.message(Command("manual"))
async def cmd_manual(m: Message):
    if not is_admin(m): return
    await set_mode("manual")
    await m.answer("✍️ Manual mode ON — messages come to you with AI suggestions.")

@router.message(Command("stats"))
async def cmd_stats(m: Message):
    if not is_admin(m): return
    async with pool.acquire() as db:
        c = await db.fetchval("SELECT COUNT(*) FROM customers")
        msgs = await db.fetchval("SELECT COUNT(*) FROM messages")
        tr = await db.fetchval("SELECT COUNT(*) FROM training_data")
    mode = await get_mode()
    await m.answer(
        f"📊 Stats\nMode: {mode}\nCustomers: {c}\n"
        f"Messages: {msgs}\nTraining pairs: {tr}")

@router.message(Command("customers"))
async def cmd_customers(m: Message):
    if not is_admin(m): return
    async with pool.acquire() as db:
        rows = await db.fetch(
            "SELECT tg_id, username, first_name, total_messages "
            "FROM customers ORDER BY last_contact DESC LIMIT 20")
    if not rows:
        return await m.answer("No customers yet.")
    lines = [f"• {r['first_name'] or ''} @{r['username'] or '—'} "
             f"(ID {r['tg_id']}) — {r['total_messages']} msgs" for r in rows]
    await m.answer("👥 Last 20 customers:\n" + "\n".join(lines))

@router.message(Command("learn"))
async def cmd_learn(m: Message):
    if not is_admin(m): return
    payload = m.text.replace("/learn", "", 1).strip()
    if "|" not in payload:
        return await m.answer("Format: /learn Question | Answer")
    q, a = [p.strip() for p in payload.split("|", 1)]
    await save_training(q, a, "manual_learn")
    await m.answer("✅ Learned.")

@router.message(Command("broadcast"))
async def cmd_broadcast(m: Message):
    if not is_admin(m): return
    text = m.text.replace("/broadcast", "", 1).strip()
    if not text:
        return await m.answer("Format: /broadcast Your message here")
    async with pool.acquire() as db:
        rows = await db.fetch("SELECT tg_id FROM customers")
    sent = 0
    for r in rows:
        try:
            await bot.send_message(r["tg_id"], text)
            sent += 1
            await asyncio.sleep(0.05)
        except Exception:
            pass
    await m.answer(f"📢 Broadcast sent to {sent} customers.")

# ---------- CALLBACKS ----------

@router.callback_query(F.data.startswith(("send:", "regen:", "noop:")))
async def on_button(cb: CallbackQuery):
    if cb.from_user.id != ADMIN_ID:
        return await cb.answer("Admins only.")
    action, pid = cb.data.split(":")
    pid = int(pid)
    async with pool.acquire() as db:
        row = await db.fetchrow(
            "SELECT customer_id, question, suggestion, business_connection_id "
            "FROM pending WHERE id=$1", pid)
    if not row:
        return await cb.answer("Expired.")
    customer_id = row["customer_id"]
    question = row["question"]
    suggestion = row["suggestion"]
    bcid = row["business_connection_id"]

    if action == "send":
        try:
            await send_to_customer(customer_id, suggestion, bcid)
        except Exception as e:
            logging.error(f"Send failed: {e}")
            return await cb.answer(f"Send failed: {type(e).__name__}", show_alert=True)
        await log_message(customer_id, "admin", suggestion)
        await save_training(question, suggestion, "approved_ai")
        await cb.message.edit_text(cb.message.text + "\n\n✅ Sent.")
        await cb.answer("Sent!")
    elif action == "regen":
        await cb.answer("Regenerating…")
        new = await generate_reply(customer_id, question)
        async with pool.acquire() as db:
            await db.execute("UPDATE pending SET suggestion=$1 WHERE id=$2", new, pid)
        await cb.message.edit_text(
            f"💬 Customer {customer_id} asked:\n{question}\n\n"
            f"🤖 New suggestion:\n{new}",
            reply_markup=suggestion_kb(pid))
    else:
        await cb.answer("Reply to this message with your own answer.")

# ---------- ADMIN MANUAL REPLY ----------

@router.message(F.from_user.id == ADMIN_ID, F.reply_to_message, F.text)
async def admin_reply(m: Message):
    async with pool.acquire() as db:
        row = await db.fetchrow(
            "SELECT customer_id, question, business_connection_id "
            "FROM pending WHERE admin_msg_id=$1",
            m.reply_to_message.message_id)
    if not row:
        return
    try:
        await send_to_customer(row["customer_id"], m.text,
                               row["business_connection_id"])
    except Exception as e:
        logging.error(f"Send failed: {e}")
        return await m.answer(f"⚠️ Send failed: {type(e).__name__}")
    await log_message(row["customer_id"], "admin", m.text)
    await save_training(row["question"], m.text, "admin_manual")
    await m.answer("✅ Sent & learned.")

# ---------- SECRETARY MODE: DMs TO YOUR PERSONAL ACCOUNT ----------

@router.business_message(F.text)
async def on_business_message(m: Message):
    if m.from_user.id == ADMIN_ID:
        return
    await upsert_customer(m)
    await log_message(m.from_user.id, "customer", m.text)

    u = m.from_user
    bcid = m.business_connection_id
    header = (f"💬 DM to your personal account\n"
              f"From: {u.first_name or ''} @{u.username or '—'} (ID {u.id})\n\n"
              f"«{m.text}»")

    mode = await get_mode()
    if mode != "auto":
        await notify_with_suggestion(header, u.id, m.text, bcid)
        return

    try:
        reply = await generate_reply(u.id, m.text)
    except Exception as e:
        logging.error(f"AI error: {e}")
        await bot.send_message(ADMIN_ID, header + "\n\n⚠️ AI failed — reply manually.")
        return

    try:
        await m.answer(reply)
    except Exception as e:
        logging.error(f"Business send failed: {e}")
        pid = await create_pending(u.id, m.text, reply, bcid)
        sent = await bot.send_message(
            ADMIN_ID,
            header + f"\n\n⚠️ Couldn't auto-send. Suggested reply:\n{reply}",
            reply_markup=suggestion_kb(pid))
        await set_pending_admin_msg(pid, sent.message_id)
        return

    await log_message(u.id, "bot", reply)
    await bot.send_message(ADMIN_ID, header + f"\n\n🤖 Auto-replied in your DM:\n{reply}")

# ---------- BOT CHAT: ADMIN ONLY ----------

@router.message(CommandStart())
async def cmd_start(m: Message):
    if is_admin(m):
        return await m.answer(
            "👋 Admin panel ready.\n"
            "/auto /manual /stats /customers /broadcast /learn Q | A\n"
            "📦 Send me a ZIP of your export's 'chats' folder to import history.")
    await m.answer(f"👋 For {BRAND} support, please message @annopow directly.")

@router.message(F.text)
async def on_other_message(m: Message):
    if is_admin(m):
        return
    await m.answer(f"👋 For {BRAND} support, please message @annopow directly.")

# ---------- MAIN ----------

async def main():
    await db_init()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
