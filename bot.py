import asyncio
import logging
import os
from datetime import datetime, timezone

import asyncpg
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID"))
BRAND = os.getenv("BRAND_NAME", "Annopow")
DATABASE_URL = os.getenv("DATABASE_URL")

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
            total_messages
