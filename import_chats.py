import json, os, sys, asyncio
from datetime import datetime, timezone

import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def main():
    my_name = sys.argv[2] if len(sys.argv) > 2 else "James"
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    chats = data.get("chats", {}).get("list", [data]) if "chats" in data else [data]

    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    pairs = 0
    for chat in chats:
        last_q = None
        for msg in chat.get("messages", []):
            text = msg.get("text")
            if isinstance(text, list):
                text = "".join(p if isinstance(p, str) else p.get("text", "")
                               for p in text)
            if not text or msg.get("type") != "message":
                continue
            if (msg.get("from") or "") != my_name:
                last_q = text
            elif last_q:
                await conn.execute(
                    "INSERT INTO training_data (question, answer, source) "
                    "VALUES ($1,$2,$3)", last_q[:1000], text[:1000], "import")
                pairs += 1
                last_q = None
    await conn.close()
    print(f"Imported {pairs} Q&A pairs.")

asyncio.run(main())
