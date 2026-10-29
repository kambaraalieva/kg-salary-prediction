"""
Collect posts from public Telegram job channels (official Telegram API via Telethon).

Setup (once):
  1. my.telegram.org -> API development tools -> create an app -> get api_id and api_hash
  2. Put them in the .env file in the repo root (never commit it):
         TG_API_ID=1234567
         TG_API_HASH=abcdef0123456789...
         TG_CHANNELS=channel_one,channel_two,https://t.me/channel_three
  3. pip install telethon python-dotenv

Run from the repo root:
    python src/collect_telegram.py                  # posts since 2026-01-01
    python src/collect_telegram.py --since 2025-10-01

The first run asks for your phone number and the login code Telegram sends you.
After that a session file (kg_jobs.session) keeps you logged in; it is in .gitignore.

Output: data/telegram/raw_posts.jsonl, one post per line.
Safe to re-run: posts already saved are skipped, new ones are added.

Privacy: only the post text, date, views and link are saved. Author names, usernames
and phone numbers are NOT collected; phone numbers in the text are masked.
"""

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError

load_dotenv()
OUT = Path("data/telegram/raw_posts.jsonl")
PHONE = re.compile(r"(\+?996|\b0)[\s\-()]*\d{3}[\s\-()]*\d{2,3}[\s\-()]*\d{2,3}[\s\-()]*\d{0,3}")
MENTION = re.compile(r"@[A-Za-z0-9_]{4,}")


def channel_name(raw):
    """Accepts t.me/name, @name, name, or a web.telegram.org link with a numeric id (#-100...)."""
    raw = raw.strip()
    if not raw:
        return ""
    last = raw.rstrip("/").split("/")[-1]
    return last.split("#")[-1].lstrip("@")


def resolve(name):
    """Numeric ids (-100...) are passed to Telegram as numbers, names as text."""
    return int(name) if name.lstrip("-").isdigit() else name


def mask(text):
    text = PHONE.sub("[телефон]", text)
    return MENTION.sub("[контакт]", text)


def load_seen():
    seen = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                seen.add((r["channel"], r["msg_id"]))
    return seen


async def collect(since):
    api_id, api_hash = os.getenv("TG_API_ID"), os.getenv("TG_API_HASH")
    channels = [channel_name(c) for c in os.getenv("TG_CHANNELS", "").split(",") if c.strip()]
    if not (api_id and api_hash and channels):
        raise SystemExit("Fill TG_API_ID, TG_API_HASH and TG_CHANNELS in .env (see the top of this file).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen()
    print(f"Already saved: {len(seen)} posts")

    async with TelegramClient("kg_jobs", int(api_id), api_hash) as client:
        await client.get_dialogs()          # lets Telegram find channels you joined by their numeric id
        with OUT.open("a", encoding="utf-8") as f:
            for raw in channels:
                added = 0
                try:
                    entity = await client.get_entity(resolve(raw))
                    ch = entity.username or str(entity.id)
                    title = getattr(entity, "title", ch)
                    link_base = f"https://t.me/{entity.username}" if entity.username else f"https://t.me/c/{entity.id}"
                    print(f"{title} ({ch}): collecting...")
                except Exception as e:                       # wrong name, not joined, private channel
                    print(f"  {raw}: skipped ({type(e).__name__}: {e})")
                    continue
                try:
                    async for m in client.iter_messages(entity, reverse=True, offset_date=since):
                        if not m.message or (ch, m.id) in seen:
                            continue
                        f.write(json.dumps({
                            "channel": ch,
                            "msg_id": m.id,
                            "date": m.date.isoformat(),
                            "views": m.views,
                            "channel_title": title,
                            "link": f"{link_base}/{m.id}",
                            "text": mask(m.message),
                        }, ensure_ascii=False) + "\n")
                        added += 1
                        if added % 500 == 0:
                            f.flush(); print(f"  {ch}: {added}")
                except FloodWaitError as e:
                    print(f"  {ch}: Telegram asks to wait {e.seconds} s. Re-run later; progress is saved.")
                except Exception as e:                       # wrong name, private channel, etc.
                    print(f"  {ch}: skipped ({type(e).__name__}: {e})")
                print(f"{ch}: +{added} posts")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-01-01", help="collect posts from this date (YYYY-MM-DD)")
    a = ap.parse_args()
    since = datetime.strptime(a.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    asyncio.run(collect(since))
