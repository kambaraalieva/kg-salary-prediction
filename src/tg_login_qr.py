"""
Log in to Telegram by scanning a QR code (no SMS or login code needed).

    pip install telethon python-dotenv qrcode
    python src/tg_login_qr.py

On your phone: Telegram -> Settings -> Devices -> Link Desktop Device -> scan the QR code
shown in the Terminal. If you have a cloud password (two-step verification), the script asks for it.

After a successful login the session file kg_jobs.session is saved, and
src/collect_telegram.py uses it without asking anything.
"""

import asyncio
import getpass
import os

import qrcode
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

load_dotenv()


def show(url):
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


async def main():
    api_id, api_hash = os.getenv("TG_API_ID"), os.getenv("TG_API_HASH")
    if not (api_id and api_hash):
        raise SystemExit("Fill TG_API_ID and TG_API_HASH in .env first.")
    client = TelegramClient("kg_jobs", int(api_id), api_hash)
    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"Already logged in as {me.first_name}. Now run: python src/collect_telegram.py")
        await client.disconnect()
        return

    qr_login = await client.qr_login()
    for attempt in range(5):
        print("\nScan this QR code: Telegram -> Settings -> Devices -> Link Desktop Device\n")
        show(qr_login.url)
        try:
            await qr_login.wait(timeout=60)
            break
        except SessionPasswordNeededError:
            await client.sign_in(password=getpass.getpass("Two-step verification password: "))
            break
        except asyncio.TimeoutError:
            print("QR code expired, showing a new one...")
            await qr_login.recreate()
    else:
        raise SystemExit("Not scanned in time. Run the script again.")

    me = await client.get_me()
    print(f"\nLogged in as {me.first_name}. Now run: python src/collect_telegram.py")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
