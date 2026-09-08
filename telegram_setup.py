"""telegram_setup.py — Ayudante único para configurar las alertas de Telegram.

Lee TELEGRAM_BOT_TOKEN del .env, consulta getUpdates para descubrir tu chat_id
y, si TELEGRAM_CHAT_ID ya está puesto, envía un mensaje de prueba.

Uso:
    1) Crea el bot con @BotFather y pega el token en .env
    2) Abre tu bot en Telegram y pulsa Start (o envíale "hola")
    3) python telegram_setup.py
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
API = f"https://api.telegram.org/bot{TOKEN}"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _post(url: str, data: dict) -> dict:
    body = urllib.parse.urlencode(data).encode()
    with urllib.request.urlopen(url, data=body, timeout=15) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    if not TOKEN:
        print("❌ Falta TELEGRAM_BOT_TOKEN en el .env. Pega el token de @BotFather.")
        return

    print("Consultando getUpdates para descubrir tu chat_id...\n")
    data = _get(f"{API}/getUpdates")
    if not data.get("ok"):
        print(f"❌ Telegram respondió error: {data}")
        print("   Revisa que el token sea correcto.")
        return

    # Extrae los chat_id únicos de los mensajes recibidos.
    chats = {}
    for upd in data.get("result", []):
        msg = upd.get("message") or upd.get("channel_post") or {}
        chat = msg.get("chat") or {}
        if "id" in chat:
            name = chat.get("username") or chat.get("title") or chat.get("first_name") or "?"
            chats[chat["id"]] = name

    if not chats:
        print("⚠️  No hay mensajes todavía. Abre tu bot en Telegram, pulsa START")
        print("   o envíale 'hola', y vuelve a ejecutar este script.")
        return

    print("✅ chat_id(s) encontrados:")
    for cid, name in chats.items():
        print(f"   {cid}   ({name})")
    print("\nPega el valor en el .env como  TELEGRAM_CHAT_ID=<ese_numero>")

    # Si ya hay un chat configurado, manda un mensaje de prueba.
    if CHAT_ID:
        res = _post(f"{API}/sendMessage",
                    {"chat_id": CHAT_ID, "text": "✅ Funding Radar conectado a Telegram."})
        print("\nMensaje de prueba:", "enviado ✅" if res.get("ok") else f"error ❌ {res}")


if __name__ == "__main__":
    main()
