Conversational support agent deployed over WhatsApp, replacing a proprietary no-code platform that lacked visibility and control over its internal building blocks. 

The system acts as a first-response layer, resolving frequent technical queries from documentation and escalating to human operators when needed.

Stack: LangGraph, LLMs, Python.


-------------------------------------------
3. Start the Python API
In a terminal at the repo root:

chatbot-api
Wait for the line:
# INFO:     Application startup complete.

-------------------------------------------
4. Start the Baileys bridge
In a second terminal:

cd "adapters/whatsapp_baileys"
node src/index.js
You will see a QR code printed in the terminal — something like:

── Scan this QR code with WhatsApp to connect ──

█████████████████████
██ ▄▄▄▄▄ █▀▄ █ ▄▄▄▄▄ ██
...
You have ~60 seconds to scan it before it expires and a new one is generated automatically.

-------------------------------------------
5. Link your phone
On your phone: Open WhatsApp
Tap the three dots (Android) or Settings (iPhone) → Linked devices
Tap Link a device
Point your camera at the QR code in the terminal
You should see in the terminal:


INFO  WhatsApp connection established
The session is saved to adapters/whatsapp_baileys/auth_state/. Next time you run node src/index.js it reconnects automatically — no QR scan needed.

-------------------------------------------
6. Send a test message
From any WhatsApp account (not the linked one — that's the bot's number), send a message to the phone number you linked.

Watch both terminals:

Bridge terminal — shows the incoming message:


INFO  {"phone":"5491155551234","text":"hola"} "Incoming message"
Python terminal — shows graph processing:


INFO  sneil.whatsapp incoming message thread_id=wa_5491155551234
INFO  sneil.whatsapp reply ready duration_ms=4231 route=TPMS
Bridge terminal — confirms reply sent:


INFO  {"phone":"5491155551234","reply":"Hola! ¿En qué puedo ayudarte..."} "Reply sent"
The reply arrives in WhatsApp on your test phone.
