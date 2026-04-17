/**
 * WhatsApp Web → Python chatbot bridge
 *
 * Uses @whiskeysockets/baileys (WebSocket-based WA Web client, no browser needed).
 * On startup: displays a QR code in the terminal. Scan it once with WhatsApp to link.
 * Session is saved to ./auth_state/ so subsequent restarts skip the QR scan.
 *
 * Flow (per incoming message):
 *   WhatsApp → Baileys → POST /whatsapp/incoming (Python FastAPI)
 *                      ← { response: "bot reply" }
 *   Baileys → sendMessage(phone, reply) → WhatsApp
 *
 * When the Meta Business API is ready, this bridge is replaced by Meta's webhook
 * delivery. The Python /whatsapp/incoming endpoint stays the same — only the
 * payload parser changes on the Python side.
 */

import 'dotenv/config';
import makeWASocket, {
  useMultiFileAuthState,
  DisconnectReason,
  isJidBroadcast,
  fetchLatestBaileysVersion,
} from '@whiskeysockets/baileys';
import qrcode from 'qrcode-terminal';
import axios from 'axios';
import pino from 'pino';

// ── Config ────────────────────────────────────────────────────────────────────

const PYTHON_API_URL  = process.env.PYTHON_API_URL   || 'http://localhost:8000';
const BRIDGE_SECRET   = process.env.WA_BRIDGE_SECRET  || '';
const AUTH_STATE_DIR  = process.env.AUTH_STATE_DIR    || './auth_state';
const REQUEST_TIMEOUT = parseInt(process.env.REQUEST_TIMEOUT_MS || '65000', 10); // 65s > Python's graph timeout

const logger = pino({ level: process.env.LOG_LEVEL || 'info' });

if (!BRIDGE_SECRET) {
  logger.warn('WA_BRIDGE_SECRET is not set — requests to Python are unauthenticated');
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function startBridge() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_STATE_DIR);

  // Fetch the current WhatsApp Web version — required or the server returns 405.
  const { version } = await fetchLatestBaileysVersion();
  logger.info({ version }, 'Using WhatsApp Web version');

  const sock = makeWASocket({
    version,
    auth: state,
    // Suppress Baileys' internal logger output (we use pino)
    logger: pino({ level: 'silent' }),
    // Receive full message objects (required to read text bodies)
    getMessage: async () => undefined,
  });

  // Persist credentials whenever they change (after QR scan, key rotation, etc.)
  sock.ev.on('creds.update', saveCreds);

  // ── Connection lifecycle ───────────────────────────────────────────────────

  sock.ev.on('connection.update', ({ connection, lastDisconnect, qr }) => {
    if (qr) {
      console.log('\n── Scan this QR code with WhatsApp to connect ──\n');
      qrcode.generate(qr, { small: true });
    }

    if (connection === 'open') {
      logger.info('WhatsApp connection established');
    }

    if (connection === 'close') {
      const statusCode = lastDisconnect?.error?.output?.statusCode;
      const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
      logger.warn({ statusCode, shouldReconnect }, 'Connection closed');
      if (shouldReconnect) {
        logger.info('Reconnecting in 5s…');
        setTimeout(startBridge, 5000);
      } else {
        logger.error('Logged out. Delete auth_state/ and restart to re-scan QR.');
      }
    }
  });

  // ── Incoming messages ─────────────────────────────────────────────────────

  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    // 'notify' = new incoming messages; 'append' = history sync — skip that
    if (type !== 'notify') return;

    for (const msg of messages) {
      // Skip: own messages, broadcast/status updates, messages without text
      if (msg.key.fromMe) continue;
      if (isJidBroadcast(msg.key.remoteJid)) continue;

      const text = extractText(msg);
      if (!text) continue;

      // The JID (e.g. "5491155551234@s.whatsapp.net") — strip the suffix for the
      // phone number we pass to Python. Python prefixes with "wa_" for thread_id.
      const jid   = msg.key.remoteJid;
      const phone = jid.split('@')[0];

      logger.info({ phone, text: text.slice(0, 80) }, 'Incoming message');

      try {
        const reply = await forwardToPython(phone, text, msg.messageTimestamp);
        if (reply) {
          await sock.sendMessage(jid, { text: reply });
          logger.info({ phone, reply: reply.slice(0, 80) }, 'Reply sent');
        }
      } catch (err) {
        logger.error({ phone, err: err.message }, 'Failed to process message');
        // Optionally send a fallback reply so the user isn't left hanging
        await sock.sendMessage(jid, {
          text: 'Lo siento, ocurrió un error al procesar tu mensaje. Por favor intentá de nuevo.',
        });
      }
    }
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Extract the plain text from a Baileys message object.
 * Handles regular text, extended text (links), and ephemeral messages.
 */
function extractText(msg) {
  const m = msg.message;
  if (!m) return null;
  return (
    m.conversation ||
    m.extendedTextMessage?.text ||
    m.ephemeralMessage?.message?.conversation ||
    m.ephemeralMessage?.message?.extendedTextMessage?.text ||
    null
  );
}

/**
 * POST the message to Python and return the bot's reply text.
 *
 * Payload sent to Python:
 *   { "from": "5491155551234", "text": "...", "timestamp": 1234567890 }
 *
 * NOTE: When the Meta Business API replaces this bridge, Meta will POST a
 * different payload format directly. The Python adapter will parse Meta's
 * format instead — this bridge and its payload shape become irrelevant.
 */
async function forwardToPython(phone, text, timestamp) {
  const url = `${PYTHON_API_URL}/whatsapp/incoming`;

  const headers = { 'Content-Type': 'application/json' };
  if (BRIDGE_SECRET) headers['X-Bridge-Secret'] = BRIDGE_SECRET;

  const { data } = await axios.post(
    url,
    { from: phone, text, timestamp: Number(timestamp) || Math.floor(Date.now() / 1000) },
    { timeout: REQUEST_TIMEOUT, headers }
  );

  return data?.response || null;
}

// ── Entry point ───────────────────────────────────────────────────────────────

startBridge().catch((err) => {
  logger.error({ err: err.message }, 'Fatal error starting bridge');
  process.exit(1);
});
