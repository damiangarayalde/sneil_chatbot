
# Treat these as "low information" replies that should NOT lock a route
import re

LOW_INFO_MSGS = {
    "hola", "buenas", "buen día", "buen dia", "buenas tardes", "buenas noches",
    "ok", "oka", "dale", "listo", "joya", "perfecto", "bien",
    "si", "sí", "no", "seguro", "claro", "gracias", "👍", "👌"
}


TROUBLE_KEYWORDS = [
    # troubleshooting / failure
    "no funciona", "no anda", "no prende", "no enciende", "no carga", "no conecta",
    "no enfría", "no enfria", "no calienta", "no responde", "se corta", "se apaga",
    "error", "falla", "problema", "ruido", "vibra", "pierde", "pierde presión", "pierde presion",
    "alarma", "sensor", "emparejar", "bluetooth", "presión", "presion", "temperatura",
    # how-to / setup
    "cómo", "como", "instalar", "configurar", "calibrar", "reset", "reiniciar",
    "paso a paso", "manual", "instrucciones", "setear", "programar"
]

_ACK_RE = re.compile(
    r"^(si|sí|no|ok|dale|listo|perfecto|gracias)[.!? ]*$", re.IGNORECASE)


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def is_low_info(text: str) -> bool:
    t = normalize(text)
    if not t:
        return True
    if len(t) <= 3:
        return True
    if t in LOW_INFO_MSGS:
        return True
    # if _ACK_RE.fullmatch(t):
    #     return True
    return False


def should_retrieve(user_text: str) -> bool:
    t = normalize(user_text)
    if is_low_info(t):
        return False

    # retrieval when it looks like a question or an issue/how-to
    return ("?" in t) or any(k in t for k in TROUBLE_KEYWORDS)
