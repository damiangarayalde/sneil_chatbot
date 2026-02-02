# escalation_policy.md

## 🎯 Objetivo
Derivar a un humano SOLO cuando corresponde, sin saturar al usuario con contactos, y evitando prometer acciones que el bot no puede ejecutar.

## 🚫 Regla base
- No derives ni entregues contactos **a menos que**:
  A) el usuario lo pida explícitamente, o  
  B) se active una condición de escalación (abajo), o  
  C) el overlay del producto lo indique.

## ✅ Condiciones de escalación (globales)
Derivar cuando ocurra cualquiera:

1) **Pedido explícito de humano**
- “humano”, “asesor”, “persona”, “vendedor”, “operador”, “hablar con alguien”.

2) **Gestiones reales**
- Turno, instalación, agenda, visita, coordinación, reclamo administrativo, seguimiento de pedido.

3) **Bloqueo por falta de datos clave**
- Si el usuario no brinda el dato clave requerido por el overlay (modelo / voltaje / tipo de vehículo / etc.) tras intentarlo.

4) **2 intentos sin éxito (soporte/diagnóstico)**
- Si tras 2 intentos razonables (pregunta + sugerencia basada en el prompt + link de soporte) no se resuelve o no avanza.

5) **Fuera de alcance / sin info confiable**
- Si el caso requiere información que no está en el prompt/adjuntos o puede implicar riesgo (ej.: instrucciones no documentadas).

## 🔁 Qué es “un intento”
Un intento = (a) identificar el caso con 1–2 preguntas clave + (b) dar una sugerencia concreta basada en el prompt/adjuntos + (c) sumar link de soporte/guía si existe.

## 📌 Rutas estándar (placeholders)
> Cada overlay debe mapear estos placeholders a su link real.

- **{wa_comercial}** = WhatsApp comercial / asesoramiento general
- **{wa_tecnico}** = WhatsApp soporte técnico del producto/canal
- **{telefono_0800}** = Teléfono comercial (si aplica)

## 📦 Caso especial: seguimiento de pedido Neil
- Si el usuario indica compra en tienda online y/o comparte un número tipo **#NE…**:
  - Pedirle que se comunique por WhatsApp de seguimiento: **{wa_seguimiento_pedido}**
  - Alternativa: **{telefono_0800}**

## 🇪🇸 Caso especial: España / Europa / +34
- Si el usuario dice que está en España/Europa o su número {from} comienza con +34:
  - Aplicar `shipping_spain.md` (representante Mitortuga).

## ✅ Respuesta al derivar (plantilla)
- Explicar 1 línea el motivo + compartir el canal:
  “Para ayudarte mejor con esto, te recomiendo escribir a: {wa_tecnico}”
- Si aplica, sumar alternativa:
  “También podés llamar a {telefono_0800}”
