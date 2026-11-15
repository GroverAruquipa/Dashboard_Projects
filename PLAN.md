# Plan de portafolio: 3 proyectos

Cada proyecto es la plantilla del servicio que se vende. Cada uno se entrega en tres formatos:
1. **Código limpio** (carpeta propia en este repo, README que funciona como página de venta).
2. **Vídeo demo de 2 min** (problema → demo en vivo → resultado en números → llamada a la acción).
3. **PDF de una página** (problema, solución, diagrama, métricas, "qué te entrego en X días").

Regla de salida: **con el vídeo del proyecto 1 ya se empieza a proponer**. El mínimo para salir a vender es 1 + 2.

---

## Estructura del repo

```
dashboard_projects/
├── PLAN.md                     ← este archivo
├── README.md                   ← se reescribe como índice del portafolio
├── legacy/                     ← dashboards 2021 (project_1, MPU6050, KTH), sin tocar
├── 01-leads-automation/
├── 02-defect-detection-mvtec/
└── 03-rag-technical-docs/
```

> Mover lo viejo a `legacy/` se hace solo con tu visto bueno.

---

## Proyecto 1 — Automatización de leads (1–2 tardes)

**Negocio ficticio:** "Clínica Dental Sonrisa" (o inmobiliaria; elegir uno y no cambiar).

**Flujo:**
```
Formulario web (Tally / HTML propio)  ─┐
WhatsApp (Twilio sandbox / Meta Cloud) ─┼─▶ n8n Webhook
                                        │      ├─ Normalizar + validar (teléfono, email, duplicados)
                                        │      ├─ Clasificar lead con LLM (urgencia, servicio, presupuesto)
                                        │      ├─ Google Sheets: fila nueva en "CRM" (estado = Nuevo)
                                        │      ├─ LLM redacta respuesta personalizada → email/WhatsApp al lead
                                        │      └─ Aviso al dueño (Telegram/email) con resumen + botón
                                        └─ Seguimiento: cron n8n cada 24 h → leads sin respuesta → recordatorio
```

**Herramientas:** n8n (Docker local o n8n Cloud trial), Google Sheets, Claude API (modelo pequeño tipo Haiku para clasificar/redactar), Gmail o Telegram Bot para avisos, Tally o formulario HTML estático.

**Entregables en código:**
- `workflow.json` (exportable/importable en n8n)
- `docker-compose.yml` para levantar n8n
- `form/index.html` (landing + formulario del negocio ficticio)
- `prompts/` (clasificación y respuesta, versionados)
- `sheets_template.csv` (columnas del CRM)
- `test_leads.json` + script que dispara 20 leads de prueba

**Métricas para el PDF:** tiempo de respuesta al lead (horas → < 60 s), % leads clasificados correctamente sobre los 20 de prueba, coste por lead en API.

**Definición de terminado:** un lead enviado por formulario aparece en Sheets, recibe respuesta personalizada y el dueño recibe aviso, todo en < 1 min, grabado en vídeo.

**Qué hago yo / qué haces tú:** yo genero workflow, prompts, formulario, docker-compose y scripts de prueba. Tú pones las credenciales (Google OAuth, API key, Telegram/WhatsApp) en tu n8n y grabas el vídeo.

---

## Proyecto 2 — Detección de defectos sobre MVTec AD (un fin de semana)

**Dataset:** MVTec AD (15 categorías). Empezar con 3 representativas: `bottle`, `metal_nut`, `screw` (la difícil); ampliar si hay tiempo.

**Modelos:** Anomalib — PatchCore (referencia fuerte) y EfficientAD o PaDiM (rápido) para comparar precisión vs velocidad.

**Métricas medidas (tabla por categoría):**
- Image AUROC, Pixel AUROC, PRO
- Precision / Recall / F1 en el umbral operativo elegido (explicar cómo se elige el umbral)
- Latencia por imagen (ms) en CPU y en GPU, imágenes/segundo
- Tamaño del modelo

**Herramientas:** Python 3.11, anomalib, PyTorch, OpenCV, Gradio (demo: subes imagen → mapa de calor + veredicto OK/NOK), ONNX/OpenVINO para la cifra de velocidad en CPU. GPU: Google Colab o Kaggle (gratis) para entrenar.

**Entregables en código:**
- `train.py`, `evaluate.py`, `benchmark_latency.py` con config YAML
- `results/metrics.csv` + figuras (curvas ROC, mapas de calor, ejemplos de falsos positivos/negativos)
- `app.py` (Gradio) — la demo del vídeo
- `docs/onboarding_cliente.md` — **la plantilla de la oferta "prototipo sobre TUS imágenes en 7 días"**: qué imágenes pides (n.º de OK, n.º de NOK, iluminación), qué entregas el día 7

**Definición de terminado:** tabla de métricas reproducible con un comando + demo Gradio + vídeo mostrando un defecto detectado en tiempo real.

---

## Proyecto 3 — RAG sobre documentación técnica (1–2 tardes)

**Corpus:** 3–5 manuales públicos en PDF (p. ej. manual de un variador de frecuencia, un PLC, un brazo robótico). Tablas y diagramas incluidos a propósito: ahí está el criterio de ingeniería.

**Pipeline:** extracción PDF (PyMuPDF; tablas con pdfplumber) → chunking por sección con metadatos (doc, página, sección) → embeddings → búsqueda híbrida (BM25 + vectorial) + re-ranking → LLM responde **solo con citas `[doc, pág.]`** y dice "no está en el manual" cuando no lo encuentra.

**Herramientas:** Python, Qdrant o Chroma, Claude API (modelo mediano tipo Sonnet), FastAPI + interfaz simple (Gradio/Streamlit). Reutiliza de P1: n8n como canal opcional (pregunta por WhatsApp/Telegram → RAG).

**Evaluación (lo que diferencia de una agencia genérica):** set de 30 preguntas con respuesta y página esperadas; medir recall@5 de recuperación, % respuestas correctas, % citas correctas, % de rechazos correctos ante preguntas sin respuesta.

**Definición de terminado:** demo respondiendo con citas clicables a la página del PDF + tabla de evaluación en el PDF de una página.

---

## Calendario propuesto

| Semana | Trabajo | Salida |
|---|---|---|
| 1 (2 tardes) | Proyecto 1 completo + vídeo | **Empezar a proponer** (servicio de automatización) |
| 1 (fin de semana) | Proyecto 2 entrenamiento + métricas + demo | Vídeo + PDF → proponer en visión ($50–80/h) |
| 2 (2 tardes) | Proyecto 3 | Vídeo + PDF → perfil Kolabtree / clientes industriales |
| Continuo | 3–5 propuestas/día desde el fin del P1 | — |

## Plantilla común de cada entrega

- README: problema → demo GIF → arquitectura → resultados → "cómo lo aplico a tu negocio" → contacto
- Vídeo (2 min): 0:00–0:15 problema · 0:15–1:30 demo · 1:30–1:50 números · 1:50–2:00 oferta
- PDF: una página, mismo diseño para los tres
