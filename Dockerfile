FROM python:3.10

# WICHTIG: Verhindert, dass Python Logs puffert. 
# Sorgt dafür, dass print() sofort in den Docker-Logs erscheint.
ENV PYTHONUNBUFFERED=1

ENV WEB_OR_LOCAL=web

# User setup für Hugging Face Security
USER root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Dependencies installieren
COPY --chown=user:user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Ordnerstruktur für Templates erstellen
RUN mkdir -p experiments/templates

# Dateien kopieren
# 1. Die Core-Logik
COPY --chown=user:user 1014ecaa4_scimind2_communicator.py 1014ecaa4_scimind2_communicator.py

# 2. Die App (HINWEIS: Stelle sicher, dass deine lokale Datei 'comm-app.py' heißt 
# oder passe den Namen hier an. Im Dockerfile unten kopierst du sie als 'app.py')
COPY --chown=user:user comm-app.py app.py

# 3. Das HTML Template
COPY --chown=user:user experiments/templates/1014ecaa4_index.html experiments/templates/1014ecaa4_index.html

# Startbefehl: 
# 1. --log-level debug hinzugefügt für mehr Server-Logs
# 2. --access-log hinzugefügt, damit HTTP Requests sichtbar sind
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--ws", "websockets", "--timeout-keep-alive", "120", "--log-level", "debug", "--access-log"]
