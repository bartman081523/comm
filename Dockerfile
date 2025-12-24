FROM python:3.10
ENV WEB_OR_LOCAL=web

# User setup für Hugging Face Security
user root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Dependencies installieren
COPY --chown=user:user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Ordnerstruktur für Templates erstellen (wie im Code erwartet)
RUN mkdir -p experiments/templates

# Dateien kopieren
# 1. Die Core-Logik (muss exakt so heißen wegen importlib im Code)
COPY --chown=user:user 1014ecaa4_scimind2_communicator.py 1014ecaa4_scimind2_communicator.py

# 2. Die App (umbenannt zu app.py für Konvention, Inhalt ist input_file_1.py)
COPY --chown=user:user app.py app.py

# 3. Das HTML Template (muss in den Unterordner und exakt so heißen)
COPY --chown=user:user experiments/templates/1014ecaa4_index.html experiments/templates/1014ecaa4_index.html

# Startbefehl: Überschreibt den __main__ Block im Code, um Port 7860 zu nutzen
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--ws", "websockets", "--timeout-keep-alive", "120"]