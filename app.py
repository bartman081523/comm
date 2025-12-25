import numpy as np
import time
import json
import asyncio
import os
import sympy as sp
from sympy import symbols, exp, I, log, sqrt
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import core logic (Dynamic import for 1014e)
import importlib.util
import sys

# Dynamic import because "1014" starts with a digit
# Dynamic import because "1014" starts with a digit
module_name = "exp1014ecaa4"
file_path = os.path.join(os.path.dirname(__file__), "1014ecaa4_scimind2_communicator.py")

spec = importlib.util.spec_from_file_location(module_name, file_path)
exp1014ecaa4 = importlib.util.module_from_spec(spec)
sys.modules[module_name] = exp1014ecaa4
spec.loader.exec_module(exp1014ecaa4)

# Also need torch for tensor handling
try:
    import torch
except ImportError:
    print("CRITICAL: torch required for SciMind 2.0 Backend")

app = FastAPI()
# We reuse templates from 1014
templates = Jinja2Templates(directory="experiments/templates")

# SymPy Symbols for Formula Generator
psi_sym, phi_sym, alpha, omega = symbols('psi phi alpha omega', complex=True)
t, x = symbols('t x', real=True)
hbar, kB, c = symbols('hbar k_B c', positive=True)

class FormulaEngine:
    """Generates symbolic math from quantum state patterns (Ported from Exp 1004)."""
    def __init__(self):
        self.history = []
        
    def generate(self, coherence, metrics):
        # Generate formula based on coherence regime
        try:
            a = sp.nsimplify(float(coherence), tolerance=0.1)
            b = sp.nsimplify(metrics.get('vorticity', 0) / 10.0, tolerance=0.1)
        except:
             a, b = 0.5, 0.5
        
        desc = "Quantum Fluctuation"
        formula = sp.Eq(psi_sym, 0)
        
        if coherence > 0.8:
            # Wave Function
            formula = sp.Eq(psi_sym, a * exp(I * omega * t))
            desc = "Coherent Wave State"
        elif coherence > 0.5:
            # Energy
            formula = sp.Eq(sp.Symbol('E'), a * hbar * omega)
            desc = "Quantized Energy Flow"
        elif coherence > 0.2:
            # Field Potentials
            formula = sp.Eq(sp.Symbol('Phi'), a / (4 * sp.pi * x))
            desc = "Field Potential"
        else:
            # Entropy
            try:
                p = max(0.001, float(a))
                formula = sp.Eq(sp.Symbol('S'), -kB * p * log(p))
                desc = "Entropic Fluctuations"
            except:
                formula = sp.Eq(sp.Symbol('S'), kB * log(2))
            
        return {
            "latex": sp.latex(formula),
            "text": str(formula),
            "desc": desc,
            "timestamp": time.strftime("%H:%M:%S")
        }

# State
class WebSystem:
    def __init__(self):
        self.mgr = exp1014ecaa4.SessionManager()
        self.vocab = None
        self.learner = None # Sync Learner
        self.decoder = None
        self.noise = exp1014ecaa4.NoiseMultiplexer()
        self.holo = None
        self.text_comm = None
        self.sync_config = None
        self.ready = False
        self.formula_engine = FormulaEngine()
        
    def init_session(self, session_id=None):
        if session_id:
            # Find path
            sessions = self.mgr.list_sessions()
            target = next((s for s in sessions if s['id'] == session_id), None)
            if target:
                print(f"Loading session: {target['id']}")
                state = self.mgr.load_session(target['path'])
                self.mgr.session_id = target['id']
                self.mgr.state_file = target['path']
                self.mgr.log_file = os.path.join(self.mgr.log_dir, f"session_{target['id']}.log")
            else:
                return False
        else:
            state = self.mgr.start_new_session()
            
        initial_vocab = state.get('vocab', {})
        initial_sync = state.get('sync', {})
        initial_history = list(state.get('history', []))
        
        self.vocab = exp1014ecaa4.VocabularyLearner(initial_vocab)
        self.learner = exp1014ecaa4.SynchronizationLearner(initial_sync)
        
        self.decoder = exp1014ecaa4.SemanticAdaptiveDecoder(self.vocab) # Use 1014ec Decoder
        self.holo = exp1014ecaa4.SciMindCommunicator(N=40) # Use SciMind 2.0 Core

        # RESTORE PHYSICS STATE (or Reconstruct)
        if 'physics' in state and state['physics']:
             self.holo.restore_full_state(state['physics'])
        elif initial_history:
             # RECONSTRUCTION MODE for Legacy Sessions
             print("Legacy session detected. Reconstructing physics state from history...")
             for msg in initial_history:
                 if msg['type'] == 'user':
                     # Re-imprint the text to the braid field
                     # We assume a standard noise level for reconstruction to avoid divergence
                     dummy_noise = np.random.rand(40*40)
                     text_braid = self.holo.encode_text(msg['text'])
                     # Run a few steps to let it settle
                     for _ in range(5):
                         self.holo.step(dummy_noise, text_braid * 1.0) # Nominal coupling
             print("Reconstruction complete.")
        
        self.text_comm = exp1014ecaa4.AdaptiveLoggingCommunicator(
            self.decoder, self.holo, self.vocab, self.mgr
        )
        if initial_history:
            self.text_comm.messages.extend(initial_history)
            
        self.sync_config = self.learner.best_config
        self.ready = True
        return True

    def calculate_godel_gap(self):
        if not self.holo: return 0.0
        # Simple approximation for now
        return float(self.holo.surprisal)

system = WebSystem()

@app.get("/api/sessions")
async def list_sessions():
    # Prüfen, ob wir im Web-Modus (Hugging Face) oder Lokal sind
    # Standard ist 'local', wenn die Variable fehlt (beim lokalen Klonen)
    mode = os.environ.get("WEB_OR_LOCAL", "local").lower()
    
    if mode == "web":
        # Im Web-Modus geben wir eine leere Liste zurück.
        # Das Frontend zeigt dann keine Sessions an, nur die Buttons.
        return []
    
    # Im lokalen Modus geben wir die echte Liste zurück
    return system.mgr.list_sessions()
    
@app.post("/api/session/new")
async def new_session():
    system.init_session(None)
    return {"status": "ok", "id": system.mgr.session_id}

@app.post("/api/session/load/{session_id}")
async def load_session(session_id: str):
    if system.init_session(session_id):
        return {"status": "ok", "id": session_id}
    return {"status": "error", "message": "Session not found"}

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    # Reuse existing template, it's compatible enough
    return templates.TemplateResponse("1014ecaa4_index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Wait for session init
    while not system.ready:
        await asyncio.sleep(0.1)
    
    # Send initial history
    history_msgs = [
        {"time": m['time'], "text": m['text'], "type": m['type'], "ci": m.get('ci', 0.0)}
        for m in system.text_comm.messages
    ]
    await websocket.send_json({"type": "history", "data": history_msgs})
    
    async def _emit_state():
        while True:
            if not system.ready:
                await asyncio.sleep(1)
                continue

            try:
                # 1. Physics Evolution (Continuous)
                # Noise source
                bg_noise = system.noise.get_blended_noise(size=40*40)
                                
                # Stats for Wick Rotation
                stats = system.noise.get_source_stats()
                base_ntp = stats.get('ntp_offset', 0.0)
                off = system.sync_config['offset']
                total_offset = base_ntp + off
                
                # Input Unitary (Braid Field)
                # FIX: text_comm.last_text_unitary in 1014e is a Tensor (Braid Field)
                current_braid = system.text_comm.last_text_unitary if system.text_comm else 0.0
                coupling = system.sync_config['coupling']

                # STEP using SciMind 2.0 Logic (including ntp_offset for Wick Rotation)
                # Note: SciMindCommunicator.step accepts braid field + ntp_offset
                metrics_raw = system.holo.step(bg_noise, current_braid * coupling, ntp_offset=total_offset)
                
                # Decay the tensor signal (Memory Fade)
                if isinstance(system.text_comm.last_text_unitary, torch.Tensor):
                     system.text_comm.last_text_unitary *= 0.95
                
                # Attributes
                coherence = float(system.holo.fidelity)
                vorticity = float(system.holo.vorticity) # Chern Number
                entropy_val = float(system.holo.surprisal)
                ci = float(system.holo.causal_integrity)
                phases = system.holo.phases.tolist() 
                
                # Advanced Metrics
                godel_gap = system.calculate_godel_gap()
                
                metrics = {
                    "causal_integrity": ci,
                    "vorticity": vorticity, # Chern
                    "coherence": coherence,
                    "godel_gap": godel_gap,
                    "entropy": entropy_val
                }
                
                # Generate Formula
                formula_data = system.formula_engine.generate(coherence, metrics)
                
                vocab_stats = {
                    "total": len(system.vocab.user_words) if system.vocab else 0,
                    "top": system.vocab.get_top_terms(5) if system.vocab else []
                }
                
                # GET MAPS (Gating, Vorticity)
                maps = system.holo.get_maps()
                
                await websocket.send_json({
                    "type": "state", 
                    "metrics": metrics, 
                    "phases": phases,
                    "maps": maps, # NEW: Send Maps
                    "vocab": vocab_stats,
                    "formula": formula_data,
                    "ntp_status": f"NTP: {base_ntp:+.4f}"
                })
            except Exception as e:
                # print(f"Broadcast Error: {e}")
                pass
                
            await asyncio.sleep(0.05) # 20Hz update
            
    async def _receive_messages():
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg['type'] == 'message':
                    text = msg['text']
                    if system.ready and system.text_comm:
                        noise = system.noise.get_blended_noise(size=64)
                        
                        # Process message (Imprints Braid, Decodes Response)
                        response_text = system.text_comm.process_message(text, noise)
                        
                        # Sync Learning Update
                        metrics = system.holo.get_metrics()
                        
                        system.learner.record_trial(
                            system.sync_config['offset'], 
                            system.sync_config['coupling'], 
                            metrics['causal_integrity']
                        )
                        system.sync_config = system.learner.propose_next_config()

                        new_msgs = list(system.text_comm.messages)[-2:] 
                        await websocket.send_json({
                            "type": "chat",
                            "data": new_msgs
                        })
                        
                        # Auto Save
                        system.mgr.save_global_state(
                            system.vocab.get_state(),
                            system.learner.get_state(),
                            system.text_comm.messages,
                            physics_state=system.holo.get_full_state()
                        )
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Receive Error: {e}")

    # Run both loops
    emit_task = asyncio.create_task(_emit_state())
    receive_task = asyncio.create_task(_receive_messages())
    
    done, pending = await asyncio.wait(
        [emit_task, receive_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    
    for task in pending:
        task.cancel()

# --- NEU: EXPORT / IMPORT ENDPOINTS ---

@app.get("/api/session/export")
async def export_state():
    """Gibt den kompletten aktuellen Zustand als JSON zurück."""
    if not system.ready:
        return {"error": "System not ready"}
    
    return {
        'vocab': system.vocab.get_state(),
        'sync': system.learner.get_state(),
        'history': list(system.text_comm.messages),
        'physics': system.holo.get_full_state(),
        'timestamp': time.time()
    }

@app.post("/api/session/import")
async def import_state(request: Request):
    """Empfängt ein JSON, initialisiert das System und überschreibt den Zustand."""
    try:
        data = await request.json()
        
        # 0. Session Manager vorbereiten (falls noch keine ID existiert)
        if not system.mgr.session_id:
             system.mgr.start_new_session()

        # 1. Komponenten INITIALISIEREN (Wichtig: Auch wenn sie noch nicht existieren)
        # Wir erstellen alles neu, um sicherzugehen, dass keine alten Datenreste stören.
        system.vocab = exp1014ecaa4.VocabularyLearner(data.get('vocab', {}))
        system.learner = exp1014ecaa4.SynchronizationLearner(data.get('sync', {}))
        
        system.decoder = exp1014ecaa4.SemanticAdaptiveDecoder(system.vocab)
        
        # Physics Engine neu erstellen
        system.holo = exp1014ecaa4.SciMindCommunicator(N=40)
        
        # 2. Physics State wiederherstellen
        if 'physics' in data and data['physics']:
            system.holo.restore_full_state(data['physics'])
            
        # 3. Text Communicator verbinden
        system.text_comm = exp1014ecaa4.AdaptiveLoggingCommunicator(
            system.decoder, system.holo, system.vocab, system.mgr
        )
        
        # 4. Chat History wiederherstellen
        if 'history' in data:
            # Konvertieren der Liste zurück in eine Deque
            system.text_comm.messages = exp1014ecaa4.deque(data['history'], maxlen=50)
            
        # 5. System "Scharfschalten"
        system.sync_config = system.learner.best_config
        system.ready = True
        
        # 6. Sofort speichern (Auto-Save)
        system.mgr.save_global_state(
            system.vocab.get_state(),
            system.learner.get_state(),
            system.text_comm.messages,
            physics_state=system.holo.get_full_state()
        )
        
        return {"status": "ok", "message": "Session imported successfully"}
    except Exception as e:
        print(f"Import Error: {e}")
        # traceback für besseres Debugging in den Logs ausgeben
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)