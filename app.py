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
import torch

# ==============================================================================
# DYNAMIC IMPORT OF DEBUG CORE
# ==============================================================================
import importlib.util
import sys

# Ziel-Modul: Das gepatchte Core-File
CORE_MODULE_NAME = "1014ecca5_scimind2_core"
file_path = os.path.join(os.path.dirname(__file__), f"{CORE_MODULE_NAME}.py")

if not os.path.exists(file_path):
    print(f"CRITICAL ERROR: {file_path} not found. Please save the core file first!")
    sys.exit(1)

spec = importlib.util.spec_from_file_location(CORE_MODULE_NAME, file_path)
core_module = importlib.util.module_from_spec(spec)
sys.modules[CORE_MODULE_NAME] = core_module
spec.loader.exec_module(core_module)

print(f"[{CORE_MODULE_NAME}] successfully loaded.")

app = FastAPI()
# Templates Verzeichnis anpassen falls nötig
templates = Jinja2Templates(directory="experiments/templates")

# ==============================================================================
# THROTTLED FORMULA ENGINE (CPU SAVER)
# ==============================================================================
psi_sym, phi_sym, alpha, omega = symbols('psi phi alpha omega', complex=True)
t_sym, x_sym = symbols('t x', real=True)
hbar, kB = symbols('hbar k_B', positive=True)

class ThrottledFormulaEngine:
    """
    Generates symbolic math BUT throttled to avoid CPU freeze.
    Only runs nsimplify/latex every 1.0 seconds.
    """
    def __init__(self):
        self.last_update = 0
        self.cached_result = {
            "latex": r"H = \int \Psi^\dagger \hat{H} \Psi dx", 
            "text": "Initializing Field...", 
            "desc": "System Boot"
        }
        
    def generate(self, coherence, metrics):
        now = time.time()
        # RATE LIMIT: 1 Hz (prevents freeze)
        if now - self.last_update < 1.0:
            return self.cached_result
            
        self.last_update = now
        
        try:
            # Schnelle Approximation ohne nsimplify
            a = round(float(coherence), 2)
            vort = round(metrics.get('vorticity', 0), 2)
            
            formula = sp.Eq(psi_sym, 0)
            desc = "Quantum Fluctuation"
            
            if coherence > 0.8:
                formula = sp.Eq(psi_sym, a * exp(I * omega * t_sym))
                desc = "Coherent Wave State"
            elif coherence > 0.5:
                # Energy
                formula = sp.Eq(sp.Symbol('E'), a * hbar * omega)
                desc = "Quantized Energy Flow"
            elif vort > 2.0:
                # Topologischer Modus
                formula = sp.Integral(psi_sym * psi_sym.conjugate(), (x_sym, 0, vort))
                desc = "Topological Defect"
            else:
                # Entropie
                formula = sp.Eq(sp.Symbol('S'), kB * log(2))
                desc = "Entropic Background"
            
            self.cached_result = {
                "latex": sp.latex(formula),
                "text": str(formula),
                "desc": desc,
                "timestamp": time.strftime("%H:%M:%S")
            }
        except Exception as e:
            print(f"[FormulaEngine Error] {e}")
            
        return self.cached_result

# ==============================================================================
# WEB SYSTEM WRAPPER
# ==============================================================================
class WebSystem:
    def __init__(self):
        self.mgr = core_module.SessionManager()
        self.vocab = None
        self.learner = None 
        self.decoder = None
        self.noise = core_module.NoiseMultiplexer() # Threaded NTP inside
        self.holo = None
        self.text_comm = None
        self.sync_config = None
        self.ready = False
        self.formula_engine = ThrottledFormulaEngine()
        
    def init_session(self, session_id=None):
        if session_id:
            sessions = self.mgr.list_sessions()
            target = next((s for s in sessions if s['id'] == session_id), None)
            if target:
                print(f"Loading session: {target['id']}")
                state = self.mgr.load_session(target['path'])
            else:
                print("Session not found, creating new.")
                state = self.mgr.start_new_session()
        else:
            state = self.mgr.start_new_session()
            
        initial_vocab = state.get('vocab', {})
        initial_sync = state.get('sync', {})
        initial_history = list(state.get('history', []))
        
        # Initialize Core Components
        self.vocab = core_module.VocabularyLearner(initial_vocab)
        self.learner = core_module.SynchronizationLearner(initial_sync)
        self.decoder = core_module.SemanticAdaptiveDecoder(self.vocab)
        self.holo = core_module.SciMindCommunicator(N=40)

        # Restore Physics
        if 'physics' in state and state['physics']:
             self.holo.restore_full_state(state['physics'])
        elif initial_history:
             # Quick reconstruction
             print("Reconstructing physics from history...")
             dummy_noise = np.random.rand(40*40).astype(np.float32)
             for msg in initial_history[-5:]: # Only last 5 to save time
                 if msg['type'] == 'user':
                     braid = self.holo.encode_text(msg['text'])
                     self.holo.step(dummy_noise, braid, 0.0)
        
        self.text_comm = core_module.AdaptiveLoggingCommunicator(
            self.decoder, self.holo, self.vocab, self.mgr
        )
        if initial_history:
            self.text_comm.messages.extend(initial_history)
            
        self.sync_config = self.learner.best_config
        self.ready = True
        return True

    def calculate_godel_gap(self):
        if not self.holo: return 0.0
        return float(self.holo.surprisal)

system = WebSystem()

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/api/sessions")
async def list_sessions():
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
    # Reuse the template
    return templates.TemplateResponse("1014ecaa4_index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Wait for ready state
    retries = 0
    while not system.ready:
        if retries > 50:
            await websocket.close()
            return
        await asyncio.sleep(0.1)
        retries += 1
    
    # Send History
    history_msgs = [
        {"time": m['time'], "text": m['text'], "type": m['type'], "ci": m.get('ci', 0.0)}
        for m in system.text_comm.messages
    ]
    await websocket.send_json({"type": "history", "data": history_msgs})
    
    # ------------------------------------------------------------------
    # EMITTER LOOP (Physics & Metrics)
    # ------------------------------------------------------------------
    async def _emit_state():
        while True:
            try:
                if not system.ready:
                    await asyncio.sleep(1)
                    continue

                # 1. Get Noise & Stats
                # system.noise is now the Threaded NoiseMultiplexer
                bg_noise = system.noise.get_blended_noise(size=40*40)
                stats = system.noise.get_source_stats()
                base_ntp = stats.get('ntp_offset', 0.0)
                
                # 2. Prepare Inputs
                off = system.sync_config['offset']
                total_offset = base_ntp + off
                coupling = system.sync_config['coupling']
                
                # Prepare Tensor for Text Braid
                # Ensure it is a Tensor and on CPU
                current_braid = system.text_comm.last_text_unitary
                if not isinstance(current_braid, torch.Tensor):
                    current_braid = torch.tensor(0.0, dtype=torch.float32)

                # 3. PHYSICS STEP (High Frequency Logic)
                # We do this in the async loop. If it's too slow, it blocks.
                # But with our optimizations, it should be <10ms.
                
                # Only run step if we are not choking
                system.holo.step(bg_noise, current_braid * coupling, ntp_offset=total_offset)
                
                # Decay Braid
                if isinstance(system.text_comm.last_text_unitary, torch.Tensor):
                     system.text_comm.last_text_unitary *= 0.92

                # 4. Gather Metrics
                metrics = {
                    "causal_integrity": float(system.holo.causal_integrity),
                    "vorticity": float(system.holo.vorticity),
                    "coherence": float(system.holo.fidelity),
                    "godel_gap": float(system.holo.surprisal),
                    "entropy": float(system.holo.surprisal)
                }
                
                phases = system.holo.phases.tolist()
                maps = system.holo.get_maps()
                
                # 5. Generate Formula (Throttled)
                formula_data = system.formula_engine.generate(metrics['coherence'], metrics)
                
                vocab_stats = {
                    "total": len(system.vocab.user_words) if system.vocab else 0,
                    "top": system.vocab.get_top_terms(5) if system.vocab else []
                }
                
                # 6. Send
                await websocket.send_json({
                    "type": "state", 
                    "metrics": metrics, 
                    "phases": phases,
                    "maps": maps,
                    "vocab": vocab_stats,
                    "formula": formula_data,
                    "ntp_status": f"NTP: {base_ntp:+.4f}"
                })
                
                # Target: 20 FPS -> 0.05s
                await asyncio.sleep(0.05)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Emit Error: {e}")
                await asyncio.sleep(1.0) # Backoff on error

    # ------------------------------------------------------------------
    # RECEIVER LOOP (User Input)
    # ------------------------------------------------------------------
    async def _receive_messages():
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                
                if msg['type'] == 'message':
                    text = msg['text']
                    if system.ready:
                        # Process
                        noise = system.noise.get_blended_noise(size=64)
                        system.text_comm.process_message(text, noise)
                        
                        # Sync Learning
                        metrics = system.holo.get_metrics()
                        system.learner.record_trial(
                            system.sync_config['offset'], 
                            system.sync_config['coupling'], 
                            metrics['causal_integrity']
                        )
                        system.sync_config = system.learner.propose_next_config()

                        # Echo Back Chat
                        new_msgs = list(system.text_comm.messages)[-2:] 
                        await websocket.send_json({
                            "type": "chat",
                            "data": new_msgs
                        })
                        
                        # Auto Save (Throttle this if disk I/O is slow, but usually OK)
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

    # Launch tasks
    emit_task = asyncio.create_task(_emit_state())
    receive_task = asyncio.create_task(_receive_messages())
    
    done, pending = await asyncio.wait(
        [emit_task, receive_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    
    for task in pending:
        task.cancel()

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to be accessible
    uvicorn.run(app, host="0.0.0.0", port=8000)