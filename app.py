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
from datetime import datetime

# Import core logic (Dynamic import for 1014e)
import importlib.util
import sys

# HELPER for logs
def APP_LOG(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[APP-LOG {ts}] {msg}", flush=True)

module_name = "exp1014ecaa4"
file_path = os.path.join(os.path.dirname(__file__), "1014ecaa4_scimind2_communicator.py")

spec = importlib.util.spec_from_file_location(module_name, file_path)
exp1014ecaa4 = importlib.util.module_from_spec(spec)
sys.modules[module_name] = exp1014ecaa4
spec.loader.exec_module(exp1014ecaa4)

try:
    import torch
except ImportError:
    APP_LOG("CRITICAL: torch required for SciMind 2.0 Backend")

app = FastAPI()
templates = Jinja2Templates(directory="experiments/templates")

# SymPy Symbols for Formula Generator
psi_sym, phi_sym, alpha, omega = symbols('psi phi alpha omega', complex=True)
t, x = symbols('t x', real=True)
hbar, kB, c = symbols('hbar k_B c', positive=True)

class FormulaEngine:
    def __init__(self):
        self.history = []
        
    def generate(self, coherence, metrics):
        try:
            a = sp.nsimplify(float(coherence), tolerance=0.1)
            b = sp.nsimplify(metrics.get('vorticity', 0) / 10.0, tolerance=0.1)
        except:
             a, b = 0.5, 0.5
        
        desc = "Quantum Fluctuation"
        formula = sp.Eq(psi_sym, 0)
        
        if coherence > 0.8:
            formula = sp.Eq(psi_sym, a * exp(I * omega * t))
            desc = "Coherent Wave State"
        elif coherence > 0.5:
            formula = sp.Eq(sp.Symbol('E'), a * hbar * omega)
            desc = "Quantized Energy Flow"
        elif coherence > 0.2:
            formula = sp.Eq(sp.Symbol('Phi'), a / (4 * sp.pi * x))
            desc = "Field Potential"
        else:
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

class WebSystem:
    def __init__(self):
        self.mgr = exp1014ecaa4.SessionManager()
        self.vocab = None
        self.learner = None 
        self.decoder = None
        self.noise = exp1014ecaa4.NoiseMultiplexer()
        self.holo = None
        self.text_comm = None
        self.sync_config = None
        self.ready = False
        self.formula_engine = FormulaEngine()
        
    def init_session(self, session_id=None):
        APP_LOG(f"Initializing session. Target ID: {session_id}")
        if session_id:
            sessions = self.mgr.list_sessions()
            target = next((s for s in sessions if s['id'] == session_id), None)
            if target:
                APP_LOG(f"Loading existing session: {target['id']}")
                state = self.mgr.load_session(target['path'])
                self.mgr.session_id = target['id']
                self.mgr.state_file = target['path']
                self.mgr.log_file = os.path.join(self.mgr.log_dir, f"session_{target['id']}.log")
            else:
                APP_LOG("Session not found")
                return False
        else:
            APP_LOG("Creating NEW session (with deterministic seeds)")
            state = self.mgr.start_new_session()
            
        initial_vocab = state.get('vocab', {})
        initial_sync = state.get('sync', {})
        initial_history = list(state.get('history', []))
        
        self.vocab = exp1014ecaa4.VocabularyLearner(initial_vocab)
        self.learner = exp1014ecaa4.SynchronizationLearner(initial_sync)
        self.decoder = exp1014ecaa4.SemanticAdaptiveDecoder(self.vocab) 
        self.holo = exp1014ecaa4.SciMindCommunicator(N=40) 

        if 'physics' in state and state['physics']:
             self.holo.restore_full_state(state['physics'])
        elif initial_history:
             APP_LOG("Legacy session detected. Reconstructing physics...")
             for msg in initial_history:
                 if msg['type'] == 'user':
                     dummy_noise = np.random.rand(40*40)
                     text_braid = self.holo.encode_text(msg['text'])
                     for _ in range(5):
                         self.holo.step(dummy_noise, text_braid * 1.0) 
             APP_LOG("Reconstruction complete.")
        
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
        return float(self.holo.surprisal)

system = WebSystem()

@app.get("/api/sessions")
async def list_sessions():
    mode = os.environ.get("WEB_OR_LOCAL", "local").lower()
    if mode == "web":
        return []
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
    return templates.TemplateResponse("1014ecaa4_index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    APP_LOG("WebSocket connected")
    
    while not system.ready:
        await asyncio.sleep(0.1)
    
    history_msgs = [
        {"time": m['time'], "text": m['text'], "type": m['type'], "ci": m.get('ci', 0.0)}
        for m in system.text_comm.messages
    ]
    await websocket.send_json({"type": "history", "data": history_msgs})
    
    async def _emit_state():
        # --- STEP 2: SIMULATION TIME TRACKING ---
        sim_time = 0.0 
        frame_count = 0
        # ----------------------------------------
        
        while True:
            if not system.ready:
                await asyncio.sleep(1)
                continue

            try:
                bg_noise = system.noise.get_blended_noise(size=40*40)
                stats = system.noise.get_source_stats()
                base_ntp = stats.get('ntp_offset', 0.0)
                off = system.sync_config['offset']
                total_offset = base_ntp + off
                
                current_braid = system.text_comm.last_text_unitary if system.text_comm else 0.0
                coupling = system.sync_config['coupling']

                metrics_raw = system.holo.step(bg_noise, current_braid * coupling, ntp_offset=total_offset)
                
                # Increment Simulation Time
                sim_time += 0.05 
                frame_count += 1
                
                if isinstance(system.text_comm.last_text_unitary, torch.Tensor):
                     system.text_comm.last_text_unitary *= 0.95
                
                coherence = float(system.holo.fidelity)
                vorticity = float(system.holo.vorticity) 
                entropy_val = float(system.holo.surprisal)
                ci = float(system.holo.causal_integrity)
                phases = system.holo.phases.tolist() 
                
                godel_gap = system.calculate_godel_gap()
                
                metrics = {
                    "causal_integrity": ci,
                    "vorticity": vorticity, 
                    "coherence": coherence,
                    "godel_gap": godel_gap,
                    "entropy": entropy_val
                }
                
                formula_data = system.formula_engine.generate(coherence, metrics)
                
                vocab_stats = {
                    "total": len(system.vocab.user_words) if system.vocab else 0,
                    "top": system.vocab.get_top_terms(5) if system.vocab else []
                }
                
                maps = system.holo.get_maps()
                
                # DEBUG LOGGING (Throat clearing)
                if frame_count % 100 == 0:
                    APP_LOG(f"Frame {frame_count} | SimTime {sim_time:.2f} | V: {vorticity:.2f} | CI: {ci:.2f}")

                await websocket.send_json({
                    "type": "state", 
                    "metrics": metrics, 
                    "phases": phases,
                    "maps": maps, 
                    "vocab": vocab_stats,
                    "formula": formula_data,
                    "ntp_status": f"NTP: {base_ntp:+.4f}",
                    "sim_time": sim_time # <-- SENDING SIM TIME
                })
            except Exception as e:
                APP_LOG(f"Broadcast Error: {e}")
                pass
                
            await asyncio.sleep(0.05) 
            
    async def _receive_messages():
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg['type'] == 'message':
                    text = msg['text']
                    APP_LOG(f"RX Message: {text}")
                    if system.ready and system.text_comm:
                        noise = system.noise.get_blended_noise(size=64)
                        response_text = system.text_comm.process_message(text, noise)
                        
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
                        
                        system.mgr.save_global_state(
                            system.vocab.get_state(),
                            system.learner.get_state(),
                            system.text_comm.messages,
                            physics_state=system.holo.get_full_state()
                        )
        except WebSocketDisconnect:
            APP_LOG("WS Disconnected")
        except Exception as e:
            APP_LOG(f"Receive Error: {e}")

    emit_task = asyncio.create_task(_emit_state())
    receive_task = asyncio.create_task(_receive_messages())
    
    done, pending = await asyncio.wait(
        [emit_task, receive_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    
    for task in pending:
        task.cancel()

@app.get("/api/session/export")
async def export_state():
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
    try:
        data = await request.json()
        APP_LOG("Importing session state...")
        
        if not system.mgr.session_id:
             system.mgr.start_new_session()

        system.vocab = exp1014ecaa4.VocabularyLearner(data.get('vocab', {}))
        system.learner = exp1014ecaa4.SynchronizationLearner(data.get('sync', {}))
        system.decoder = exp1014ecaa4.SemanticAdaptiveDecoder(system.vocab)
        system.holo = exp1014ecaa4.SciMindCommunicator(N=40)
        
        if 'physics' in data and data['physics']:
            system.holo.restore_full_state(data['physics'])
            
        system.text_comm = exp1014ecaa4.AdaptiveLoggingCommunicator(
            system.decoder, system.holo, system.vocab, system.mgr
        )
        
        if 'history' in data:
            system.text_comm.messages = exp1014ecaa4.deque(data['history'], maxlen=50)
            
        system.sync_config = system.learner.best_config
        system.ready = True
        
        system.mgr.save_global_state(
            system.vocab.get_state(),
            system.learner.get_state(),
            system.text_comm.messages,
            physics_state=system.holo.get_full_state()
        )
        
        return {"status": "ok", "message": "Session imported successfully"}
    except Exception as e:
        APP_LOG(f"Import Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Standard Uvicorn startup
    uvicorn.run(app, host="0.0.0.0", port=8000)
