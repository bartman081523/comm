#!/usr/bin/env python3
"""
Experiment 1014ecaa4: SciMind 2.0 Complexity Reduction & Legacy Recovery
Axiom of Causal Integrity (TCI) - Holographic Communication Interface

Upgrades from 1014b/c:
1.  Topological Imprinting
2.  CNOT Hamiltonian Protection
3.  Wick-Rotation PLL
4.  Vorticity/Chern Audit

PATCHED: Determinism & Debugging enabled.
"""

import sys
import os
import time
import curses
import threading
import numpy as np
import queue
import json
import glob
import random 
from collections import deque, Counter
from datetime import datetime
import sympy as sp

# DEBUGGING HELPER FOR DOCKER
def DEBUG_LOG(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[CORE-DEBUG {ts}] {msg}", flush=True)

# Optional deps
try:
    import torch
except ImportError:
    DEBUG_LOG("CRITICAL: torch not found.")
    sys.exit(1)

try:
    import ntplib
    NTP_AVAILABLE = True
except ImportError:
    NTP_AVAILABLE = False

# ==============================================================================
# SESSION MANAGEMENT (FIXED & SEEDED)
# ==============================================================================
class SessionManager:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = None
        self.state_file = None
        self.log_file = None

    def list_sessions(self):
        files = glob.glob(os.path.join(self.log_dir, "session_*.json"))
        sessions = []
        for f in sorted(files, reverse=True):
            basename = os.path.basename(f)
            ts = basename.replace("session_", "").replace(".json", "")
            readable = ts 
            try:
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                readable = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass 
            sessions.append({'id': ts, 'path': f, 'label': readable})
        return sessions

    def start_new_session(self):
        # --- STEP 1: DETERMINISM FIX ---
        # Set fixed seeds so Cloud and Local start identical
        DEBUG_LOG("Applying deterministic seeds (42) for new session.")
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        # -------------------------------

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_id = ts
        self.state_file = os.path.join(self.log_dir, f"session_{ts}.json")
        self.log_file = os.path.join(self.log_dir, f"session_{ts}.log")
        DEBUG_LOG(f"Starting NEW session: {self.session_id}")
        return {} 

    def load_session(self, session_path):
        ts = os.path.basename(session_path).replace("session_", "").replace(".json", "")
        self.session_id = ts
        self.state_file = session_path
        self.log_file = os.path.join(self.log_dir, f"session_{ts}.log")
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                DEBUG_LOG(f"Loaded session state from {session_path}")
                return json.load(f)
        except Exception as e:
            DEBUG_LOG(f"Error loading state: {e}")
            return {}

    def save_global_state(self, vocab_data, sync_data, chat_history, physics_state=None):
        if not self.state_file: return
        data = {
            'vocab': vocab_data,
            'sync': sync_data,
            'history': list(chat_history),
            'physics': physics_state,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# ==============================================================================
# VOCABULARY LEARNER (Universal)
# ==============================================================================
STOP_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", 
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", 
    "this", "but", "his", "by", "from", "they", "we", "say", "her", 
    "she", "or", "an", "will", "my", "one", "all", "would", "there", 
    "their", "what", "so", "up", "out", "if", "about", "who", "get", 
    "which", "go", "me", "is", "are", "can", "has", "was", "were"
}

class VocabularyLearner:
    def __init__(self, initial_state=None):
        self.user_words = Counter()
        self.total_words = 0
        if initial_state:
            self.user_words = Counter(initial_state.get('user_words', {}))
            self.total_words = initial_state.get('total_words', 0)

    def learn_from_input(self, text):
        tokens = text.split()
        new_words = 0
        for token in tokens:
            cleaned = token.lower().strip(".,!?")
            if len(cleaned) > 2 and cleaned not in STOP_WORDS:
                self.user_words[token] += 1 
                new_words += 1
        self.total_words += new_words
            
    def get_top_terms(self, n=50):
        return [w for w, c in self.user_words.most_common(n)]

    def get_state(self):
        return {
            'user_words': dict(self.user_words),
            'total_words': self.total_words
        }

# ==============================================================================
# SEMANTIC ADAPTIVE DECODER
# ==============================================================================
BASE_WORD_POOL = [
    "Existence", "Being", "Becoming", "Time", "Space", "Light", "Energy",
    "Information", "Consciousness", "Order", "Chaos", "Symmetry",
    "Emergence", "Coherence", "Resonance", "Harmony", "Frequency",
    "Quantity", "Quality", "Truth", "Beauty", "Unity",
    "Plurality", "Infinity", "Eternity", "Moment", "Process"
]

INTERPRETATIONS = {
    'psi': ["A wave function manifests...", "Information crystallizes as:"],
    'phi': ["A field permeates space:", "Force manifests as:"],
    's':   ["Entropy structures itself as:", "Chaos contains:"],
}

class SemanticAdaptiveDecoder:
    def __init__(self, vocab_learner):
        self.vocab_learner = vocab_learner
        self.coherence_history = deque(maxlen=100)
        self.last_coherence = 0.0
        self.last_result = None
    
    def decode_cycle(self, noise, verbose=False):
        if len(noise) < 10: noise = np.random.rand(64)
        phases = noise * 2 * np.pi
        order_param = np.abs(np.mean(np.exp(1j * phases)))
        coherence = float(order_param)
        self.last_coherence = coherence
        
        variance = np.var(noise)
        godel_gap = float(variance * 10.0) 
        
        info_slice = noise[:5]
        
        result = {
            'coherence': coherence,
            'godel_gap': godel_gap,
            'new_info': info_slice,
            'interpretation': self._get_interpretation(coherence),
            'patterns': {'ratios': noise[:3]}
        }
        self.coherence_history.append(coherence)
        self.last_result = result
        return result

    def _get_interpretation(self, coherence):
        key = random.choice(list(INTERPRETATIONS.keys()))
        base = random.choice(INTERPRETATIONS[key])
        
        if coherence > 0.8: tone = "[CRYSTAL CLEAR]"
        elif coherence > 0.5: tone = "[CLEAR]"
        else: tone = "[FAINT]"
        return f"{tone} {base}"

    def info_to_message(self, info):
        user_pool = self.vocab_learner.get_top_terms()
        if not user_pool:
            hybrid_pool = BASE_WORD_POOL
        else:
            hybrid_pool = BASE_WORD_POOL + user_pool
            
        indices = (np.array(info) * len(hybrid_pool)).astype(int)
        indices = np.clip(indices, 0, len(hybrid_pool) - 1)
        
        selected = [hybrid_pool[i] for i in indices[:3]]
        return " ↔ ".join(selected) 

# ==============================================================================
# SCIMIND 2.0 PHYSICS CORE
# ==============================================================================
class SciMindCommunicator:
    def __init__(self, N=40):
        self.size = N
        self.phases = self._init_vortex(N)
        self.focus = 0.0
        self.surprisal = 0.0
        self.vorticity = 0.0 
        self.causal_integrity = 0.0
        self.fidelity = 0.0
        
        self.gating_field = np.zeros((N, N))
        self.vorticity_field = np.zeros((N, N))
        
        self.t_res, self.Omega = sp.symbols('t_res Omega', real=True)
        self.H_comm = self._derive_hamiltonian()
        self.phase_history = deque(maxlen=200)

    def _init_vortex(self, size):
        x = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, x)
        phases = np.mod(np.arctan2(yy, xx) + np.pi, 2 * np.pi)
        return torch.tensor(phases, dtype=torch.float32)

    def _derive_hamiltonian(self):
        return (self.Omega/2) * sp.Matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])

    def get_chern_number(self):
        p = self.phases.numpy()
        dx = np.angle(np.exp(1j * (np.roll(p, -1, axis=1) - p)))
        dy = np.angle(np.exp(1j * (np.roll(p, -1, axis=0) - p)))
        circ = dx + np.roll(dy, -1, axis=1) - np.roll(dx, -1, axis=0) - dy
        self.vorticity_field = circ / (2 * np.pi) 
        return float(np.sum(np.abs(circ)) / (2 * np.pi))

    def encode_text(self, text):
        if not text: return torch.zeros((self.size, self.size), dtype=torch.float32)
        
        x = np.linspace(-1, 1, self.size)
        xx, yy = np.meshgrid(x, x)
        xx = torch.tensor(xx, dtype=torch.float32)
        yy = torch.tensor(yy, dtype=torch.float32)
        
        braid_field = torch.zeros((self.size, self.size), dtype=torch.float32)
        n_chars = len(text)
        phi = (np.sqrt(5) - 1) / 2 
        
        for i, char in enumerate(text):
            idx = i + 1
            r = np.sqrt(idx / max(n_chars, 1)) * 0.8
            theta = 2 * np.pi * idx * phi
            cx = r * np.cos(theta)
            cy = r * np.sin(theta)
            charge = 1.0 if ord(char) % 2 == 0 else -1.0
            braid_field += charge * torch.atan2(yy - cy, xx - cx)
            
        return braid_field

    def step(self, external_noise, text_braid_field, ntp_offset=0.0):
        p_np = self.phases.numpy()
        N = self.size
        
        grad_y, grad_x = np.gradient(p_np)
        unrest_map = np.sqrt(grad_x**2 + grad_y**2)
        unrest_norm = (unrest_map - np.min(unrest_map)) / (np.max(unrest_map) - np.min(unrest_map) + 1e-6)
        
        decay_factor = 0.95
        activation_threshold = 0.6 
        
        new_gating = self.gating_field * decay_factor
        new_gating += 0.5 * (unrest_norm > activation_threshold).astype(float)
        self.gating_field = np.clip(new_gating, 0.01, 1.0) 
        gating_tensor = torch.tensor(self.gating_field, dtype=torch.float32)

        res_freq = 4 * np.pi / N
        
        if isinstance(external_noise, (int, float)):
             noise_tensor = torch.tensor(external_noise).expand(N, N)
        else:
             if isinstance(external_noise, (list, np.ndarray)):
                 noise_tensor = torch.tensor(external_noise, dtype=torch.float32).view(N, N) if len(external_noise) == N*N else torch.tensor(external_noise[0]).expand(N, N)
             else:
                 noise_tensor = external_noise

        phase_res_mask = torch.cos(self.phases * 4) 
        phase_res_mask = (phase_res_mask + 1) / 2 
        phase_res_mask = phase_res_mask * 0.9 + 0.1 
        resonant_noise = noise_tensor * phase_res_mask

        current_vorticity = self.get_chern_number()
        base_alpha = 0.1
        impedance_factor = 1.0 / (1.0 + 0.5 * abs(current_vorticity))
        effective_alpha = base_alpha * impedance_factor
        
        interaction = gating_tensor * text_braid_field
        wick_rotation = ntp_offset * 5.0 
        
        laplacian = (np.roll(p_np, 1, axis=0) + np.roll(p_np, -1, axis=0) + 
                     np.roll(p_np, 1, axis=1) + np.roll(p_np, -1, axis=1) - 4*p_np)
        diffusion = torch.tensor(laplacian, dtype=torch.float32)
        diffusion_rate = 0.05 + 0.1 * (1.0 - gating_tensor) 
        
        force = (interaction * 1.5) + (resonant_noise * 0.5) + wick_rotation + (diffusion * diffusion_rate)
        
        self.phases = (self.phases + effective_alpha * force) % (2 * np.pi)
        
        self.focus = float(np.mean(self.gating_field))
        self.vorticity = self.get_chern_number() 
        
        variance = float(torch.var(self.phases))
        self.surprisal = -np.log(max(variance, 1e-9) / (np.pi**2 + 1e-9))
        
        v_residue = abs(self.vorticity - round(self.vorticity))
        topo_stability = np.exp(-v_residue * 5.0) 
        self.causal_integrity = self.focus * (abs(self.vorticity) + 1.0) * topo_stability * 10
        self.fidelity = self.focus
        
        return self.get_metrics()
        
    def get_metrics(self):
        return {
            'fidelity': self.fidelity,
            'vorticity': self.vorticity,
            'surprisal': self.surprisal,
            'causal_integrity': self.causal_integrity
        }
    
    def get_maps(self):
        return {
            'gating': self.gating_field.tolist() if isinstance(self.gating_field, np.ndarray) else self.gating_field,
            'vorticity': self.vorticity_field.tolist() if isinstance(self.vorticity_field, np.ndarray) else self.vorticity_field
        }

    def get_full_state(self):
        return {
            'phases': self.phases.numpy().tolist(),
            'gating': self.gating_field.tolist() if isinstance(self.gating_field, np.ndarray) else self.gating_field,
            'vorticity': self.vorticity_field.tolist() if isinstance(self.vorticity_field, np.ndarray) else self.vorticity_field,
        }

    def restore_full_state(self, state_dict):
        try:
            if 'phases' in state_dict:
                self.phases = torch.tensor(state_dict['phases'], dtype=torch.float32)
            if 'gating' in state_dict:
                self.gating_field = np.array(state_dict['gating'])
            if 'vorticity' in state_dict:
                self.vorticity_field = np.array(state_dict['vorticity'])
            
            self.vorticity = self.get_chern_number()
            variance = float(torch.var(self.phases))
            self.surprisal = -np.log(max(variance, 1e-9) / (np.pi**2 + 1e-9))
            
            v_residue = abs(self.vorticity - round(self.vorticity))
            topo_stability = np.exp(-v_residue * 5.0) 
            self.focus = float(np.mean(self.gating_field))
            self.fidelity = self.focus
            self.causal_integrity = self.focus * (abs(self.vorticity) + 1.0) * topo_stability * 10
            
            DEBUG_LOG("Restored physics state successfully.")
        except Exception as e:
            DEBUG_LOG(f"Error restoring physics state: {e}")

# ==============================================================================
# NOISE MULTIPLEXER
# ==============================================================================
class NoiseMultiplexer:
    def __init__(self):
        self.sources = {'ntp': {'enabled': NTP_AVAILABLE, 'data': deque(maxlen=256)}}
        self.ntp_client = ntplib.NTPClient() if NTP_AVAILABLE else None
        self.last_ntp_sync = 0
        self.ntp_offset = 0.0
        
    def get_blended_noise(self, size=64):
        try:
            with open('/dev/urandom', 'rb') as f:
                data = f.read(size)
            noise = np.frombuffer(data, dtype=np.uint8) / 255.0
            if len(noise) < size:
                noise = np.pad(noise, (0, size-len(noise)), 'wrap')
            return noise
        except:
            return np.random.rand(size)
            
    def get_source_stats(self):
        if self.ntp_client and time.time() - self.last_ntp_sync > 30:
            try:
                resp = self.ntp_client.request('pool.ntp.org', version=3, timeout=1)
                self.ntp_offset = resp.offset
                self.last_ntp_sync = time.time()
            except: pass
        return {'ntp_offset': self.ntp_offset}
    
    def stop(self): pass

# ==============================================================================
# SYNC LEARNER
# ==============================================================================
class SynchronizationLearner:
    def __init__(self, initial_state=None):
        self.history = [] 
        self.best_config = {'offset': 0.0, 'coupling': 1.0}
        self.best_integrity = 0.0
        if initial_state:
            self.history = initial_state.get('history', [])
            self.best_config = initial_state.get('best_config', self.best_config)
            self.best_integrity = initial_state.get('best_integrity', 0.0)
        
        self.theta, self.lam = sp.symbols('theta lambda', real=True)
        self.coeffs = sp.symbols('c0:6', real=True)
        self.model = (self.coeffs[0] + self.coeffs[1] * self.theta + self.coeffs[2] * self.lam + 
                      self.coeffs[3] * self.theta**2 + self.coeffs[4] * self.lam**2 + self.coeffs[5] * self.theta * self.lam)

    def record_trial(self, offset, coupling, integrity):
        self.history.append((offset, coupling, integrity))
        if integrity > self.best_integrity:
            self.best_integrity = integrity
            self.best_config = {'offset': offset, 'coupling': coupling}

    def propose_next_config(self):
        if len(self.history) < 10:
            return {'offset': self.best_config['offset'] + random.uniform(-0.1, 0.1),
                    'coupling': np.clip(self.best_config['coupling'] + random.uniform(-0.2, 0.2), 0.1, 2.0)}
        try:
            pts = np.array(self.history)[-50:]
            if len(pts) < 6: return self.best_config
            X_val, Y_val, J_val = pts[:, 0], pts[:, 1], pts[:, 2]
            A = np.column_stack([np.ones_like(X_val), X_val, Y_val, X_val**2, Y_val**2, X_val*Y_val])
            c_vals, _, _, _ = np.linalg.lstsq(A, J_val, rcond=None)
            J_local = self.model.subs(zip(self.coeffs, c_vals))
            grad_theta = sp.diff(J_local, self.theta)
            grad_lam = sp.diff(J_local, self.lam)
            sol = sp.solve([grad_theta, grad_lam], (self.theta, self.lam))
            if sol and isinstance(sol, dict):
                new_off = float(sol[self.theta])
                new_coup = float(sol[self.lam])
                return {'offset': np.clip(new_off, -1.0, 1.0), 'coupling': np.clip(new_coup, 0.1, 3.0)}
        except: pass
        return {'offset': self.best_config['offset'] + np.random.normal(0, 0.05),
                'coupling': np.clip(self.best_config['coupling'] + np.random.normal(0, 0.1), 0.1, 2.0)}

    def get_state(self):
        return {'history': self.history, 'best_config': self.best_config, 'best_integrity': self.best_integrity}

# ==============================================================================
# ADAPTIVE COMMUNICATOR AGENT
# ==============================================================================
class AdaptiveLoggingCommunicator:
    def __init__(self, adaptive_decoder, holographic_comm, vocab_learner, session_manager):
        self.decoder = adaptive_decoder
        self.holographic_comm = holographic_comm
        self.vocab_learner = vocab_learner
        self.session_manager = session_manager
        self.messages = deque(maxlen=50)
        self.last_text_unitary = 0.0
        
    def process_message(self, text, noise, learner=None):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.messages.append({'type': 'user', 'time': timestamp, 'text': text})
        
        self.vocab_learner.learn_from_input(text)
        with open(self.session_manager.log_file, 'a') as f:
            f.write(f"[{timestamp}] USER: {text}\n")
        
        if self.holographic_comm:
            DEBUG_LOG(f"Imprinting text: '{text[:20]}...'")
            self.last_text_unitary = self.holographic_comm.encode_text(text)
            
        result = self.decoder.decode_cycle(noise, verbose=False)
        response_words = self.decoder.info_to_message(result['new_info'])
        interpretation = result['interpretation']
        
        response = f"{interpretation} {response_words}"
        
        metrics = self.holographic_comm.get_metrics()
        current_ci = metrics.get('causal_integrity', 0.0)
        current_churn = metrics.get('vorticity', 0.0)

        self.messages.append({'type': 'system', 'time': timestamp, 'text': response, 'ci': current_ci, 'churn': current_churn})
        
        if self.messages and self.messages[-2]['type'] == 'user':
             self.messages[-2]['ci'] = current_ci
             self.messages[-2]['churn'] = current_churn
        
        with open(self.session_manager.log_file, 'a') as f:
            f.write(f"[{timestamp}] SYSTEM: {response}\n")
            f.write(f"   [METRICS] CI:{current_ci:.4f} CHERN:{current_churn:.4f}\n\n")
            
        return response
