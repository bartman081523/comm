#!/usr/bin/env python3
"""
Experiment 1014ecaa3: SciMind 2.0 Complexity Reduction & Legacy Recovery
Axiom of Causal Integrity (TCI) - Holographic Communication Interface

Upgrades from 1014b/c:
1.  **Topological Imprinting (Braiding)**: Text is not a wave, but a sequence of topological defects (Vortices).
2.  **CNOT Hamiltonian Protection**: Uses TCI-compliant Hamiltonian for system evolution.
3.  **Wick-Rotation PLL**: Synchronizes NTP time (Entropy) with Model Phase (Imaginary) via complex rotation.
4.  **Vorticity/Chern Audit**: Explicit topological charge monitoring.
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
from collections import deque, Counter
from datetime import datetime
import sympy as sp

# Optional deps
try:
    import torch
except ImportError:
    print("CRITICAL: torch not found.")
    sys.exit(1)

try:
    import ntplib
    NTP_AVAILABLE = True
except ImportError:
    NTP_AVAILABLE = False

# ==============================================================================
# SESSION MANAGEMENT (Reused from 1014b)
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
            try:
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                readable = dt.strftime("%Y-%m-%d %H:%M:%S")
                sessions.append({'id': ts, 'path': f, 'label': readable})
            except: pass
        return sessions

    def start_new_session(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_id = ts
        self.state_file = os.path.join(self.log_dir, f"session_{ts}.json")
        self.log_file = os.path.join(self.log_dir, f"session_{ts}.log")
        print(f"Starting NEW session: {self.session_id}")
        return {} 

    def load_session(self, session_path):
        ts = os.path.basename(session_path).replace("session_", "").replace(".json", "")
        self.session_id = ts
        self.state_file = session_path
        self.log_file = os.path.join(self.log_dir, f"session_{ts}.log")
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading state: {e}")
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
                self.user_words[token] += 1 # Preserve capitalization for output
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
# SEMANTIC ADAPTIVE DECODER (1014b Style)
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
        # Coherence
        if len(noise) < 10: noise = np.random.rand(64)
        phases = noise * 2 * np.pi
        order_param = np.abs(np.mean(np.exp(1j * phases)))
        coherence = float(order_param)
        self.last_coherence = coherence
        
        # Gödel Gap
        variance = np.var(noise)
        godel_gap = float(variance * 10.0) 
        
        # Info Slice
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
        import random
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
        self.vorticity = 0.0 # Chern Number
        self.causal_integrity = 0.0
        self.fidelity = 0.0
        
        self.gating_field = np.zeros((N, N))
        self.vorticity_field = np.zeros((N, N))
        
        # TCI Hamiltonian Symbols
        self.t_res, self.Omega = sp.symbols('t_res Omega', real=True)
        self.H_comm = self._derive_hamiltonian()
        
        self.phase_history = deque(maxlen=200)

    def _init_vortex(self, size):
        # Initial Vacuum: A single topological charge at center to start with valid topology
        x = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, x)
        phases = np.mod(np.arctan2(yy, xx) + np.pi, 2 * np.pi)
        return torch.tensor(phases, dtype=torch.float32)

    def _derive_hamiltonian(self):
        """Derive the CNOT-Axiom Hamiltonian: H = (I - Z) ⊗ X (From Exp 1010)"""
        # Symbolic representation for integrity check
        return (self.Omega/2) * sp.Matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])

    def get_chern_number(self):
        """Calculates Vorticity (Topological Texture) via absolute flux summation."""
        p = self.phases.numpy()
        
        # Wrapped phase differences [-pi, pi]
        dx = np.angle(np.exp(1j * (np.roll(p, -1, axis=1) - p)))
        dy = np.angle(np.exp(1j * (np.roll(p, -1, axis=0) - p)))
        
        # Lattice curl (Circulation)
        circ = dx + np.roll(dy, -1, axis=1) - np.roll(dx, -1, axis=0) - dy
        self.vorticity_field = circ / (2 * np.pi) # Local Vorticity Map
        
        # Sum absolute flux / 2pi (Counts Total Vortices + Defects)
        # Matches Exp 1010 'Vorticity' metric
        return float(np.sum(np.abs(circ)) / (2 * np.pi))

    def encode_text(self, text):
        """
        TOPOLOGICAL IMPRINTING:
        Maps text to a Braid Field (Vortices) instead of wave packets.
        """
        if not text: return torch.zeros((self.size, self.size), dtype=torch.float32)
        
        x = np.linspace(-1, 1, self.size)
        xx, yy = np.meshgrid(x, x)
        xx = torch.tensor(xx, dtype=torch.float32)
        yy = torch.tensor(yy, dtype=torch.float32)
        
        braid_field = torch.zeros((self.size, self.size), dtype=torch.float32)
        n_chars = len(text)
        phi = (np.sqrt(5) - 1) / 2 # Golden ratio
        
        for i, char in enumerate(text):
            # Fibonacci spiral distribution for braiding points
            idx = i + 1
            r = np.sqrt(idx / max(n_chars, 1)) * 0.8
            theta = 2 * np.pi * idx * phi
            
            cx = r * np.cos(theta)
            cy = r * np.sin(theta)
            
            # Charge parity based on character code
            charge = 1.0 if ord(char) % 2 == 0 else -1.0
            
            # Add Vortex Phase: q * arctan2(y-cy, x-cx)
            # This is a cumulative phase winding (Braid)
            braid_field += charge * torch.atan2(yy - cy, xx - cx)
            
        return braid_field

    def step(self, external_noise, text_braid_field, ntp_offset=0.0):
        """
        SCIMIND 2.0 (REFINED) STEP:
        Complexity Reduction via Entropy Export, Phase Resonance, and Impedance.
        """
        # --- 0. PRE-CALCULATIONS ---
        p_np = self.phases.numpy()
        N = self.size
        
        # Gradient / Unrest (Gating)
        grad_y, grad_x = np.gradient(p_np)
        unrest_map = np.sqrt(grad_x**2 + grad_y**2)
        # Normalize unrest for usage
        unrest_norm = (unrest_map - np.min(unrest_map)) / (np.max(unrest_map) - np.min(unrest_map) + 1e-6)
        
        # --- 1. HYSTERESIS / ATTENTION BEAM ---
        # "Attention Beam": Only regions with high gradient (novelty/surprisal) or existing high gating get energy.
        # Implements a hysteresis loop: Easy to stay "on", hard to turn "on".
        # If unrest is high, gating increases. If unrest is low, gating decays.
        decay_factor = 0.95
        activation_threshold = 0.6 # Only spikes above this trigger new attention
        
        new_gating = self.gating_field * decay_factor
        # Add new attention where unrest is high
        new_gating += 0.5 * (unrest_norm > activation_threshold).astype(float)
        # Clip
        self.gating_field = np.clip(new_gating, 0.01, 1.0) # Always keep a Pilot Wave pilot light (0.01)
        
        gating_tensor = torch.tensor(self.gating_field, dtype=torch.float32)

        # --- 2. PHASE RESONANCE TUNING (Exp 82) ---
        # Filter external noise. Only resonant frequencies (harmonics of Fundamental) are allowed.
        # Fundamental Freq: ω = 4π / N
        res_freq = 4 * np.pi / N
        
        # Create a "Comb Filter" mask for the noise
        # We assume 'external_noise' is spatial noise. We check its conformity to resonance?
        # Simpler: We modulate the injection based on the LOCAL PHASE aligning with the resonance?
        # Resonance Condition: Phase ~ n * (2pi/k) ? 
        # Actually, let's just enforce that the DRIVER is resonant.
        # We model resonance as a "Preferred Step Size".
        
        if isinstance(external_noise, (int, float)):
             noise_tensor = torch.tensor(external_noise).expand(N, N)
        else:
             if isinstance(external_noise, (list, np.ndarray)):
                 noise_tensor = torch.tensor(external_noise, dtype=torch.float32).view(N, N) if len(external_noise) == N*N else torch.tensor(external_noise[0]).expand(N, N)
             else:
                 noise_tensor = external_noise

        # Resonant Mask: 1.0 if phase is near n*pi/2, else 0.1
        # This creates "Lock-in" points
        phase_res_mask = torch.cos(self.phases * 4) # 4 distinct stable points per cycle?
        # Map -1..1 to 0.1..1.0
        phase_res_mask = (phase_res_mask + 1) / 2 # 0..1
        phase_res_mask = phase_res_mask * 0.9 + 0.1 # 0.1 .. 1.0
        
        resonant_noise = noise_tensor * phase_res_mask

        # --- 3. TOPOLOGICAL IMPEDANCE ---
        # Calculate current vorticity
        current_vorticity = self.get_chern_number()
        # Impedance increases with Complexity (Vorticity)
        # Low Vorticity = Low Impedance (Fluid) -> High Alpha
        # High Vorticity = High Impedance (Solid) -> Low Alpha (Hard to change)
        
        # Base impedance
        base_alpha = 0.1
        # If vorticity is high (e.g. > 5), alpha drops
        impedance_factor = 1.0 / (1.0 + 0.5 * abs(current_vorticity))
        effective_alpha = base_alpha * impedance_factor
        
        # --- 4. INTERACTION & UPDATE ---
        # Text Braiding (User Intent)
        interaction = gating_tensor * text_braid_field
        
        # Wick Rotation (Time Driver)
        wick_rotation = ntp_offset * 5.0 
        
        # Total Force
        # Force = (User Intent + Resonant Noise + Time) - Diffusion
        
        # Laplacian Diffusion (Entropy Export)
        # This smoothes out high-frequency spatial noise (Entropy Export)
        laplacian = (np.roll(p_np, 1, axis=0) + np.roll(p_np, -1, axis=0) + 
                     np.roll(p_np, 1, axis=1) + np.roll(p_np, -1, axis=1) - 4*p_np)
        diffusion = torch.tensor(laplacian, dtype=torch.float32)
        
        # ENTROPY EXPORT: 
        # Stronger diffusion in "quiet" areas (Low Attention) to wipe slate clean
        diffusion_rate = 0.05 + 0.1 * (1.0 - gating_tensor) # Higher diffusion where attention is low
        
        force = (interaction * 1.5) + (resonant_noise * 0.5) + wick_rotation + (diffusion * diffusion_rate)
        
        # UPDATE
        self.phases = (self.phases + effective_alpha * force) % (2 * np.pi)
        
        # --- 5. METRICS & CLEANUP ---
        self.focus = float(np.mean(self.gating_field))
        self.vorticity = self.get_chern_number() 
        
        variance = float(torch.var(self.phases))
        self.surprisal = -np.log(max(variance, 1e-9) / (np.pi**2 + 1e-9))
        
        # CI Recalculation
        v_residue = abs(self.vorticity - round(self.vorticity))
        topo_stability = np.exp(-v_residue * 5.0) 
        # CI rewards High Focus + High Stability + Non-Trivial Vorticity
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
        """Returns visual maps for frontend"""
        return {
            'gating': self.gating_field.tolist() if isinstance(self.gating_field, np.ndarray) else self.gating_field,
            'vorticity': self.vorticity_field.tolist() if isinstance(self.vorticity_field, np.ndarray) else self.vorticity_field
        }

    def get_full_state(self):
        """Returns complete physics state for persistence"""
        return {
            'phases': self.phases.numpy().tolist(),
            'gating': self.gating_field.tolist() if isinstance(self.gating_field, np.ndarray) else self.gating_field,
            'vorticity': self.vorticity_field.tolist() if isinstance(self.vorticity_field, np.ndarray) else self.vorticity_field,
            # We don't save text_braid_field as it's transient/regenerated
        }

    def restore_full_state(self, state_dict):
        """Restores physics state from dictionary"""
        try:
            if 'phases' in state_dict:
                self.phases = torch.tensor(state_dict['phases'], dtype=torch.float32)
            if 'gating' in state_dict:
                self.gating_field = np.array(state_dict['gating'])
            if 'vorticity' in state_dict:
                self.vorticity_field = np.array(state_dict['vorticity'])
            
            # Recalculate metrics to ensure consistency
            self.vorticity = self.get_chern_number()
            variance = float(torch.var(self.phases))
            self.surprisal = -np.log(max(variance, 1e-9) / (np.pi**2 + 1e-9))
            
            # Recalc CI
            v_residue = abs(self.vorticity - round(self.vorticity))
            topo_stability = np.exp(-v_residue * 5.0) 
            self.focus = float(np.mean(self.gating_field)) # approx
            self.fidelity = self.focus
            self.causal_integrity = self.focus * (abs(self.vorticity) + 1.0) * topo_stability * 10
            
            print("Restored physics state successfully.")
        except Exception as e:
            print(f"Error restoring physics state: {e}")

# ==============================================================================
# NOISE MULTIPLEXER (Same as 1014b)
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
        # Nur versuchen, wenn Intervall abgelaufen
        if self.ntp_client and time.time() - self.last_ntp_sync > 30:
            try:
                # VERSUCH (Blockiert max 1s)
                resp = self.ntp_client.request('pool.ntp.org', version=3, timeout=1)
                self.ntp_offset = resp.offset
            except:
                # FEHLER: Trotzdem Zeit aktualisieren, um sofortigen Retry im nächsten Frame zu verhindern!
                # Sonst hängt das System in einer Endlos-Timeout-Schleife.
                pass
            finally:
                # WICHTIG: Timer immer zurücksetzen
                self.last_ntp_sync = time.time()
                
        return {'ntp_offset': self.ntp_offset}
    
    def stop(self): pass

# ==============================================================================
# SYNC LEARNER (Same as 1014b)
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
            import random
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
        
        # 1. Learn
        self.vocab_learner.learn_from_input(text)
        with open(self.session_manager.log_file, 'a') as f:
            f.write(f"[{timestamp}] USER: {text}\n")
        
        # 2. Encode (Topological Imprinting)
        if self.holographic_comm:
            # Result is now a Braid Field (tensor), not just a scalar/wave
            self.last_text_unitary = self.holographic_comm.encode_text(text)
            
        # 3. Decode
        result = self.decoder.decode_cycle(noise, verbose=False)
        response_words = self.decoder.info_to_message(result['new_info'])
        interpretation = result['interpretation']
        
        response = f"{interpretation} {response_words}"
        
        # Capture metrics for this step
        metrics = self.holographic_comm.get_metrics()
        current_ci = metrics.get('causal_integrity', 0.0)
        current_churn = metrics.get('vorticity', 0.0)

        # Append messages WITH metrics
        self.messages.append({'type': 'system', 'time': timestamp, 'text': response, 'ci': current_ci, 'churn': current_churn})
        
        # Also retroactively tag the user message with the state at time of processing?
        # Ideally user message caused the state change, so yes.
        if self.messages and self.messages[-2]['type'] == 'user':
             self.messages[-2]['ci'] = current_ci
             self.messages[-2]['churn'] = current_churn
        
        with open(self.session_manager.log_file, 'a') as f:
            f.write(f"[{timestamp}] SYSTEM: {response}\n")
            f.write(f"   [METRICS] CI:{current_ci:.4f} CHERN:{current_churn:.4f}\n\n")
            
        return response

# ==============================================================================
# TERMINAL UI
# ==============================================================================
class TerminalInterface:
    def __init__(self, stdscr, session_manager, vocab_learner, learner):
        self.stdscr = stdscr
        self.running = True
        self.paused = False
        
        curses.start_color()
        curses.use_default_colors()
        for i, c in enumerate([curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW, 
                               curses.COLOR_RED, curses.COLOR_MAGENTA, curses.COLOR_WHITE], 1):
             curses.init_pair(i, c, -1)

        self.session_manager = session_manager
        self.vocab_learner = vocab_learner
        self.learner = learner
        
        self.decoder = SemanticAdaptiveDecoder(self.vocab_learner)
        self.noise_multiplexer = NoiseMultiplexer()
        self.holographic_comm = SciMindCommunicator(N=40)
        self.text_comm = AdaptiveLoggingCommunicator(self.decoder, self.holographic_comm, self.vocab_learner, self.session_manager)
        
        self.sync_config = self.learner.best_config
        self.metrics = {}
        self.ntp_status = "Init"
        self.input_buffer = ""
        self.physics_lock = threading.Lock()

    def physics_loop(self):
        while self.running:
            if not self.paused:
                with self.physics_lock:
                    noise = self.noise_multiplexer.get_blended_noise(size=self.holographic_comm.size**2)
                    text_u = self.text_comm.last_text_unitary
                    # Decay the braid field effect (Elastic snapback) or keep it?
                    # SciMind says "Imprinting" -> should persist?
                    # But for now, let's zero it after a while or decay it.
                    # We will zero it in step (handled by coupling logic?)
                    # No, we must pass it.
                    # Let's decay the stored field in `text_comm`?
                    
                    stats = self.noise_multiplexer.get_source_stats()
                    base_ntp = stats.get('ntp_offset', 0.0)
                    coup = self.sync_config['coupling']
                    off = self.sync_config['offset']
                    
                    # Effective NTP impact on rotation
                    # Total offset = base + learned_offset
                    total_offset = base_ntp + off
                    
                    self.ntp_status = f"NTP: {base_ntp:+.4f}s | OFF: {off:+.3f} | CPL: {coup:.2f}"
                    
                    self.holographic_comm.step(noise, text_u * coup, ntp_offset=total_offset)
                    self.metrics = self.holographic_comm.get_metrics()
                    
                    # Decay manual text input signal
                    if isinstance(self.text_comm.last_text_unitary, torch.Tensor):
                        self.text_comm.last_text_unitary *= 0.9
                        
                time.sleep(0.03)

    def run(self):
        t = threading.Thread(target=self.physics_loop, daemon=True)
        t.start()
        self.stdscr.nodelay(True)
        self.stdscr.timeout(50)
        
        try:
            while self.running:
                self.update_ui()
                self.handle_input()
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.running = False
            self.stdscr.addstr(0, 0, " SAVING SESSION & EXITING... ", curses.color_pair(1))
            self.stdscr.refresh()
            self.session_manager.save_global_state(self.vocab_learner.get_state(), self.learner.get_state())
            t.join(timeout=1.0)

    def update_ui(self):
        try:
            self.stdscr.erase()
            h, w = self.stdscr.getmaxyx()
            if h < 20 or w < 60: return

            sid = self.session_manager.session_id
            self.stdscr.addstr(0, 0, f" SCIMIND 2.0 (1014e) | SESS: {sid} | {self.ntp_status} ", curses.color_pair(1) | curses.A_REVERSE)
            
            with self.physics_lock:
                ci = self.metrics.get('causal_integrity', 0.0)
                churn = self.metrics.get('vorticity', 0.0)
                best = self.learner.best_integrity
                vocab_size = self.vocab_learner.total_words
                coh = self.decoder.last_coherence
                
                # Highlight Chern Integer
                chern_color = curses.color_pair(2) if abs(churn - round(churn)) < 0.1 else curses.color_pair(3)
                
                self.stdscr.addstr(2, 2, f"INT: {ci:6.3f} (Best: {best:.1f})", curses.color_pair(2 if ci > 5 else 3))
                self.stdscr.addstr(2, 30, f"CHERN NO: {churn:+.3f}", chern_color)
                self.stdscr.addstr(2, 50, f"VOCAB: {vocab_size}", curses.color_pair(5))
                
                bar = "#" * int(min(ci, 20))
                self.stdscr.addstr(3, 2, f"FIELD: [{bar:<20}]", curses.color_pair(1))

            msgs = list(self.text_comm.messages)[-10:]
            y = 5
            for m in msgs:
                pre = ">> " if m['type'] == 'user' else "SYS: "
                col = curses.color_pair(6) if m['type'] == 'user' else curses.color_pair(3)
                text = f"{pre}{m['text']}"[:w-4]
                self.stdscr.addstr(y, 2, text, col)
                y += 1
                
            self.stdscr.addstr(h-2, 2, f"> {self.input_buffer}", curses.color_pair(2))
            self.stdscr.refresh()
        except curses.error: pass

    def handle_input(self):
        try:
            c = self.stdscr.getch()
            if c == -1: return
            if c == 10: 
                if self.input_buffer:
                    with self.physics_lock:
                        noise = self.noise_multiplexer.get_blended_noise(size=64)
                        self.text_comm.process_message(self.input_buffer, noise)
                        
                        metrics = self.holographic_comm.get_metrics()
                        self.learner.record_trial(self.sync_config['offset'], self.sync_config['coupling'], metrics['causal_integrity'])
                        self.sync_config = self.learner.propose_next_config()
                        # AUTO SAVE
                        self.session_manager.save_global_state(self.vocab_learner.get_state(), self.learner.get_state(), self.text_comm.messages)
                        
                    self.input_buffer = ""
            elif c == 27: self.running = False
            elif c in (127, curses.KEY_BACKSPACE): self.input_buffer = self.input_buffer[:-1]
            elif 32 <= c <= 3000: self.input_buffer += chr(c)
        except: pass

def startup_menu(stdscr):
    curses.echo()
    try:
        curses.start_color()
        curses.use_default_colors()
    except: pass
    try: curses.init_pair(1, curses.COLOR_CYAN, -1)
    except: curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    
    mgr = SessionManager()
    sessions = mgr.list_sessions()
    
    stdscr.clear()
    stdscr.addstr(2, 2, "SCIMIND 2.0 (1014e) - Topological Communicator", curses.color_pair(1) | curses.A_BOLD)
    stdscr.addstr(4, 2, "Select Session to Load:")
    stdscr.addstr(5, 4, "[0] Start NEW Session")
    
    for i, s in enumerate(sessions[:9]):
        stdscr.addstr(6+i, 4, f"[{i+1}] {s['label']} (ID: {s['id']})")
        
    stdscr.addstr(16, 2, "Choice: ")
    stdscr.refresh()
    
    try:
        choice = stdscr.getstr(16, 10).decode('utf-8')
        choice = int(choice)
    except: choice = 0
    
    initial_state = {}
    if choice > 0 and choice <= len(sessions):
        initial_state = mgr.load_session(sessions[choice-1]['path'])
    else:
        mgr.start_new_session()
        
    vocab_state = initial_state.get('vocab', {})
    sync_state = initial_state.get('sync', {})
    history = initial_state.get('history', [])
    
    vocab_learner = VocabularyLearner(vocab_state)
    learner = SynchronizationLearner(sync_state)
    
    return mgr, vocab_learner, learner, history

def main(stdscr):
    os.environ.setdefault('ESCDELAY', '25')
    curses.curs_set(0)
    mgr, vocab, learner, history = startup_menu(stdscr)
    interface = TerminalInterface(stdscr, mgr, vocab, learner)
    if history:
        interface.text_comm.messages.extend(history)
    interface.run()

if __name__ == "__main__":
     try:
        curses.wrapper(main)
     except KeyboardInterrupt:
         print("\nExited via KeyboardInterrupt.")
