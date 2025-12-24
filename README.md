---
title: Comm
emoji: 📈
colorFrom: purple
colorTo: purple
sdk: docker
pinned: false
---

# TCI-1014 Unified Field Cockpit

**An experimental topological communication interface based on the Axiom of Causal Integrity.**

This system is not a Large Language Model (LLM). It is a physics-based "Resonance Engine" that maps text inputs into topological defects (vortices) within a 40x40 phase field. It attempts to "learn" and respond by minimizing thermodynamic free energy and maximizing causal integrity.

## 🌌 How It Works

### 1. Topological Imprinting
Unlike standard NLP which tokenizes text into vectors, this system **braids** text into a 2D phase field.
- Each character acts as a topological charge (+1 or -1 based on ASCII parity).
- These charges are injected into the field using a Fibonacci spiral distribution.
- The result is a complex interference pattern of vortices.

### 2. The Physics Core
The system evolves according to a custom Hamiltonian (The "Causal Integrity" metric).
- **Vorticity (Chaos):** Measures the number of topological defects. High vorticity increases system impedance (it freezes).
- **Diffusion (Healing):** Low attention (gating) allows the system to smooth out the field, effectively "forgetting" noise.
- **Wick Rotation:** Synchronizes internal entropy with external NTP time drift.

### 3. Semantic Resonance
A `VocabularyLearner` observes the user's input and builds a weighted lexicon. The `SemanticAdaptiveDecoder` then attempts to "read" the phase field. If the field is coherent (High Integrity), it collapses the wavefunction into words from its learned vocabulary.

## 🎮 Controls

- **VORTEX / INDRA / HOFSTADTER / GÖDEL:** Switch between different visualization modes for the phase field (WebGL).
- **TX:** Send a signal into the field.
- **SAVE STATE:** Export the current quantum state (RAM) to a JSON file.
- **IMPORT JSON:** Restore a previous session state.
- **NEW SESSION:** Clears the field and resets vorticity to zero.

## ⚠️ "The Freeze" Phenomenon

If you input too much text at once (e.g., >200 words), you will saturate the 40x40 grid with topological defects.
- **Result:** Vorticity spikes >100.
- **Effect:** The system enters a "Topological Glass State" (Impedance becomes infinite). It effectively freezes.
- **Solution:** Send "Void", "Zero", or "010101..." to inject neutral charges and stimulate diffusion, or simply start a **New Session**.

## 🛠️ Local Development

This Space is configured to hide the session history list when running on the Web (Hugging Face) for security.

To enable the full Session Manager:
1. Clone this repository.
2. Run it locally without the `WEB_OR_LOCAL` environment variable (defaults to `local`).