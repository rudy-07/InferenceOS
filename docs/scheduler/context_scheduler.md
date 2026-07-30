# Context Window Scheduler (`scheduler/context_scheduler.py`)

The **Context Scheduler** dynamically scales the maximum safe context window length ($N_{ctx}$) based on model architecture, KV cache precision, and real-time available system memory (VRAM and RAM).

---

## Decision Logic

1. **KV Footprint Estimation**:
   $$\text{KV}_{\text{bytes}} = 2 \times N_{\text{layers}} \times D_{\text{hidden}} \times N_{\text{ctx}} \times \text{bytes\_per\_element}$$
2. **Safety Scaling**:
   If requested $N_{ctx}$ (e.g. 128k) exceeds safe VRAM/RAM limits, `ContextScheduler` automatically calculates the largest safe power-of-two or 1024-step context window (e.g. 16384 or 8192) and recommends an appropriate prefill microbatch.

---

## Configuration Reference

- `min_context`: Minimum allowable context window (default: 512).
- `max_context`: Hard upper ceiling (default: 131072).
- `safety_margin`: Fraction of memory reserved as buffer (default: 0.15).
- `manual_override`: Explicit user override length.
