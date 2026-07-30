# Tiered Memory Scheduler (`scheduler/memory_scheduler.py`)

The **Memory Scheduler** orchestrates multi-tiered memory offloading across GPU VRAM, System RAM, and Swap spaces.

---

## Memory Tiering & Policies

```mermaid
graph TD
    Alloc["Memory Allocation Request"] --> Tier1["Tier 1: GPU VRAM (High Bandwidth ~1TB/s)"]
    Tier1 -- VRAM Full --> Tier2["Tier 2: Host RAM via PCIe (Medium Bandwidth ~50GB/s)"]
    Tier2 -- RAM Full --> Tier3["Tier 3: OS Swap / PageFile (Emergency Buffer)"]
```

### Supported Strategies (`memory_policies.py`)
- **`max_vram`**: Offload maximum possible layers to VRAM; spill remainder to RAM.
- **`balanced`**: Distribute layers proportionally to prevent RAM or VRAM pressure bottlenecks.
- **`conservative`**: Maintain generous VRAM headroom (25%+) for large batching and long context expansions.
