# Runtime Knowledge Base (RKB) (`runtime_kb/`)

The **Runtime Knowledge Base (RKB)** indexes hardware fingerprints and model architectures to build a localized knowledge graph of optimal execution parameters.

---

## Fingerprinting & Lookup

- **`HardwareFingerprint` (`hardware.py`)**: Computes a stable hash based on GPU name, driver version, VRAM size, and backend support.
- **`ModelFingerprint` (`models.py`)**: Hashes model architecture name, total layer count, hidden dimension size, and vocabulary size.
- **`KnowledgeBase` (`knowledge_base.py`)**: Queries stored execution records to quickly provide optimal layer placement plans without needing full re-benchmarking.
