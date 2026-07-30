# Contributing to InferenceOS

Thank you for your interest in contributing to **InferenceOS**! We welcome contributions from systems developers, ML engineers, technical writers, and open-source enthusiasts.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the project maintainers.

---

## How to Contribute

### 1. Reporting Bugs
Before creating a bug report, please check existing issues to avoid duplicates. When submitting a bug report via [GitHub Issues](https://github.com/InferenceOS/InferenceOS/issues), please include:
- A clear description of the bug.
- Steps to reproduce the issue.
- System hardware profile (output of `inferenceos hardware` or `hardware_profile.json`).
- Relevant runtime logs (output of `inferenceos logs`).

### 2. Suggesting Features
We welcome feature proposals! Submit a feature request issue outlining:
- The problem your proposal solves.
- Proposed implementation details or architecture impact.
- Any alternative solutions considered.

### 3. Submitting Pull Requests
1. **Fork the Repository**: Create a fork of `InferenceOS/InferenceOS` on GitHub.
2. **Clone & Setup Environment**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/InferenceOS.git
   cd InferenceOS
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   # source .venv/bin/activate
   pip install -e .
   pip install -r requirements.txt
   ```
3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/my-new-scheduler
   ```
4. **Make & Test Your Changes**:
   - Write clean, well-typed Python code with type annotations (`from __future__ import annotations`).
   - Add unit tests under `tests/` matching your changes.
   - Run the test suite:
     ```bash
     pytest tests/ -v
     ```
5. **Commit Your Changes**: Follow clear commit message conventions:
   ```bash
   git commit -m "feat(scheduler): add dynamic VRAM pressure hysteresis thresholding"
   ```
6. **Push & Open PR**: Push to your fork and submit a Pull Request targeting the `main` branch.

---

## Code Standards & Style Guide

- **Formatting**: We use `black` for code formatting and `ruff` / `flake8` for linting.
- **Type Annotations**: All public classes, methods, and functions must include full type hints.
- **Documentation**: All new modules, classes, and public functions must include Python docstrings (Google or NumPy style).
- **Error Handling**: Prefer explicit, informative exceptions with actionable messages.

---

## Codebase Architecture Overview

For a detailed breakdown of the codebase architecture, see [`docs/developer/architecture_guide.md`](docs/developer/architecture_guide.md).

- `cli/`: 21-command entry points and rich TUI modules.
- `inference_runtime/`: Adaptive runtime engine, backend selector, process manager.
- `scheduler/`: Dynamic Microbatch Scheduler, Context Scheduler, Memory Scheduler.
- `async_scheduler/`: Async pipeline coordinator, CUDA streams, prefetch managers.
- `layer_placement/`: Placement engine & integer linear programming cost models.
- `layer_migration/`: Live hysteresis layer migration engine.
- `kv_manager/`: KV cache eviction (H2O, StreamingLLM) & quantization.
- `runtime_learning/`: SQLite feedback store & predictive recommendation models.
- `server/`: OpenAI & Ollama dual REST API HTTP server.

---

## Developer Tutorials

Want to extend InferenceOS? Check out our step-by-step guides in the documentation tree:
- [Adding a Custom Scheduler](docs/developer/adding_scheduler.md)
- [Adding a New Execution Backend](docs/developer/adding_backend.md)
- [Release Workflow Guide](docs/developer/release_process.md)

---

## License

By contributing to InferenceOS, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
