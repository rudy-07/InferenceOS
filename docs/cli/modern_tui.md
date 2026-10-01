# Modern Interactive Terminal UI (TUI)

InferenceOS 1.1.0 revamps the interactive terminal interface into a next-generation developer console engineered for speed, aesthetic richness, and zero-friction keyboard navigation.

---

## Key UI/UX Capabilities

### 1. Dynamic Floating Slash Autocompleter
Typing `/` dynamically pops up an interactive completion overlay showing all commands, parameter syntax, and live descriptions:
- `/model [path|nickname]`: Switch active model or launch visual picker
- `/mode [system2|system1_reflex|symbiosis]`: Switch cognitive inference mode
- `/tau <float>`: Dynamically adjust confidence escalation threshold (e.g. `/tau 0.90`)
- `/dagger [export|status]`: Manage continuous imitation learning distillation logs
- `/stats`: Display runtime performance telemetry and memory distribution
- `/clear`: Clear conversation chat history
- `/theme [tokyo-night|nord|dracula|matrix|amber]`: Switch visual color palette
- `/help`: Show command cheat sheet
- `/exit`: Terminate chat session

### 2. Split-View Interactive Model Picker Modal
Pressing <kbd>Ctrl+O</kbd> (or running `/model`) displays a full-screen split-view modal dialog:
- **Left Panel**: Searchable list of registered GGUF and System 1 models with live fuzzy filtering.
- **Right Panel**: Real-time parameter preview displaying file size, parameter count, architecture, quantization type, and context length.
- **Controls**:
  - `↑` / `↓`: Navigate model list
  - `Enter`: Load selected model into active runtime session
  - `Esc`: Dismiss picker and return to chat prompt

### 3. Persistent Docked Bottom Status Ribbon
The bottom of the terminal features a live status ribbon displaying:
- Active **System 1** & **System 2** model identifiers
- Current **Cognitive Mode** (`system2`, `system1_reflex`, `symbiosis`)
- Response **Latency** and **Confidence** metric
- Active **$\tau$ Threshold**
- **VRAM / RAM** utilization

### 4. Visual Deliberation Flowcards
Messages are rendered using styled Rich containers:
- **User Cards**: Slate blue headers with crisp prompt framing.
- **System 1 Reflex Cards**: High-speed cyan badges showing classification prediction, confidence progress bars, and millisecond latency.
- **System 2 Reasoning Cards**: Deep purple accents with token generation speed, time-to-first-token (TTFT), and formatted markdown syntax highlighting.

---

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| <kbd>Ctrl</kbd> + <kbd>O</kbd> | Open interactive split-view model picker modal |
| <kbd>Ctrl</kbd> + <kbd>T</kbd> | Cycle cognitive mode (`system2` → `system1_reflex` → `symbiosis`) |
| <kbd>Tab</kbd> | Autocomplete current slash command or model path |
| <kbd>Ctrl</kbd> + <kbd>C</kbd> | Clear current prompt buffer without exiting session |
| <kbd>Ctrl</kbd> + <kbd>D</kbd> | Exit chat session |
