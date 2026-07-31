# Getting Help with InferenceOS

Thank you for using **InferenceOS**! We offer several channels for getting support, troubleshooting issues, and engaging with the community.

---

## Documentation First

Before submitting a support ticket, please consult the official documentation:
- **[Quick Start Guide](README.md#quick-start)**: Initial installation and usage.
- **[Documentation Hub](docs/)**: Comprehensive reference manuals.
- **[CLI Reference](docs/cli/command_reference.md)**: Full 21-command syntax breakdown.
- **[Configuration Reference](docs/configuration/settings_reference.md)**: Runtime settings and environment variables.
- **[Troubleshooting FAQ](docs/faq/faq.md)**: Answers to common questions and issues.

---

## Built-in System Diagnostics

InferenceOS includes an interactive diagnostic doctor to verify system drivers, SDKs, and executable binaries:

```bash
# Run automatic diagnostic check
inferenceos doctor
```

To view live hardware parameters and detected backends:

```bash
inferenceos hardware
```

---

## Submitting Issues

If you encounter a bug or have a feature request:
- **Bug Reports**: Open a [Bug Report Issue](https://github.com/InferenceOS/InferenceOS/issues/new?template=bug_report.md). Please include the output of `inferenceos doctor` and `inferenceos hardware`.
- **Feature Requests**: Open a [Feature Request Issue](https://github.com/InferenceOS/InferenceOS/issues/new?template=feature_request.md).
- **Performance Issues**: Open a [Performance Report Issue](https://github.com/InferenceOS/InferenceOS/issues/new?template=performance_report.md).
