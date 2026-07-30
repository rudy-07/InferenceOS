# Release Workflow & Build Guide

This document covers the release process and CI workflow for InferenceOS maintainers.

---

## Pre-Release Checklist

1. **Run Full Test Suite**:
   ```bash
   pytest tests/ -v
   ```
2. **Update Version**: Increment version string in `pyproject.toml` and `cli/commands/version_cmd.py`.
3. **Update Changelog**: Document new features, enhancements, and fixes in `CHANGELOG.md`.
4. **Build Distribution Package**:
   ```bash
   python -m build
   ```
5. **Tag Release**:
   ```bash
   git tag -a v1.0.0 -m "InferenceOS v1.0.0 Release"
   git push origin v1.0.0
   ```
