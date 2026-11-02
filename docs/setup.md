# Setup

This guide installs and verifies `LiteARM-AI-PyRUA-Lean` on a development machine.

## Prerequisites

- Python 3.9 or newer
- Git
- A Lean toolchain when working on the Lean components

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## 2. Build

The Python package builds through `pip install -e`; Lean components will document
their own build step when they land.

## 3. Verify

```bash
pytest
```

Expected result: all tests pass.

## Troubleshooting

**`pytest` is not found**

Activate the virtual environment first, or install the test extra with
`pip install -e ".[test]"`.

**Lean commands are unavailable**

Install the Lean toolchain separately and confirm it is on `PATH`.
