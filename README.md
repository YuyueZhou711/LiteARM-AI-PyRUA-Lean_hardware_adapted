LiteARM AI, PyRUA and Lean tooling for the LiteArm robotic manipulator series.

# LiteARM-AI-PyRUA-Lean

## Scope

This repository hosts AI experiments and tooling for LiteArm, combining Python
(PyRUA) and Lean components. The concrete scope will be defined by the first
delivery-board request.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Usage

```bash
pytest
```

The command-line entry point will be added with the first implementation.

## Documentation

- [Documentation index](docs/index.md)
- [Setup](docs/setup.md)
- [Architecture](docs/architecture.md)

## License

To be decided before the first public release.
