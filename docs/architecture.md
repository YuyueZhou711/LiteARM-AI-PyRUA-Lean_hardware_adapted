# Architecture

This document describes the architecture of `LiteARM-AI-PyRUA-Lean`.

## Context

The repository hosts AI experiments and tooling for LiteArm. The Python side
provides the runtime and integration layer; Lean is used where formal
specification or verification is required. The concrete scope and interfaces are
defined by the first delivery-board request.

## Components

| Component | Responsibility | Depends on |
|---|---|---|
| Python package | Runtime, integration, and experiment tooling | PyRUA, LiteArm SDK |
| Lean components | Formal specifications and verified logic | Lean toolchain |
| Tests and fixtures | Regression coverage for both sides | pytest |

## Data flow

The data flow will be documented once the first workstream is defined. The
intended boundary is that Python owns runtime behaviour and Lean owns the formal
model it is checked against.

## Boundaries

- This repository owns AI tooling, PyRUA integration, and Lean specifications.
- The LiteArm SDK, firmware, and robot description remain in their own repositories.

## Decisions

- Python is the integration and experiment language; Lean is used selectively.
- New components need a delivery-board request before implementation.
