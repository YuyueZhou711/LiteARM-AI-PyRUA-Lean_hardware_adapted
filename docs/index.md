Use these pages to install the LiteArm stack, configure the model endpoint and
run a `stack_blocks` evaluation; they are for engineers setting up the simulator
and policy server.

# Documentation

## Start here

- [Setup](setup.md): prerequisites, installation on one or two machines, asset download and verification.
- [Usage](usage.md): model endpoint configuration, single-machine and split evaluation, output locations and common failures.
- [Architecture](architecture.md): the policy server, simulator client, LiteArm asset pipeline and the pinned upstream revisions.

## Scope

This repository combines three upstream projects:

| Project | Role |
| --- | --- |
| PyRUA-Lean | Policy server; writes robot programs and drives the tools |
| RoboDojo | Isaac Sim benchmark and evaluation client |
| XPolicyLab | WebSocket transport and policy adapter framework |

The LiteArm additions are the robot description and assets, the `litearm`
environment configuration, the policy adapter and tools, and the changes needed
to evaluate `stack_blocks` and related tasks.

## Reference

- [Project README](../README.md)
- [Chinese README](../README.zh-CN.md)
- [Third-party notices](../NOTICE)
