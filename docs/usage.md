# Usage

This guide configures the model endpoint and runs LiteArm evaluations. All
commands run from the repository root.

## Model endpoint configuration

The policy server calls an OpenAI-compatible chat endpoint. Put the endpoint,
key and model name in `config/env.sh`:

```bash
cp config/env.example.sh config/env.sh
```

```bash
# config/env.sh
export OPENAI_BASE_URL="https://your-endpoint.example/v1"
export OPENAI_API_KEY="replace-with-your-key"
export PYRUALEAN_MODEL="your-model-name"
```

`config/env.sh` is git-ignored. **Do not** commit it, print it in logs, or attach
it to an issue. `scripts/run_eval.sh` and `scripts/run_server.sh` source the file
automatically and stop if a variable is missing.

The policy adapter's `deploy.yml` sets `max_turns: 100` and `budget_s: 5400` and
requests `reasoning_effort: high`. Adjust `robodojo/XPolicyLab/policy/PyRUA_Lean/deploy.yml`
to change those limits.

## Single-machine evaluation

```bash
bash scripts/run_eval.sh --task stack_blocks --seed 0
```

The wrapper starts the policy server and the Isaac Sim client on localhost and
passes `--action-type joint`. Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--task` | `stack_blocks` | RoboDojo task name |
| `--seed` | `0` | Layout seed |
| `--env-cfg` | `litearm` | RoboDojo environment configuration |
| `--eval-num` | `1` | Episodes per task |

Extra arguments after `--` go to `robodojo/scripts/robodojo.sh eval`, for example
`bash scripts/run_eval.sh -- --dry-run`.

## Split evaluation

Policy server host:

```bash
bash scripts/run_server.sh
```

This binds `0.0.0.0:18231` by default. Set `BIND_HOST` and `POLICY_PORT` to
change the address. The server needs `config/env.sh` and the SAM3 checkpoint.

Simulator host:

```bash
POLICY_HOST=192.168.1.10 bash scripts/run_client.sh --task stack_blocks --seed 0
```

Set `POLICY_PORT` if the server does not use 18231. The client needs the
RoboDojo environment and the LiteArm assets; it does not need the API key.

## Tasks

| Task | Status |
| --- | --- |
| `stack_blocks` | Verified, seed 0: success rate 1.0, score 100, 1379 steps |
| `cover_blocks` | Demonstrated in the recording; not part of the verified result |
| Other RoboDojo tasks | Runnable with `--env-cfg litearm`, but not verified on LiteArm |

`fold_clothes` needs at least 16 GB of GPU memory for the simulator; the reference
8 GB card fails on Vulkan memory during scene setup. `stack_bowls` is not a good
LiteArm task: the LiteGrip opens to 8.7 cm and the bowls are 12 cm wide, so the
gripper can only catch the rim.

## Outputs

Each episode writes a timestamped directory under:

```text
robodojo/eval_result/RoboDojo/<task>/<policy>/<env_cfg>/<seed and options>/<timestamp>/
```

The directory contains `_result.json`, the policy call log, and images from the
cameras. The `github` directory in this repository was copied from a successful
`stack_blocks` run; the evaluation writes new directories beside it.

## Grasp diagnostics

`tools/litearm/validate_block_grasp.py` drives the production LiteArm tools
without the policy server, using ground-truth object poses. Run it with the
RoboDojo environment's Python:

```bash
conda activate RoboDojo
OMNI_KIT_ACCEPT_EULA=YES python tools/litearm/validate_block_grasp.py \
  --headless --enable_cameras --device cuda:0 --trials 1 --contacts \
  --output /tmp/litearm_grasp.jsonl
```

Exit code 0 means every trial passed the lift and transport checks, 1 means a
grasp threshold failed, and 2 means the simulation or script failed. The script
is a diagnostic; it does not produce a benchmark result.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `ModuleNotFoundError: websockets` or WebSocket handshake errors | Policy server started in the `RoboDojo` environment | Use the `pyrualean` environment; never run the policy server in `RoboDojo` |
| CUDA or Vulkan out-of-memory during scene setup | Too little GPU memory for the cameras or the task | Reduce camera count or resolution, use a larger GPU, or split the hosts |
| SAM3 download returns 401 or 403 | The `facebook/sam3` licence is not accepted | Accept it on Hugging Face, run `hf auth login`, then retry |
| Policy server times out waiting for a model reply | Wrong URL, key or model name; endpoint unreachable | Check `config/env.sh` and the server log |
| `--action-type` error from `robodojo.sh` | The command was called directly without `joint` | Pass `--action-type joint`, or use `scripts/run_eval.sh` |
| `LiteArm USD asset is missing` | `scripts/fetch_assets.sh` was not run, or ran with `--policy-only` | Run `bash scripts/fetch_assets.sh --skip-sam3` on the simulator host |
| cuRobo reports the old checkout path | The repository was moved after `curobo.yml` was generated | Re-run `bash scripts/fetch_assets.sh --skip-sam3` |
| `doctor` reports missing runtime items | Endpoint configuration or the SAM3 checkpoint is absent | Complete steps 4 and 5 in [Setup](setup.md) |
