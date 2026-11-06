This guide installs the LiteArm stack on one or two machines for engineers who
need a reproducible simulator and policy-server setup.

# Setup

All commands run from the repository root unless stated otherwise.

## 1. Check prerequisites

| Item | Requirement |
| --- | --- |
| OS | Ubuntu 22.04, x86_64 |
| GPU | NVIDIA RTX with a current driver; 8 GB for the simulator at the reference camera resolution |
| Disk | About 40 GB for Isaac Sim and IsaacLab, tens of GB for the RoboDojo dataset, 3.4 GB for SAM3 |
| Tools | `git`, `git-lfs`, `wget`, `conda` or Miniconda |
| Endpoint | An OpenAI-compatible chat endpoint, its API key, and a model name |

The policy server needs 5-6 GB of GPU memory for SAM3. A single 8 GB GPU cannot
run the simulator and the policy server together; use split hosts or a GPU with
at least 16 GB.

## 2. Clone the repository

```bash
git clone https://github.com/YuyueZhou711/LiteARM-AI-PyRUA-Lean_hardware_adapted.git
cd LiteARM-AI-PyRUA-Lean_hardware_adapted
```

## 3. Install the environments

Single machine:

```bash
bash scripts/setup.sh
```

Split setup:

```bash
# Policy server host
bash scripts/setup.sh --skip-sim

# Simulator host
bash scripts/setup.sh --skip-policy
```

`scripts/setup.sh` runs three steps:

1. `scripts/fetch_sources.sh` clones XPolicyLab, IsaacLab and cuRobo at the revisions recorded by the RoboDojo base commit, applies the LiteArm overlay to XPolicyLab, and links the PyRUA-Lean policy adapter.
2. `robodojo/scripts/install.sh -i` creates the `RoboDojo` conda environment and installs Isaac Sim 5.1, IsaacLab and cuRobo. It installs Miniconda if `conda` is missing and uses `sudo` for `cmake`, `build-essential` and `ffmpeg`.
3. `scripts/setup_policy_env.sh` creates the `pyrualean` conda environment with PyTorch 2.9.1 + cu128, then installs XPolicyLab, PyRUA-Lean and SAM3.

**Do not** install or run the policy server in the `RoboDojo` environment. Its
`websockets` version is 12; the policy server requires 14 or newer. The
`pyrualean` environment is the only one that runs `setup_policy_server.py`.

The full install can take an hour or more. It downloads Isaac Sim and its
dependencies from NVIDIA and PyPI.

## 4. Download assets

Single machine, or simulator host:

```bash
bash scripts/fetch_assets.sh
```

Policy server host:

```bash
bash scripts/fetch_assets.sh --policy-only
```

The script performs these steps:

1. Downloads the RoboDojo asset dataset through `robodojo/scripts/init_assets.sh` (Hugging Face, git-lfs, tens of GB). This step is skipped with `--policy-only`.
2. Copies `assets/litearm/` into `robodojo/Assets/Robots/litearm/` and generates `curobo.yml` from the shipped `curobo_tmp.yml` template. This step is skipped with `--policy-only`.
3. Downloads the SAM3 checkpoint to the policy adapter's `checkpoints/` directory unless `--skip-sam3` is passed. `facebook/sam3` is gated: accept its licence on Hugging Face and run `hf auth login` first.

Use `--skip-sam3` on a simulator-only host, and copy the checkpoint separately
if the policy host has no network access to Hugging Face.

## 5. Configure the model endpoint

```bash
cp config/env.example.sh config/env.sh
```

Edit `config/env.sh` and set:

| Variable | Meaning |
| --- | --- |
| `OPENAI_BASE_URL` | OpenAI-compatible chat endpoint |
| `OPENAI_API_KEY` | Endpoint credential |
| `PYRUALEAN_MODEL` | Model name at that endpoint |

`config/env.sh` is git-ignored. **Do not** commit it or paste the key into an
issue, a script, or a pull request.

## 6. Verify the installation

Install the repository CLI in the environment you use for checks:

```bash
python -m pip install -e .
```

Run the doctor:

```bash
python -m litearm_ai_pyrua_lean doctor
```

The doctor reports the source tree, the cloned XPolicyLab checkout, the
installed LiteArm asset, the SAM3 checkpoint, `config/env.sh`, the endpoint
variables and the conda environments. Missing runtime items are listed
separately from failed install checks.

Run the repository tests:

```bash
pip install -e ".[test]" -e ./pyrua-lean
pytest tests pyrua-lean/tests/test_litearm.py pyrua-lean/tests/test_robots.py
```

Check the simulator installation without launching Isaac Sim:

```bash
bash robodojo/scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy
```

Dry-run the evaluation command:

```bash
bash robodojo/scripts/robodojo.sh eval \
  --policy-dir robodojo/XPolicyLab/policy/PyRUA_Lean \
  --task stack_blocks --ckpt no-checkpoint \
  --policy-env pyrualean --eval-env RoboDojo \
  --env-cfg litearm --action-type joint --seed 0 --eval-num 1 --dry-run
```

## Split-host network

- `scripts/run_server.sh` binds the policy server to `0.0.0.0:18231` by default. Set `BIND_HOST` and `POLICY_PORT` to change this.
- `scripts/run_client.sh` needs `POLICY_HOST` set to the policy server's address.
- Open the policy port only on the network that connects the two hosts.

## Verification status

The commands in this guide encode the manual steps used on the reference machine
(Ubuntu 22.04, RTX 4060 8 GB simulator, Jetson AGX Thor policy server). The
`stack_blocks` and `align_blocks` results and the video recordings were produced
there. A fresh-host run of `scripts/setup.sh` and `scripts/fetch_assets.sh` has
not been verified; report differences with the output of `doctor` and the
failing command.

## Real hardware

The real-hardware station is a separate path with its own environments. It does
not need Isaac Sim, the RoboDojo dataset or the policy-server conda environment.
Follow [LiteArm Real](../pyrua-lean/docs/robots/litearm-real.md): create the
`litearm-hw-device` and `pyrualean-hw-agent` environments from
`pyrua-lean/environments/`, copy the example station and calibration files, and
start the service in its default `read-only` mode first.
