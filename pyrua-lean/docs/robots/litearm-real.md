# LiteArm Real hardware path

This page documents the real-hardware adapter for a LiteArm / LiteGrip
workstation: one or two arms, with or without a gripper on each active arm,
and any configured subset of the head and wrist cameras.  It is separate from
the RoboDojo LiteArm simulation adapter:
the real path does not import or start RoboDojo, does not read its tasks,
layouts, assets or `.cache`, and does not write `eval_result`.

## Status at a glance

| Area | Status |
| --- | --- |
| Offline end-to-end: runner → `robo` → toolkit → localhost RPC → fake station → recording | Implemented and tested |
| Adapter, config/calibration validation, FK/IK/TCP, planning, collision primitives | Implemented and tested offline |
| RPC de-duplication, exclusive session, cancel, timeout, disconnect, fault state | Implemented and tested over a real localhost socket |
| RGB-D and multi-view perception | Implemented; tested with synthetic calibrated geometry; SAM3 HTTP endpoint `http://192.168.55.1:8790` verified on 2026-11-05 |
| Vendor SDK wrapper | Tested with `litearm.testing.FakeTransport` and a mock gripper in the device environment |
| Real hardware motion, emergency-stop hold/disable, firmware preemption, calibrated accuracy | **Not run here; field verification required** |
| Real camera capture (OpenCV / RealSense) | Implemented; not connected in this validation |

## Architecture

Two processes:

1. **Runner / agent** (`pyrualean-hw-agent` environment).  It runs the model's
   Python through the normal `robo` API.  `LitearmRealRobot` talks to
   `HardwareToolkit`; the toolkit plans, perceives, records evidence and is the
   RPC client.
2. **Hardware service** (`litearm-hw-device` environment).  It owns the
   configured arms, LiteGrips and cameras.  It validates every trajectory again,
   executes synchronous joint segments, captures measured state, and owns the
   session, stop and fault state machine.

The service listens on `127.0.0.1` by default.  The wire protocol is
newline-delimited JSON over TCP; it can be used across machines later, but the
first version has no TLS/authentication and must be protected by a host firewall
or a secure tunnel if it is not local.  A runner session is exclusive.

```
policy code -> LitearmRealRobot -> HardwareToolkit -> HardwareClient
                                                          |
                                             localhost TCP JSON
                                                          |
HardwareDaemon -> Planner validation -> LiteArmStation -> LiteArm SDK
                                                       -> LiteGrip SDK
                                                       -> CameraManager
```

All hardware SDK and camera imports happen inside the device creation
functions.  Importing `pyrualean.hardware` or `pyrualean.litearm_real` does not
connect a device, start a thread, load a GPU model or require a hardware SDK.

## Verified SDK interfaces and versions

Local sources checked:

- `litearm-python` **2.1.0**, `/home/hzh/Code/litearm-python`, commit
  `5643e47` (git describe `v1.3.0-3-g5643e47`; the package's `__version__` and
  pyproject say 2.1.0).  It is not the same version as the PyPI 1.3.0.
- `litegrip-python` **2.2.0**, local `/home/hzh/Code/NEXTFORM/litegrip-python`,
  commit `6b5fdce` (git describe `v1.2.1-2-g6b5fdce`; the source tree carries
  the semantic-release placeholder `0.0.0`, while the verified install reports
  `2.2.0`).  It is not the PyPI 1.2.1.

Used LiteArm API:

- `Arm.connect()`, `get_state(refresh=True)` (`Msg.value`), `get_tcp()`,
  `get_tool_tcp()`, `set_tcp_offset(xyz, rpy)`;
- `movej_sync(q, speed)` is blocking and uses `speed ∈ [0,1]`; arrival is read
  from measured state.  `movej` is not used for segments because its per-joint
  S-curves make the intermediate Cartesian path unpredictable.
- `emergency_stop()` is a firmware command; `disable()` and `park()` are
  separate and are not treated as controlled stops.
- `move_js` is deliberately not exposed: it needs ≥10 Hz resend and has a
  0.1 s watchdog; the first version has no fixed-rate servo scheduler.
- `get_tcp()` returns the flange pose; `get_tool_tcp()` applies the host-side
  TCP offset.  The station computes TCP from measured joints and the
  calibration, and uses the firmware flange pose as a cross-check.

Used LiteGrip API:

- `LiteGrip(channel, can_id)`, `connect()`, `load_calibration(path=...)`,
  `get_state(wait=False)` with `position_mm`, `force_n`, `fresh`,
  `data_age_s`, `error_code`;
- `grasp(force_n, hold_s=0)` blocks while holding and is ended by
  `release()` from another thread; one persistent grasp thread per gripper
  implements `KEEP` without restarting the squeeze;
- `move_at_speed(target_mm, speed_mm_s)` for open/partial open; `request_release(abort=True)`
  for the emergency path; `stop()` is one zero-torque frame and is not a
  safety-rated stop.

The vendor SDKs are installed from fixed, non-editable wheels.  Build them
with `scripts/build_litearm_device_wheels.sh`; the LiteGrip repository's
semantic-release placeholder is replaced with the verified distribution
version in a temporary build copy.  The inputs are not modified.

## Installation

### Agent environment

```bash
conda env create -f environments/litearm-agent.yml
conda activate pyrualean-hw-agent
python -m pip install -e .          # in this worktree
```

The agent environment has no hardware SDK.  SAM3 runs on the Jetson Thor; the
agent only needs the stdlib HTTP client and the endpoint in the station file.
The agent tests use a scripted/HTTP segmenter and do not need `torch` or the
`sam3` package.

### SAM3 on the Jetson Thor

SAM3 is deployed as an HTTP service on the Jetson Thor.  On Thor (where the
`sam3` package/torch/checkpoint live):

```bash
python src/pyrualean/hosts/sam3_http.py \
  --host 0.0.0.0 --port 8790 \
  --checkpoint /home/nextform/PyRUA/PyRUA-Lean/integrations/xpolicylab/PyRUA_Lean/checkpoints/sam3.pt
```

Check it from the agent machine (the device network is direct, so bypass any
HTTP proxy):

```bash
curl --noproxy '*' http://192.168.55.1:8790/health
# {"ok": true, "served": 0}
```

The agent station file then points at it:

```json
"perception": {
  "sam3": {
    "mode": "http",
    "endpoint": "http://192.168.55.1:8790",
    "timeout_s": 60.0
  }
}
```

`192.168.55.1` is the Thor USB device-network address recorded in the
workstation handoff; if the network changes, update this field.  Connectivity
was checked on 2026-11-05 with `/health` and one point-prompt segmentation over
this address.  The service has no authentication in this version, so restrict
it to the trusted robot network or a tunnel.  A station file with
`mode="local"` and a checkpoint still works on a machine with a local GPU, but
the default hardware example uses the Thor endpoint.

### Device environment

```bash
conda env create -f environments/litearm-device.yml
conda activate litearm-hw-device
LITEARM_SDK_SRC=/path/to/litearm-python \
LITEGRIP_SDK_SRC=/home/hzh/Code/NEXTFORM/litegrip-python \
scripts/build_litearm_device_wheels.sh
python -m pip install --no-index --find-links wheelhouse \
    litearm-python==2.1.0 litegrip-python==2.2.0
python -m pip install --no-deps -e .
```

Do not use an editable install of a vendor SDK on the driving machine.

## Station configuration

Copy `configs/litearm_real.example.json` to a local, git-ignored file (for
example `configs/litearm_real.json`) and copy the example calibration to a real
calibration.  Relative paths in the station file are resolved relative to the
station file.

Key sections:

- `service`: host, port, connect/request timeouts, heartbeat period and
  timeout, maximum clock skew.  Defaults to localhost.
- `execution`: `mode` (`read-only`, `offline-fake`, `live`), optional
  `active_arms` (default `["left", "right"]`; set `["left"]` or `["right"]`
  when only one arm is physically present — the other arm's `devices` entry
  and wrist camera may then be omitted, and it is never opened or commanded),
  speed fraction,
  configured joint/TCP ceilings, segment length, settle tolerance and command
  timeout.  The SDK exposes only a speed fraction; acceleration is owned by the
  firmware and is **not** claimed to be enforced by this layer.
- `devices`: per-arm serial port and minimum firmware, per-gripper CAN channel,
  CAN ID, optional per-gripper calibration path, force, grasp/open speeds.  A
  `gripper` may be `null` or omitted when that active arm has no LiteGrip: the
  station then opens, reads and stops only the arm, `state()` reports
  `gripper`/`width_m` as `None`, and every OPEN/CLOSE or `place_on` request is
  refused instead of driving a made-up gripper.  If `calibration_path` is
  omitted, the SDK's shipped factory calibration is used; a `can1` gripper
  needs its own file because the factory file declares `can0`.
- `cameras`: driver (`opencv`, `realsense`, or `fake` only in offline-fake),
  device index/serial, resolution, FPS, depth flag.  The `opencv` driver also
  accepts `device` (an explicit path such as `/dev/video4`, opened with
  `cv2.CAP_V4L2`; it takes precedence over `index`) and `fourcc` (for example
  `"YUYV"`, set before the resolution/FPS negotiation).  An integer index
  cannot address every V4L2 node.  `device`/`fourcc` are rejected for
  non-opencv drivers, and each opencv camera needs an index or a device path.
- `perception`: SAM3 mode (`http` for the Jetson Thor, or `local`), endpoint
  and timeout, optional local checkpoint/GPU, voxel size, maximum object
  height, minimum valid-depth ratio, and `primary_camera` (default
  `"cam_head"`; must name a configured camera).  The runner's `locate`,
  `segment`, `back_project`, `image` and `camera_meta` use it when the caller
  omits `camera`, so a wrist-only station can call `locate(prompt)`.
- `snapshot`: maximum age, camera/joint synchronization limit, refresh interval,
  retained history.
- `results`: output root, device-action budget, wall-clock budget.
- `safety`: stop timeout, stationary confirmation threshold, home lift height,
  operator-reset policy.
- `fake`: offline-only deterministic settings; ignored by live mode.

`execution.mode=live` is refused unless the calibration has
`"executable": true` and a measured (non-example) quality status for the
selected arms and cameras.  The example is intentionally `executable: false`.

## Calibration

The calibration file defines the station:

- world origin at the centre of the near table edge, +x right, +y into the
  table, +z up, table `z=0`;
- each arm's full base pose, flange-to-TCP transform, joint map
  (`model_joint`, `sdk_index`, sign, zero offset), measured joint limits and
  safe home;
- measured LiteGrip closed/open widths (normalised opening is
  `(width - closed)/(open - closed)`), force, minimum holding width;
- each camera's camera-to-world (head) or camera-to-tool (wrist) transform,
  intrinsic matrix, distortion, resolution and depth flag;
- table extent and fixed obstacles;
- source, timestamp, version and quality.

All matrices are validated (shape, finite values, orthonormal rotation with
determinant +1).  Joint maps must use model joints 0..6 and SDK indices 0..6
exactly once with sign ±1.  Limits must be within the packaged URDF.  The safe
home must be inside the limits.

The calibration must describe **exactly** the station's devices: its `arms`
keys must equal `execution.active_arms`, and its `cameras` keys must equal the
configured `cameras`.  A single-arm, wrist-only station therefore uses a
trimmed calibration file with only the entries it actually has; the loader
neither requires nor fabricates entries for absent devices.  The shipped
two-arm/three-camera example remains valid for the default configuration.
`load_calibration` accepts explicit `arms=`/`cameras=` selectors (default:
all) and enforces the same exact-set rule.

### Measurement procedure (manual)

`tools/litearm_real/calibrate.py` validates and stamps a measurement file.  It
does **not** move an arm; passing `--allow-motion` makes that refusal explicit.
Measure each item by hand and record how:

1. **Base pose.**  With the arm at a known joint configuration, measure the
   base frame origin and axes in the world frame (fiducials or a metrology
   arm).  Write the 4x4 base-to-world transform.  Verify by comparing
   `get_tcp()` (flange) with the computed FK for several poses; disagreement
   over 20 mm is rejected by the observation builder.
2. **TCP.**  Measure the grasp-point position/orientation relative to the
   flange, including the physical LiteGrip mount.  Write `flange_to_tcp`.  Set
   the same offset in the SDK and compare `get_tool_tcp()` with the computed
   TCP at several joint configurations.
3. **Joints.**  For each arm, verify the SDK joint order, direction and zero
   offset against the URDF.  Move one joint at a time by hand (controller
   disabled) or with the operator's low-speed tool, and record the mapping and
   actual soft limits.  Never guess a sign.
4. **Gripper.**  The LiteGrip SDK ships `src/litegrip/factory_calibration.json`
   (`calibrated: true`, `channel: can0`), so a `can0` gripper needs no extra
   file: a no-argument `load_calibration()` uses it.  The SDK deliberately
   refuses to reuse that file implicitly on `can1`; run the vendor
   `zero()`/`calibrate_guided()`/`calibrate_manual()` for that unit and point
   `devices.<arm>.gripper.calibration_path` at its
   `~/.litegrip/<channel>_calibration.json`.  Measure the closed and fully open
   jaw width with a caliper and verify that a commanded `goto`/`move_at_speed`
   stops within the calibrated travel.  Force is a nominal SDK scaling, not a
   calibrated force; do not use it for safety decisions.  `calibrate.py` below
   stamps the **station** calibration (base/TCP/camera), not this vendor
   gripper file.
5. **Cameras.**  Measure intrinsics with a target and distortion model; measure
   camera-to-world for the fixed head and camera-to-tool for each wrist camera.
   Confirm the image resolution matches the calibration.  For RealSense, the
   driver rejects a device intrinsic matrix that differs from the calibration
   by more than 2% (or 4 px for the principal point).  The RealSense colour
   stream is requested as `rs.format.bgr8` and converted to RGB before it is
   returned: D435i firmware 5.17.0.9 timed out on an `rgb8` colour stream at
   640x480@30 while `bgr8` delivered colour and depth reliably.
   For an OpenCV camera whose V4L2 node cannot be opened by integer index, set
   `cameras.<name>.device` to the path (for example `/dev/video4`) and, when
   the node needs it, `fourcc` (for example `"YUYV"`).  The driver opens the
   path with `cv2.CAP_V4L2` and sets the FourCC before width/height/FPS.
6. **Home.**  Choose a collision-free joint configuration, record it as
   `safe_home_rad`, and verify the lift-and-return path with the planner.  This
   is not the SDK's zero pose.

Replace every example value in a working copy, set its `source` to the actual
measurement record and `quality.status` to a measured status, then stamp it:

```bash
python tools/litearm_real/calibrate.py \
  --from /abs/path/to/measured_calibration.json \
  --out configs/calibration/litearm_real.json \
  --measurement-quality measured \
  --mark-executable
```

The tool refuses to mark a file executable while its input still carries an
`example`, `template` or `unverified` marker.  Only mark `executable=true`
after every section is measured and checked.  The loader refuses a live station
with an unverified/example calibration.

## Starting the service

```bash
LITEARM_HW_CONFIG=/abs/path/station.json \
LITEARM_HW_MODE=live \
LITEARM_HW_LOG=/abs/path/service.log \
scripts/litearm_hardware_service.sh
```

For offline verification use `configs/litearm_real.offline.example.json` (or a
copy) and `LITEARM_HW_MODE=offline-fake`.
The service:
- never enables, resets, clears faults, or homes an arm/gripper automatically;
- refuses to start live if an arm/gripper is disabled or faulted (enable and
  clear faults outside the service after inspection);
- owns one runner session at a time;
- keeps read-only snapshots available, but read-only mode refuses motion.

## Running an episode

```bash
LITEARM_HW_CONFIG=/abs/path/station.json \
LITEARM_HW_INSTRUCTION='put the red block in the tray' \
LITEARM_HW_MODE=live \
LITEARM_HW_BUDGET_S=1200 \
scripts/litearm_hardware_play.sh
```

Equivalent direct command:

```bash
pyrualean play \
  --robot litearm_real \
  --arm cells \
  --primitives no-vla \
  --hardware-config /abs/path/station.json \
  --task pick_place \
  --instruction "put the red block in the tray" \
  --out /abs/path/results_hw/run_id
```

The adapter is selected only by `--robot litearm_real`; no environment variable
changes the default LiteArm simulation backend.  `read-only` still connects and
perceives but refuses all motion.  `offline-fake` uses the synthetic station and
cameras; it is never selected because a real driver failed.  `live` requires
an executable calibration with measured quality for the selected active arms
and configured cameras.

## Stop semantics

- **Cancel**: confirmed at a synchronous segment boundary with a fresh state
  read; the arm holds its measured pose and the service enters `stopped`.  A
  new trajectory requires restarting the service.
- **Timeout / heartbeat loss / explicit emergency**: the service calls the
  firmware emergency stop and enters `fault`.  The result records whether the
  stop could be confirmed; an unconfirmed stop is reported as
  `stop_unconfirmed`, not success.
- The service does not call `disable()` or `park()` as a controlled stop.
- A fault never clears itself.  Reconnecting does not resume motion.
- Real emergency-stop hold/disable behaviour and preemption of a blocking
  segment are **field-verification items**.

## Results and diagnostics

An episode directory contains:

- `prompt.txt`, `calls.jsonl`, `tool_turns.json`, `result.json`;
- `steps/NNNNNN/` snapshots: `snapshot.json`, `observation.json`,
  `camera_meta.json`, `device.json`, and `artifacts/...` PNG/NumPy files;
- `episode.json` with the stop evidence and success semantics;
- `hardware_host.json`, `card/` (images and `card.json`).

`State.step` is the snapshot number.  `steps_used`/`env_steps` count executed
trajectory segments or gripper actions.  `environment_success` is false and
`environment_success_known` is false; `task_success` is `not_measured`.  A
model's `finish(status="success")` is recorded separately and is not a task
verdict.

## Diagnostics

- Use `tools/litearm_real/smoke.py` for read-only checks of the handshake,
  hashes, clock, snapshots, TCP and stop evidence.  With `--allow-motion` it
  can run a small motion and a single-object pick/place; the live-motion path
  was **not** run in this repository's validation.
- Use the service log for state read failures, fault reasons and stop
  confirmation.
- Check `steps/*/camera_meta.json` and `device.json` for consistent capture
  metadata.
- `camera_meta` for `step=-1` and `image(..., step=-1)` come from the same
  pinned snapshot; a later state refresh creates a new snapshot.

## Tests

Agent environment (no hardware SDK; no local SAM3/torch):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_litearm.py tests/test_litearm_real.py \
  tests/test_litearm_real_config.py tests/test_litearm_real_geometry.py \
  tests/test_litearm_real_perception.py tests/test_litearm_real_execution.py \
  tests/test_litearm_real_cameras.py tests/test_cli.py -q
```

Device environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_litearm_real_devices.py -q
```

The device test uses `litearm.testing.FakeTransport` and a mock LiteGrip; it is
not physical evidence.  The execution tests use a real localhost TCP socket and
the synthetic station, not mocked client methods.

## Known limits / field verification

- All physical motion, gripper force, emergency stop, firmware behavior, cable
  timing and calibrated accuracy remain to be verified on the workstation.
- Collision geometry is primitive-based and conservative; it is not mesh-exact
  or certified.
- The SDK exposes a speed fraction, not acceleration; the configured segment
  length and timeouts are runtime guards, not a proof of dynamic limits.
- The service has no TLS/authentication in this version.  Keep it on localhost
  or a protected network.
- SAM3 is expected on the Jetson Thor; the examples point at the verified
  address above, but the HTTP service must be running.  If it is unreachable,
  perception calls fail explicitly.  A local `sam3`/checkpoint remains an
  optional fallback.
- RealSense support requires `pyrealsense2` and matching firmware; no camera
  was connected here.
- The example config/calibration are not executable.  Do not set
  `executable=true` without measurements.

## API reference

<!-- api-reference: litearm_real primitives=no-vla -->
