# Operating knowledge for a calibrated, real dual LiteArm / LiteGrip station (no-VLA)

This run uses `primitives=no-vla`: the VLA primitives are not available.  The
geometric primitives below are the complete plan surface.

## What this robot is (and is not)
- The station is calibrated by the operator; the values below are conventions,
  not numbers to assume.  Read the task card for the devices actually present:
  it may have one or both 7-DoF LiteArm arms (`"left"`, `"right"`), a LiteGrip
  gripper on each arm or none, a fixed head camera, and one wrist camera per
  arm.  `state().active_arms` lists the arms; a camera appears in the card's
  available cameras only when it is configured.
- `perception.primary_camera` (shown as "primary camera" on the task card) is
  the camera used when a perception or image call omits `camera`.  On a
  wrist-only station it is the wrist camera, so `locate(prompt)` works without
  a head camera.
- The instruction is `robo.task`.  Read it before planning and follow it.
- There is **no independent task-success evaluator** on this path.  `robo.done`
  raises, no result carries a task-success flag, and
  `finish(status="success")` is your statement only.  The host records
  `environment_success_known = false` and `task_success = not_measured`.
  Do not claim that the task is proven complete; stop when the measured scene
  and your verification show that it is.

## Frames, units, coordinates
- World origin: the centre of the table edge nearest the robots.  `+x` points
  right, `+y` points into the table, `+z` points up.  The table top is `z = 0`
  (this is the station's calibrated table plane, not a number to add yourself).
  Positions are metres; joint angles and orientation internals are radians.
- The arms' base poses, reachable limits, TCP (grasp-point) transforms, camera
  poses, table extent and obstacles are in the station calibration.  The task
  card reports the measured table extent and current wrist-camera calibration;
  use those, never a remembered height or base coordinate.
- `move_to` drives the **grasp point**, the middle of the LiteGrip finger
  contact region.  The measured pose is `state().left.tcp_pos`; the flange pose
  is also in the observation, but it is not the policy-facing end effector.
- Orientations are `pitch`, `yaw`, `roll` in degrees: pitch 90 points straight
  down, 0 level, -90 straight up; yaw is the heading of the pointing direction
  on the table; roll turns the fingers about the pointing direction.  An angle
  left as `None` keeps its measured value, so a position-only move does not
  rotate the wrist.
- Quaternions in the observation are `[w, x, y, z]`.  Camera extrinsic
  matrices are camera-to-world and use the camera convention: looking down its
  own `-z`, `+y` up.

## Measured state versus command
- Every pose, joint angle and gripper width in `state()` is measured.  A
  commanded target is never reported as measured.  `measured=True` is present
  in the observation.
- `state.step` is an observation snapshot number.  `steps_used` in a primitive
  result counts executed synchronous trajectory segments or gripper actions;
  `robo`'s host `env_steps` is the same device-action count, not simulator
  frames.  They are separate counters.
- A `Move.reached` / `Gripper.settled` result is decided after reading the
  hardware back.  An ACK or a queued command is not arrival.
- The device-action budget is shown in the task card and enforced by the
  toolkit; exceeding it refuses the call.  The episode also has a wall-clock
  and tool-call budget from the host.

## Motion
- The first version executes **synchronous joint segments** (`movej_sync` on
  the SDK).  A straight TCP move is generated as IK waypoints and checked
  before execution; a `path="joint"` move swings the joints.  There is no
  continuous-servo (`move_js`) mode, no raw motor command and no SDK handle in
  the `robo` API.
- `move_to`'s `gripper=robo.OPEN` is applied before the motion.  A
  `gripper=robo.CLOSE` on an empty gripper is not applied during the move
  (the result carries a note); call `set_gripper` after arriving.  `KEEP`
  changes nothing.
- `move_to` returns `reached` only when the measured TCP is within 0.01 m and
  the measured orientation within 5 degrees of the plan, after the final
  segment settles.  `blocked` means the measured arm stopped short of a
  feasible plan; do not push through it.
- The planner checks the calibrated joint limits, the table, configured
  fixed obstacles, self-collision and the other arm's **measured** pose.
  Collision geometry is conservative primitive-based, not mesh-exact; it is
  not a certification of clearance.  Move one arm at a time and keep the idle
  arm out of the way.
- `path="line"` refuses when the straight TCP line is not achievable; use it
  for approaches, descents and lifts near objects.  `path="auto"` falls back to
  a joint swing.  A near miss (up to 2 cm / 8 degrees) may be taken with a
  `note`; read it.
- The measured start is authoritative.  If the plan is refused because the
  state changed or is old, take a fresh `state()` and re-plan; do not retry a
  stale trajectory.
- The service validates every trajectory again against the current measured
  start and the other arm.  A mismatch is a refusal, not a silent adjustment.

## Gripper
- `robo.CLOSE` (1) closes, `robo.OPEN` (-1) opens, `robo.KEEP` (0) keeps the
  last command.  MEASUREMENTS are openings: 1.0 is the calibrated open limit,
  0.0 the calibrated closed limit.
- An arm with no configured gripper reports its `gripper` and `width_m` fields
  as `None`; `set_gripper`/`place_on`, and `move_to`
  with OPEN/CLOSE, raise "this arm has no gripper configured".  That arm can
  still move; never pretend it has a gripper.
- With `OPEN`, `opening=` in `(0, 1]` opens to a fraction of the measured
  closed-to-open span.  The width is the station's calibrated
  `closed_m + (open_m - closed_m) * opening`, not a fixed 2 mm + 85 mm rule.
- A close starts a persistent calibrated-force squeeze.  Asking to close again
  while it holds does not restart the grasp.  `holding` means the measured
  fingers stopped apart; it is a **candidate**, not proof that the intended
  object is held.
- After a lift, verify the object moved with the arm: look at the wrist image,
  or `locate` the object again and compare its position.  A close that stops
  near the closed limit is probably empty; a close whose width is far larger
  than the object may have closed on the gripper body or another object.
- `width_m` is measured across the fingers.  Objects can be compliant; do not
  treat a width as a material-independent size.

## Perception
- `segment` returns a SAM3 mask and centroid.  `back_project` returns a
  measured world point.  `locate` measures an object's centre, top, height and
  footprint.  Omitting `camera` uses `perception.primary_camera`; pass an
  explicit configured camera to override it.
- With valid depth, `locate` uses the mask's depth pixels.  Without depth it
  needs at least two calibrated views; the views' silhouettes are intersected
  on a voxel grid.  One RGB view cannot determine height: the result has
  `measurement_failed=True` and no centre/extents.  Do not substitute a table
  intersection or a fixed height.
- `views` says how many calibrated views agreed.  A `note` or `measurement_quality`
  says when the fit is rough, hidden, too flat or too far.  A mask is the best
  text match, not proof of identity; check the image.
- Occlusion removes points: the measured centre and footprint can be short or
  off-centre.  Do not infer a full object from a partial mask.
- `back_project` at a pixel with invalid depth segments the object under it and
  measures it from the calibrated views; if no surface can be established, it
  raises instead of returning a made-up point.

## Pick and place (first field task: one object into a loose target area)
1. Read `robo.task`, capture `state()` and the current images.
2. `locate` the object; check `measurement_failed`, `views`, `top_z` and the
   footprint.  If height is uncertain, move to a higher observation pose and
   locate again before committing.
3. Choose a grasp point from the **measured** object geometry (top/rim and
   footprint), not from a task constant.  Approach above it with `path="line"`.
4. Descend with `path="line"`; do not close after `blocked` or a descent that
   did not arrive.
5. Close; check `settled`, `holding` and `width_m`.  Lift a few cm with the
   same gripper command.  Verify visually or by re-locating that the intended
   object moved.
6. Move in short segments above the scene.  A carry baseline is compared with
   the measured opening; a large drop is reported as `slip`; stop and recover.
7. At the target, use `place_on` with **measured** `support_top` and the
   measured grasp-point-to-object-bottom distance.  `place_on` opens only
   after a safe descent; `safe_to_home` is true only after the lift and retreat
   both arrived.  Do not use a universal 5 cm offset.
8. `home()` returns to the calibrated safe pose, not the SDK zero pose.  It
   first plans a lift and checks the return path.  Test `reached`; do not claim
   it from a queued/ACK result.

## Stop, faults and recovery
- A normal cancel stops between synchronous segments.  A timeout, lost
  heartbeat or explicit emergency stop calls the firmware emergency stop and
  puts the service into a fault state.  The service never calls `disable`,
  `park` or `home` as a substitute for a controlled stop.
- After a fault, new trajectories are refused and reconnecting does not clear
  the fault or resume motion.  Stop planning, call `finish` if appropriate,
  and let the operator inspect the hardware and restart the service.
- If a tool result says `stop_unconfirmed`, do not treat the arm as stopped;
  do not home or approach.  Report the condition.
- The real emergency-stop hold/disable behaviour and the ability to preempt a
  blocking segment are field-verification items; do not assume them.

## Budgets and reporting
- Keep the device-action budget in mind: each synchronous segment and each
  gripper action counts.  Long lines and repeated approaches spend it
  quickly.  Prefer short, verified moves.
- Record progress with `print`.  `finish(status="success")` is a model claim;
  the host records it separately from execution completion and task success,
  and task success remains unmeasured.
