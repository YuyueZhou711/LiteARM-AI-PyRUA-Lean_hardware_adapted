# litearm_posecheck

A no-model wiring/visual check for the dual LiteArm station (RoboDojo
`--env-cfg litearm`, `--action-type joint`). It sends one bent 7-joint pose
through the regular evaluation path and then holds zero, so the 7-value action
keys, the interpolation and the joint targets are exercised. It is not a
policy submission.

```bash
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/litearm_posecheck \
  --task stack_bowls --ckpt no-checkpoint --policy-env pyrualean --eval-env RoboDojo \
  --env-cfg litearm --action-type joint --seed 0 --eval-num 1
```

On an 8 GB GPU the full episode may hit a PhysX GPU out-of-memory error during
setup (Isaac plus three cameras plus the planner). The demo policy run fits; a
run with motion may need a larger card or a lighter camera setup.
