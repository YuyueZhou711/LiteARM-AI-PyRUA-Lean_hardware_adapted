#!/usr/bin/env python3
"""Print the LiteArm slider drives and PhysX mimic coupling in the built asset.

Run inside Kit (needs pxr/PhysxSchema):

  OMNI_KIT_ACCEPT_EULA=YES python tools/litearm/inspect_asset_mimic.py [usd]
"""

from __future__ import annotations

import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 320, "height": 240})

from pxr import PhysxSchema, Usd  # noqa: E402

usd = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[2] / "assets" / "litearm" / "litearm.usd"
)
stage = Usd.Stage.Open(str(usd))
root = stage.GetDefaultPrim()
print(f"[inspect] {usd}")
print(f"[inspect] default prim: {root.GetPath()}")
for prim in Usd.PrimRange(root):
    if not prim.GetName().startswith("gripper_slide_joint_"):
        continue
    print(f"joint: {prim.GetPath()}")
    for attr in prim.GetAttributes():
        if attr.GetName().startswith(("drive:", "physxMimicJoint:")):
            print(f"  attr {attr.GetName()} = {attr.Get()}")
    for rel in prim.GetRelationships():
        if rel.GetName().startswith("physxMimicJoint:"):
            print(f"  rel {rel.GetName()} -> {[str(t) for t in rel.GetTargets()]}")
    print(f"  HasAPI PhysxMimicJointAPI: {prim.HasAPI(PhysxSchema.PhysxMimicJointAPI)}")

# Raw layers: which file actually authors the drive type and the relationship.
for layer_path in sorted((usd.parent / "configuration").glob("*.usd")) + [usd]:
    print(f"layer: {layer_path}")
    layer_stage = Usd.Stage.Open(str(layer_path))
    for prim in layer_stage.Traverse():
        if not prim.GetName().startswith("gripper_slide_joint_"):
            continue
        print(f"  prim {prim.GetPath()}")
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(("drive:", "physxMimicJoint:")):
                print(f"    attr {attr.GetName()} = {attr.Get()}")
        for rel in prim.GetRelationships():
            if rel.GetName().startswith("physxMimicJoint:"):
                print(f"    rel {rel.GetName()} -> {[str(t) for t in rel.GetTargets()]}")
app.close()
