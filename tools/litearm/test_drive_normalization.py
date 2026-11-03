"""Offline regressions for LiteGrip drive units."""

from pathlib import Path
import unittest

from pxr import Sdf, Usd, UsdPhysics

from drive_normalization import normalize_gripper_drives


class DriveNormalizationTests(unittest.TestCase):
    def test_changes_only_slider_drive_type_and_is_idempotent(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/robot")
        arm = UsdPhysics.RevoluteJoint.Define(stage, "/robot/joint1").GetPrim()
        UsdPhysics.DriveAPI.Apply(arm, "angular").CreateTypeAttr("acceleration")
        for side in ("left", "right"):
            joint = UsdPhysics.PrismaticJoint.Define(stage, f"/robot/gripper_slide_joint_{side}")
            joint.CreateLowerLimitAttr(0.0)
            joint.CreateUpperLimitAttr(0.0425)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("acceleration")
            drive.CreateStiffnessAttr(6600.0)
            drive.CreateDampingAttr(130.0)
            drive.CreateMaxForceAttr(17.5)
            drive.CreateTargetPositionAttr().Set(0.01, 1.0)
        before = {str(p.GetPath()): {a.GetName(): (a.Get(), a.GetTimeSamples())
                                    for a in p.GetAuthoredAttributes()
                                    if a.GetName() != "drive:linear:physics:type"}
                  for p in stage.Traverse()}
        self.assertEqual(normalize_gripper_drives(stage, "/robot"), 2)
        for p in stage.Traverse():
            for name, value in before[str(p.GetPath())].items():
                attr = p.GetAttribute(name)
                self.assertEqual((attr.Get(), attr.GetTimeSamples()), value)
        layer = stage.GetRootLayer().ExportToString()
        self.assertEqual(normalize_gripper_drives(stage, "/robot"), 0)
        self.assertEqual(stage.GetRootLayer().ExportToString(), layer)

    def test_rejects_missing_slider(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/robot")
        with self.assertRaisesRegex(ValueError, "two LiteGrip"):
            normalize_gripper_drives(stage, "/robot")

    def test_zeroes_mimic_joint_drive(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/robot")
        joints = {}
        for side in ("left", "right"):
            joint = UsdPhysics.PrismaticJoint.Define(stage, f"/robot/gripper_slide_joint_{side}")
            joint.CreateLowerLimitAttr(0.0)
            joint.CreateUpperLimitAttr(0.0425)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(6600.0)
            drive.CreateDampingAttr(130.0)
            joints[side] = joint
        joints["left"].GetPrim().CreateAttribute(
            "physxMimicJoint:rotX:gearing", Sdf.ValueTypeNames.Float).Set(1.0)
        joints["left"].GetPrim().CreateAttribute(
            "physxMimicJoint:rotX:naturalFrequency", Sdf.ValueTypeNames.Float).Set(25.0)
        joints["left"].GetPrim().CreateAttribute(
            "physxMimicJoint:rotX:dampingRatio", Sdf.ValueTypeNames.Float).Set(0.005)
        normalize_gripper_drives(stage, "/robot")
        left = UsdPhysics.DriveAPI.Get(joints["left"].GetPrim(), "linear")
        right = UsdPhysics.DriveAPI.Get(joints["right"].GetPrim(), "linear")
        self.assertEqual(left.GetStiffnessAttr().Get(), 0.0)
        self.assertEqual(left.GetDampingAttr().Get(), 0.0)
        self.assertEqual(right.GetStiffnessAttr().Get(), 6600.0)
        self.assertEqual(right.GetDampingAttr().Get(), 130.0)
        rel = joints["left"].GetRelationship("physxMimicJoint:rotX:referenceJoint")
        self.assertEqual([str(t) for t in rel.GetTargets()], ["/robot/gripper_slide_joint_right"])
        self.assertEqual(
            joints["left"].GetPrim().GetAttribute("physxMimicJoint:rotX:naturalFrequency").Get(), 0.0)
        self.assertEqual(
            joints["left"].GetPrim().GetAttribute("physxMimicJoint:rotX:dampingRatio").Get(), 0.0)

    def test_follower_without_drive_gets_reference_and_force_on_driver(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/robot")
        left = UsdPhysics.PrismaticJoint.Define(stage, "/robot/gripper_slide_joint_left")
        left.CreateLowerLimitAttr(0.0)
        left.CreateUpperLimitAttr(0.0425)
        left.GetPrim().CreateAttribute(
            "physxMimicJoint:rotX:gearing", Sdf.ValueTypeNames.Float).Set(-1.0)
        right = UsdPhysics.PrismaticJoint.Define(stage, "/robot/gripper_slide_joint_right")
        right.CreateLowerLimitAttr(0.0)
        right.CreateUpperLimitAttr(0.0425)
        drive = UsdPhysics.DriveAPI.Apply(right.GetPrim(), "linear")
        drive.CreateTypeAttr("acceleration")
        normalize_gripper_drives(stage, "/robot")
        self.assertEqual(drive.GetTypeAttr().Get(), "force")
        rel = left.GetRelationship("physxMimicJoint:rotX:referenceJoint")
        self.assertEqual([str(t) for t in rel.GetTargets()], ["/robot/gripper_slide_joint_right"])

    def test_installed_asset_has_force_driver_and_mimic_follower(self):
        path = Path(__file__).resolve().parents[2] / "RoboDojo/Assets/Robots/litearm/litearm.usd"
        if not path.exists():
            self.skipTest("LiteArm asset not installed")
        stage = Usd.Stage.Open(str(path))
        joints = {p.GetName(): p for p in stage.Traverse()
                  if p.GetName().startswith("gripper_slide_joint_")}
        self.assertEqual(len(joints), 2)
        right = joints["gripper_slide_joint_right"]
        left = joints["gripper_slide_joint_left"]
        self.assertEqual(UsdPhysics.DriveAPI.Get(right, "linear").GetTypeAttr().Get(), "force")
        self.assertIsNone(UsdPhysics.DriveAPI.Get(left, "linear"))
        self.assertEqual(left.GetAttribute("physxMimicJoint:rotX:gearing").Get(), -1.0)
        rel = left.GetRelationship("physxMimicJoint:rotX:referenceJoint")
        self.assertEqual([str(t) for t in rel.GetTargets()], [str(right.GetPath())])


if __name__ == "__main__":
    unittest.main()
