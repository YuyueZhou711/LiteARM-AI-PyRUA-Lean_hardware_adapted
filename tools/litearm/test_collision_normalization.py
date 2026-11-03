"""Offline USD regression tests; run with Isaac Sim's pxr on PYTHONPATH."""

from pathlib import Path
import unittest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from collision_normalization import normalize_collisions, validate_collision_shapes


class CollisionNormalizationTests(unittest.TestCase):
    def make_stage(self):
        stage = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(stage, "/robot")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        wrapper = UsdGeom.Xform.Define(stage, "/robot/collisions/stl")
        wrapper.AddTranslateOp().Set((0.1, 0.2, 0.3))
        collision = UsdPhysics.CollisionAPI.Apply(wrapper.GetPrim())
        collision.CreateCollisionEnabledAttr(True)
        collision.CreateSimulationOwnerRel().SetTargets([Sdf.Path("/scene")])
        UsdPhysics.Scene.Define(stage, "/scene")
        UsdPhysics.MeshCollisionAPI.Apply(wrapper.GetPrim()).CreateApproximationAttr("convexHull")
        wrapper.GetPrim().CreateRelationship("material:binding:physics").SetTargets([Sdf.Path("/material")])
        mesh = UsdGeom.Mesh.Define(stage, "/robot/collisions/stl/mesh")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
        return stage, wrapper.GetPrim(), mesh.GetPrim()

    def test_migrates_properties_without_geometry_changes(self):
        stage, wrapper, mesh = self.make_stage()
        points = mesh.GetAttribute("points").Get()
        transform = UsdGeom.Xformable(wrapper).GetLocalTransformation()
        self.assertEqual(normalize_collisions(stage, "/robot"), 1)
        self.assertFalse(wrapper.HasAPI(UsdPhysics.CollisionAPI))
        self.assertFalse(wrapper.HasAPI(UsdPhysics.MeshCollisionAPI))
        self.assertEqual(mesh.GetAttribute("physics:approximation").Get(), "convexHull")
        self.assertTrue(mesh.GetAttribute("physics:collisionEnabled").Get())
        self.assertEqual(mesh.GetRelationship("physics:simulationOwner").GetTargets(), [Sdf.Path("/scene")])
        self.assertEqual(mesh.GetRelationship("material:binding:physics").GetTargets(), [Sdf.Path("/material")])
        self.assertEqual(mesh.GetAttribute("points").Get(), points)
        self.assertEqual(UsdGeom.Xformable(wrapper).GetLocalTransformation(), transform)
        self.assertEqual(validate_collision_shapes(stage, "/robot"), 1)
        layer = stage.GetRootLayer().ExportToString()
        self.assertEqual(normalize_collisions(stage, "/robot"), 0)
        self.assertEqual(stage.GetRootLayer().ExportToString(), layer)

    def test_rejects_geometryless_collision_wrapper(self):
        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Xform.Define(stage, "/robot").GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        with self.assertRaisesRegex(ValueError, "no mesh"):
            normalize_collisions(stage, "/robot")

    def test_repairs_instanceable_collision_reference(self):
        stage, _, _ = self.make_stage()
        layer = stage.GetRootLayer()
        Sdf.CopySpec(layer, Sdf.Path("/robot/collisions"), layer, Sdf.Path("/library"))
        stage.RemovePrim("/robot/collisions")
        reference = stage.DefinePrim("/robot/collisions")
        reference.GetReferences().AddInternalReference("/library")
        reference.SetInstanceable(True)
        self.assertTrue(reference.IsInstance())
        self.assertEqual(normalize_collisions(stage, "/robot"), 1)
        self.assertFalse(reference.IsInstance())
        self.assertEqual(validate_collision_shapes(stage, "/robot"), 1)

    def test_real_asset_all_links_and_fingers_parse_as_shapes(self):
        asset = Path(__file__).resolve().parents[2] / "RoboDojo/Assets/Robots/litearm/litearm.usd"
        if not asset.exists():
            self.skipTest("LiteArm asset not installed")
        stage = Usd.Stage.Open(str(asset))
        # Never write the installed asset during a regression test.
        stage.SetEditTarget(stage.GetSessionLayer())
        root = stage.GetDefaultPrim().GetPath()
        normalize_collisions(stage, root)
        self.assertEqual(validate_collision_shapes(stage, root), 12)
        colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        finger_colliders = [p for p in colliders if "gripper_slider_link" in str(p.GetPath())]
        self.assertEqual(len(finger_colliders), 2)
        self.assertTrue(all(len(UsdGeom.Mesh(p).GetPointsAttr().Get()) > 0 for p in colliders))
        self.assertEqual(normalize_collisions(stage, root), 0)


if __name__ == "__main__":
    unittest.main()
