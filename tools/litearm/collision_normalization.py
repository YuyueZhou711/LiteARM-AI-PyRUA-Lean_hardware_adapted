"""Repair URDF-imported collision schemas without altering mesh geometry."""

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def normalize_collisions(stage, root_path):
    """Author repairs in the current edit target; return migrated collider count.

    Some importer versions attach CollisionAPI to an STL wrapper Xform rather
    than its Mesh. PhysX ignores that wrapper, even after de-instancing it.
    Keep transforms on the wrapper and move collision APIs/properties onto the
    real geometry. The source meshes and physical parameter values are retained.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError(f"Missing robot root: {root_path}")
    # Repeat traversal because clearing an instance exposes its descendants.
    while True:
        instances = [p for p in Usd.PrimRange(root)
                     if p.IsInstance() and "collisions" in str(p.GetPath()).split("/")]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    migrated = 0
    for prim in list(Usd.PrimRange(root)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Gprim):
            continue
        meshes = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise ValueError(f"Collision wrapper has no mesh: {prim.GetPath()}")
        schemas = [s for s in prim.GetAppliedSchemas() if "Collision" in s]
        property_names = set()
        registry = Usd.SchemaRegistry()
        for schema in schemas:
            definition = registry.FindAppliedAPIPrimDefinition(schema)
            if definition:
                property_names.update(definition.GetPropertyNames())
        property_names.update(
            p.GetName() for p in prim.GetAuthoredProperties()
            if p.GetName().startswith("material:binding")
        )
        # Flatten captures composed values, time samples and relationship targets
        # from referenced layers; copy only physics properties, never transforms.
        composed = stage.Flatten()
        for mesh in meshes:
            for schema in schemas:
                mesh.AddAppliedSchema(schema)
            for name in property_names:
                source = prim.GetPath().AppendProperty(name)
                if composed.GetPropertyAtPath(source):
                    destination = stage.GetEditTarget().MapToSpecPath(
                        mesh.GetPath().AppendProperty(name)
                    )
                    if not Sdf.CopySpec(composed, source,
                                        stage.GetEditTarget().GetLayer(), destination):
                        raise RuntimeError(f"Cannot copy {source} to {destination}")
        for schema in schemas:
            prim.RemoveAppliedSchema(schema)
        migrated += 1
    return migrated


def validate_collision_shapes(stage, root_path):
    """Require every enabled collider to be a mesh parsed as a physics shape."""
    expected = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Collider is not a Mesh: {prim.GetPath()}")
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            expected.add(prim.GetPath())
    if not expected:
        raise ValueError("Robot has no enabled collision meshes")
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, [Sdf.Path(root_path)])
    actual = set()
    for kind, (paths, descriptions) in parsed.items():
        if "Shape" in str(kind):
            for path, description in zip(paths, descriptions):
                if not description.isValid:
                    raise ValueError(f"Invalid physics shape: {path}")
                actual.add(path)
    missing = expected - actual
    if missing:
        raise ValueError(f"Physics parser ignored collision meshes: {sorted(map(str, missing))}")
    return len(expected)
