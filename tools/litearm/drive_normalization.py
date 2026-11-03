"""Keep LiteGrip's SDK stiffness and damping in force-drive units."""

from pxr import Usd, UsdPhysics


def _mimic_instance(joint):
    """Return the PhysxMimicJointAPI instance name (e.g. ``rotX``) if this joint
    is a follower, else None.  Checking attribute names avoids importing
    PhysxSchema, which is only available inside Kit."""
    for attr in joint.GetAttributes():
        name = attr.GetName()
        if name.startswith("physxMimicJoint:"):
            return name.split(":")[1]
    return None


def normalize_gripper_drives(stage, root_path):
    """Use force drives on the two sliders without changing limits or geometry.

    The importer defaults to acceleration drives. LiteGrip's configured gains
    are N/m and N*s/m, so an acceleration drive silently scales the resulting
    force by the effective finger mass. The actuator's estimated force then
    overstates the actual contact force.

    A mimic (follower) joint must not also be driven: the PhysX mimic constraint
    alone sets its position, and a nonzero drive would fight it.  The importer
    already omits the follower's drive when parsing ``<mimic>``; this also zeroes
    one if present, and repairs the follower's ``referenceJoint`` relationship,
    which the importer leaves empty.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError(f"Missing robot root: {root_path}")
    joints = [p for p in Usd.PrimRange(root)
              if p.IsA(UsdPhysics.PrismaticJoint)
              and p.GetName().startswith("gripper_slide_joint_")]
    if len(joints) != 2:
        raise ValueError(f"Expected two LiteGrip slider joints, found {len(joints)}")
    changed = 0
    for joint in joints:
        instance = _mimic_instance(joint)
        drive = UsdPhysics.DriveAPI.Get(joint, "linear")
        if drive is not None:
            if drive.GetTypeAttr().Get() != "force":
                drive.CreateTypeAttr("force")
                changed += 1
            if instance:
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(0.0)
        elif not instance:
            # A driven slider must keep its drive; the follower is expected to
            # have none (the importer omits it) because the mimic moves it.
            raise ValueError(f"Missing slider drive: {joint.GetPath()}")
        if instance:
            reference = next(
                (other for other in joints if other.GetPath() != joint.GetPath()), None
            )
            rel = joint.GetRelationship(f"physxMimicJoint:{instance}:referenceJoint")
            if reference is None:
                raise ValueError(f"Unpaired mimic joint: {joint.GetPath()}")
            if not rel:
                rel = joint.CreateRelationship(f"physxMimicJoint:{instance}:referenceJoint")
            if not rel.GetTargets():
                rel.SetTargets([reference.GetPath()])
                changed += 1
            # The importer writes a soft mimic spring (25 Hz, ratio 0.005); it
            # lags the driver during closing and the gripper settles too wide.
            # A zero natural frequency is the rigid constraint (same as leaving
            # the compliance attributes unset, which held 8.8 N per finger).
            for attr_name in ("naturalFrequency", "dampingRatio"):
                attr = joint.GetAttribute(f"physxMimicJoint:{instance}:{attr_name}")
                if attr:
                    attr.Set(0.0)
    return changed
