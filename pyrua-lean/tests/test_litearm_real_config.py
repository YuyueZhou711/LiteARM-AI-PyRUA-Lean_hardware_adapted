"""Station config and calibration validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyrualean.hardware.config import (
    ConfigError,
    ExecutionMode,
    load_calibration,
    load_station,
    load_station_and_compare,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "litearm_real.example.json"
EXAMPLE_CAL = ROOT / "configs" / "calibration" / "litearm_real.example.json"


def _station() -> dict:
    return json.loads(EXAMPLE.read_text())


def _calibration() -> dict:
    return json.loads(EXAMPLE_CAL.read_text())


def _write_case(
    tmp_path: Path, station: dict | None = None, calibration: dict | None = None
) -> Path:
    cal = _calibration() if calibration is None else calibration
    cal_path = tmp_path / "calibration" / "site.json"
    cal_path.parent.mkdir(parents=True)
    cal_path.write_text(json.dumps(cal))
    cfg = _station() if station is None else station
    cfg["calibration"] = "calibration/site.json"
    cfg_path = tmp_path / "station.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


def test_examples_load_and_live_is_refused_until_measured():
    config = load_station(EXAMPLE)
    assert config.execution.mode is ExecutionMode.READ_ONLY
    assert config.calibration.executable is False
    assert config.station_id == "litearm-demo-station"
    assert config.sha256 and config.calibration_sha256
    with pytest.raises(ConfigError, match="executable=false"):
        load_station(EXAMPLE, mode=ExecutionMode.LIVE)
    offline = load_station(ROOT / "configs" / "litearm_real.offline.example.json")
    assert offline.execution.mode is ExecutionMode.OFFLINE_FAKE


def test_relative_paths_resolve_from_the_config_file(tmp_path):
    path = _write_case(tmp_path)
    config = load_station(path)
    assert config.calibration_path == (tmp_path / "calibration" / "site.json").resolve()
    assert config.results.directory == str((tmp_path / "runs" / "hardware").resolve())


def test_hash_changes_with_config_and_calibration(tmp_path):
    path = _write_case(tmp_path)
    one = load_station(path)
    two = load_station(path)
    assert one.sha256 == two.sha256 and one.calibration_sha256 == two.calibration_sha256
    station = _station()
    station["execution"]["max_speed"] = 0.25
    path2 = _write_case(tmp_path / "b", station=station)
    other = load_station(path2)
    assert other.sha256 != one.sha256
    assert other.calibration_sha256 == one.calibration_sha256
    cal = _calibration()
    cal["calibration_version"] = "different"
    path3 = _write_case(tmp_path / "c", calibration=cal)
    third = load_station(path3)
    assert third.calibration_sha256 != one.calibration_sha256
    assert third.sha256 == one.sha256
    with pytest.raises(ConfigError, match="hash changed"):
        load_station_and_compare(path3, expected_sha256="0" * 64)


def test_missing_calibration_and_nan_are_refused(tmp_path):
    station = _station()
    station["calibration"] = "missing.json"
    path = tmp_path / "station.json"
    path.write_text(json.dumps(station))
    with pytest.raises(ConfigError, match="does not exist"):
        load_station(path)
    cal = _calibration()
    cal["arms"]["left"]["joint_limits_rad"][0][0] = float("nan")
    path2 = _write_case(tmp_path / "nan", calibration=cal)
    with pytest.raises(ConfigError, match="JSON constant|finite"):
        load_station(path2)


def test_rotation_and_joint_map_validation(tmp_path):
    cal = _calibration()
    cal["cameras"]["cam_head"]["camera_to_world"][0][0] = 2.0
    path = _write_case(tmp_path / "rot", calibration=cal)
    with pytest.raises(ConfigError, match="orthonormal"):
        load_calibration(path.parent / "calibration" / "site.json")
    cal = _calibration()
    cal["arms"]["left"]["joint_map"][1]["sdk_index"] = 0
    path = _write_case(tmp_path / "map", calibration=cal)
    with pytest.raises(ConfigError, match="SDK indices"):
        load_station(path)


def test_limits_and_home_must_be_consistent(tmp_path):
    cal = _calibration()
    cal["arms"]["left"]["joint_limits_rad"][0] = [0.5, 0.4]
    path = _write_case(tmp_path / "limits", calibration=cal)
    with pytest.raises(ConfigError, match="lower >= upper"):
        load_station(path)
    cal = _calibration()
    cal["arms"]["left"]["safe_home_rad"][0] = 9.0
    path = _write_case(tmp_path / "home", calibration=cal)
    with pytest.raises(ConfigError, match="outside"):
        load_station(path)


def test_secret_fields_are_rejected(tmp_path):
    station = _station()
    station["service"]["token"] = "secret"
    path = tmp_path / "station.json"
    path.write_text(json.dumps(station))
    with pytest.raises(ConfigError, match="credential-like"):
        load_station(path)


def test_live_gripper_can_use_the_sdk_factory_calibration(tmp_path):
    cal = _calibration()
    cal["executable"] = True
    cal["quality"] = {"status": "measured"}
    station = _station()
    station["execution"]["mode"] = "live"
    station["devices"]["left"]["gripper"]["calibration_path"] = None
    path = _write_case(tmp_path, station=station, calibration=cal)
    config = load_station(path)
    assert config.arms["left"].gripper.calibration_path is None
    assert config.execution.mode is ExecutionMode.LIVE


def test_calibration_tool_refuses_to_promote_an_example(tmp_path):
    from pyrualean.hardware.calibrate import main as calibrate_main

    out = tmp_path / "cal.json"
    code = calibrate_main(
        [
            "--from",
            str(EXAMPLE_CAL),
            "--out",
            str(out),
            "--measurement-quality",
            "measured",
            "--mark-executable",
        ]
    )
    assert code == 2
    assert not out.exists()


def test_http_sam3_requires_an_endpoint(tmp_path):
    cal = _calibration()
    station = _station()
    station["perception"]["sam3"] = {"mode": "http", "min_score": 0.2}
    path = _write_case(tmp_path, station=station, calibration=cal)
    with pytest.raises(ConfigError, match="endpoint"):
        load_station(path)


def _single_arm_station(arm: str = "left") -> dict:
    """A station whose config mentions only one physically present arm."""

    station = _station()
    station["execution"]["active_arms"] = [arm]
    station["devices"] = {arm: station["devices"][arm]}
    keep = {"cam_head", f"cam_{arm}_wrist"}
    station["cameras"] = {name: cfg for name, cfg in station["cameras"].items() if name in keep}
    return station


def _subset_calibration(
    arms: tuple[str, ...] | None = None, cameras: tuple[str, ...] | None = None
) -> dict:
    """The example calibration trimmed to the station's actual devices."""

    cal = _calibration()
    if arms is not None:
        cal["arms"] = {name: cal["arms"][name] for name in arms}
    if cameras is not None:
        cal["cameras"] = {name: cal["cameras"][name] for name in cameras}
    return cal


def test_active_arms_defaults_to_both_arms():
    config = load_station(EXAMPLE)
    assert config.execution.active_arms == ("left", "right")
    assert config.active_arms == ("left", "right")
    summary = config.summary()
    assert summary["execution"]["active_arms"] == ["left", "right"]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_single_active_arm_may_omit_the_other_arm_and_camera(tmp_path, arm):
    station = _single_arm_station(arm)
    cameras = ("cam_head", f"cam_{arm}_wrist")
    calibration = _subset_calibration(arms=(arm,), cameras=cameras)
    path = _write_case(tmp_path, station=station, calibration=calibration)
    config = load_station(path)
    assert config.execution.active_arms == (arm,)
    assert set(config.arms) == {arm}
    assert set(config.cameras) == {"cam_head", f"cam_{arm}_wrist"}
    assert config.summary()["execution"]["active_arms"] == [arm]


def test_active_arms_validation_errors(tmp_path):
    cases = [
        ([], "at least one arm"),
        (["left", "left"], "must not repeat"),
        (["center"], "must contain only"),
        ("left", "must be a JSON array"),
        ([1], "entries must be strings"),
    ]
    for index, (value, message) in enumerate(cases):
        station = _station()
        station["execution"]["active_arms"] = value
        path = _write_case(tmp_path / str(index), station=station)
        with pytest.raises(ConfigError, match=message):
            load_station(path)


def test_active_arms_must_have_a_devices_entry(tmp_path):
    # Both arms are active and the calibration covers both; only the devices
    # entry is missing, so that is the error the loader must report.
    station = _station()
    station["execution"]["active_arms"] = ["left", "right"]
    station["devices"] = {"left": station["devices"]["left"]}
    path = _write_case(tmp_path, station=station)
    with pytest.raises(ConfigError, match="no devices entry"):
        load_station(path)


def test_devices_may_not_name_unknown_arms(tmp_path):
    station = _station()
    station["devices"]["center"] = station["devices"].pop("right")
    path = _write_case(tmp_path, station=station)
    with pytest.raises(ConfigError, match="unknown"):
        load_station(path)


def test_single_arm_live_checks_only_the_active_port(tmp_path):
    cal = _subset_calibration(arms=("left",), cameras=("cam_head", "cam_left_wrist"))
    cal["executable"] = True
    cal["quality"] = {"status": "measured"}
    station = _single_arm_station("left")
    station["execution"]["mode"] = "live"
    path = _write_case(tmp_path, station=station, calibration=cal)
    config = load_station(path)
    assert config.execution.active_arms == ("left",)

    station = _single_arm_station("left")
    station["execution"]["mode"] = "live"
    station["devices"]["left"]["arm"]["port"] = None
    path = _write_case(tmp_path / "no-port", station=station, calibration=cal)
    with pytest.raises(ConfigError, match="serial port"):
        load_station(path)


def test_live_still_refuses_unmeasured_quality_for_a_single_arm(tmp_path):
    cal = _subset_calibration(arms=("left",), cameras=("cam_head", "cam_left_wrist"))
    cal["executable"] = True
    cal["quality"] = {"status": "unverified"}
    station = _single_arm_station("left")
    station["execution"]["mode"] = "live"
    path = _write_case(tmp_path, station=station, calibration=cal)
    with pytest.raises(ConfigError, match="measured quality"):
        load_station(path)


def test_calibration_subset_is_parsed_without_fabricated_entries(tmp_path):
    station = _single_arm_station("left")
    calibration = _subset_calibration(arms=("left",), cameras=("cam_head", "cam_left_wrist"))
    path = _write_case(tmp_path, station=station, calibration=calibration)
    config = load_station(path)
    assert tuple(config.calibration.arms) == ("left",)
    assert tuple(config.calibration.cameras) == ("cam_head", "cam_left_wrist")

    loaded = load_calibration(
        path.parent / "calibration" / "site.json",
        arms=("left",),
        cameras=("cam_head", "cam_left_wrist"),
    )
    assert set(loaded.arms) == {"left"}
    assert set(loaded.cameras) == {"cam_head", "cam_left_wrist"}

    single = _subset_calibration(arms=("left",), cameras=("cam_left_wrist",))
    single_path = tmp_path / "single_calibration.json"
    single_path.write_text(json.dumps(single))
    single_loaded = load_calibration(single_path, arms=("left",), cameras=("cam_left_wrist",))
    assert set(single_loaded.arms) == {"left"}
    assert set(single_loaded.cameras) == {"cam_left_wrist"}


def test_calibration_tool_stamps_a_single_arm_subset(tmp_path):
    from pyrualean.hardware.calibrate import main as calibrate_main

    source = tmp_path / "trimmed.json"
    source.write_text(json.dumps(_subset_calibration(arms=("left",), cameras=("cam_left_wrist",))))
    out = tmp_path / "out.json"
    code = calibrate_main(
        ["--from", str(source), "--out", str(out), "--measurement-quality", "measured"]
    )
    assert code == 0
    loaded = load_calibration(out, arms=("left",), cameras=("cam_left_wrist",))
    assert set(loaded.arms) == {"left"}
    assert set(loaded.cameras) == {"cam_left_wrist"}
    assert loaded.quality["status"] == "measured"


def test_calibration_must_match_the_station_exactly(tmp_path):
    cameras = ("cam_head", "cam_left_wrist")
    station = _single_arm_station("left")

    missing_arm = _subset_calibration(arms=("left",), cameras=cameras)
    missing_arm["arms"] = {}
    path = _write_case(tmp_path / "no-arm", station=station, calibration=missing_arm)
    with pytest.raises(ConfigError, match="calibration.arms must have exactly"):
        load_station(path)

    extra_arm = _subset_calibration(arms=("left", "right"), cameras=cameras)
    path = _write_case(tmp_path / "extra-arm", station=station, calibration=extra_arm)
    with pytest.raises(ConfigError, match="calibration.arms must have exactly"):
        load_station(path)

    missing_camera = _subset_calibration(arms=("left",), cameras=("cam_head",))
    path = _write_case(tmp_path / "no-camera", station=station, calibration=missing_camera)
    with pytest.raises(ConfigError, match="calibration.cameras must have exactly"):
        load_station(path)

    extra_camera = _subset_calibration(
        arms=("left",), cameras=("cam_head", "cam_left_wrist", "cam_right_wrist")
    )
    path = _write_case(tmp_path / "extra-camera", station=station, calibration=extra_camera)
    with pytest.raises(ConfigError, match="calibration.cameras must have exactly"):
        load_station(path)


def test_opencv_camera_device_and_fourcc_are_parsed(tmp_path):
    station = _station()
    wrist = station["cameras"]["cam_left_wrist"]
    wrist["device"] = "/dev/video4"
    wrist["fourcc"] = "YUYV"
    wrist["index"] = None
    path = _write_case(tmp_path, station=station)
    config = load_station(path)
    camera = config.cameras["cam_left_wrist"]
    assert camera.device == "/dev/video4"
    assert camera.fourcc == "YUYV"
    assert camera.index is None
    summary = config.summary()["cameras"]["cam_left_wrist"]
    assert summary["device"] == "/dev/video4"
    assert summary["fourcc"] == "YUYV"


def test_opencv_device_takes_precedence_and_keeps_the_index(tmp_path):
    station = _station()
    wrist = station["cameras"]["cam_left_wrist"]
    wrist["device"] = "/dev/video4"
    assert wrist["index"] is not None
    path = _write_case(tmp_path, station=station)
    camera = load_station(path).cameras["cam_left_wrist"]
    assert camera.device == "/dev/video4"
    assert camera.index == wrist["index"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cam: cam.update(fourcc="YUV"), "fourcc"),
        (lambda cam: cam.update(fourcc="YU V"), "fourcc"),
        (lambda cam: cam.update(device=""), "non-empty string"),
        (lambda cam: cam.update(device=4), "non-empty string"),
        (lambda cam: (cam.pop("index", None), cam.pop("device", None)), "index or a device path"),
    ],
)
def test_opencv_camera_validation_errors(tmp_path, mutation, message):
    station = _station()
    mutation(station["cameras"]["cam_left_wrist"])
    path = _write_case(tmp_path, station=station)
    with pytest.raises(ConfigError, match=message):
        load_station(path)


def test_device_and_fourcc_are_opencv_only(tmp_path):
    station = _station()
    station["cameras"]["cam_head"]["device"] = "/dev/video4"
    path = _write_case(tmp_path / "device", station=station)
    with pytest.raises(ConfigError, match="only valid for the opencv driver"):
        load_station(path)

    station = _station()
    station["cameras"]["cam_head"]["fourcc"] = "YUYV"
    path = _write_case(tmp_path / "fourcc", station=station)
    with pytest.raises(ConfigError, match="only valid for the opencv driver"):
        load_station(path)


def test_primary_camera_defaults_to_cam_head(tmp_path):
    config = load_station(_write_case(tmp_path))
    assert config.primary_camera == "cam_head"
    assert config.summary()["perception"]["primary_camera"] == "cam_head"


def test_wrist_only_station_names_its_primary_camera(tmp_path):
    station = _single_arm_station("left")
    station["cameras"] = {"cam_left_wrist": station["cameras"]["cam_left_wrist"]}
    station["perception"]["primary_camera"] = "cam_left_wrist"
    calibration = _subset_calibration(arms=("left",), cameras=("cam_left_wrist",))
    path = _write_case(tmp_path, station=station, calibration=calibration)
    config = load_station(path)
    assert config.primary_camera == "cam_left_wrist"

    station = _single_arm_station("left")
    station["cameras"] = {"cam_left_wrist": station["cameras"]["cam_left_wrist"]}
    path = _write_case(tmp_path / "default", station=station, calibration=calibration)
    with pytest.raises(ConfigError, match="primary_camera"):
        load_station(path)


def test_primary_camera_must_be_configured(tmp_path):
    station = _single_arm_station("left")
    station["perception"]["primary_camera"] = "cam_right_wrist"
    calibration = _subset_calibration(arms=("left",), cameras=("cam_head", "cam_left_wrist"))
    path = _write_case(tmp_path, station=station, calibration=calibration)
    with pytest.raises(ConfigError, match="not a configured camera"):
        load_station(path)


@pytest.mark.parametrize("value", ["null", "missing"])
def test_gripper_may_be_absent_for_an_arm_only_station(tmp_path, value):
    station = _single_arm_station("left")
    if value == "null":
        station["devices"]["left"]["gripper"] = None
    else:
        station["devices"]["left"].pop("gripper")
    calibration = _subset_calibration(arms=("left",), cameras=("cam_head", "cam_left_wrist"))
    path = _write_case(tmp_path, station=station, calibration=calibration)
    config = load_station(path)
    assert config.arms["left"].gripper is None
    assert config.gripper_arms == ()
    summary_arm = config.summary()["arms"]["left"]
    assert summary_arm["has_gripper"] is False
    assert summary_arm["gripper"] is None
