# Spawn the official UR description into Isaac Sim and publish joint state
# over the ROS 2 bridge. Run with Isaac Sim's Python, for example:
#   ~/isaacsim/python.sh spawn_ur5e.py --xacro /path/ur_isaac.urdf.xacro

import argparse
import os
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--xacro", help="Path to ur_isaac.urdf.xacro")
parser.add_argument(
    "--urdf",
    help="Already expanded URDF, for when xacro is not available here",
)
parser.add_argument("--ur-type", default="ur5e")
parser.add_argument("--joint-commands-topic", default="/isaac_joint_commands")
parser.add_argument("--joint-states-topic", default="/isaac_joint_states")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--cube-xyz",
    type=float,
    nargs=3,
    default=[0.5, 0.0, 0.0403],
    metavar=("X", "Y", "Z"),
    help="Cube centre in base_link (m)",
)
parser.add_argument("--cube-size", type=float, default=0.05, help="Cube edge (m)")
parser.add_argument("--no-cube", action="store_true")
args, _unknown = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"renderer": "RealTimePathTracing", "headless": args.headless}
)

import numpy as np
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from isaacsim.core.experimental.utils import app as app_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Usd, UsdGeom, UsdPhysics

# Rendering runs at 60 Hz, publishing every 4th frame gives 15 Hz
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FRAME = "camera_optical_frame"
CAM_FRAME_SKIP = 3


def expand_urdf() -> str:
    """Expand the official xacro with absolute mesh paths for the importer."""
    if args.urdf:
        return args.urdf
    if not args.xacro:
        raise RuntimeError("Pass --xacro or --urdf.")
    out_dir = tempfile.mkdtemp(prefix="ur5e_isaac_urdf_")
    urdf_path = os.path.join(out_dir, "ur5e.urdf")
    command = [
        "xacro",
        args.xacro,
        f"ur_type:={args.ur_type}",
        "name:=ur",
        "force_abs_paths:=true",
        f"joint_commands_topic:={args.joint_commands_topic}",
        f"joint_states_topic:={args.joint_states_topic}",
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(
            "xacro failed. Source the ROS 2 workspace before launching Isaac Sim."
        )
    with open(urdf_path, "w", encoding="utf-8") as urdf_file:
        urdf_file.write(result.stdout)
    return urdf_path


def articulation_path(hint: str) -> str:
    stage = stage_utils.get_current_stage(backend="usd")
    prim = stage.GetPrimAtPath(hint)
    if prim and prim.IsValid() and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return hint
    if prim and prim.IsValid():
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.ArticulationRootAPI):
                return str(child.GetPath())
    return hint


def find_prim_by_name(root: str, name: str) -> str:
    stage = stage_utils.get_current_stage(backend="usd")
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.GetName() == name:
            return str(prim.GetPath())
    raise RuntimeError(f"No prim named {name!r} under {root}. Is it in the xacro?")


def add_wrist_camera(robot_root: str) -> str:
    optical_path = find_prim_by_name(robot_root, "camera_optical_frame")
    cam_path = optical_path + "/wrist_cam"
    stage = stage_utils.get_current_stage(backend="usd")
    cam = UsdGeom.Camera.Define(stage, cam_path)
    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))  # 180 about X, USD cameras look down -Z with +Y up

    # fx = 640 * 18 / 20.955 = 549.75
    h_ap = 20.955
    cam.GetFocalLengthAttr().Set(18.0)
    cam.GetHorizontalApertureAttr().Set(h_ap)
    cam.GetVerticalApertureAttr().Set(h_ap * CAM_HEIGHT / CAM_WIDTH)
    cam.GetHorizontalApertureOffsetAttr().Set(0.0)
    cam.GetVerticalApertureOffsetAttr().Set(0.0)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    cam.GetProjectionAttr().Set("perspective")
    print(f"Wrist camera: {cam_path}")
    return cam_path


def add_target_cube(xyz, size: float) -> None:
    stage = stage_utils.get_current_stage(backend="usd")
    cube = UsdGeom.Cube.Define(stage, "/World/target_cube")
    cube.GetSizeAttr().Set(size)
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.05, 0.05)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    print(f"Target cube: centre {list(xyz)} m in world/base_link, edge {size} m")


def report_surface_below(x: float, y: float) -> None:
    from omni.physx import get_physx_scene_query_interface

    hits = []

    def on_hit(hit):
        # the arm starts right above the cube
        if not hit.collision.startswith(("/ur5e", "/World/target_cube")):
            hits.append((hit.distance, hit.position[2], hit.collision))
        return True

    get_physx_scene_query_interface().raycast_all((x, y, 2.0), (0.0, 0.0, -1.0), 5.0, on_hit)
    if hits:
        _, z, path = min(hits)
        print(f"Surface under cube at z={z:.4f} ({path})")
    else:
        print("No collider under the cube (it floats; that is fine for the exercise)")


def add_camera_graph(cam_path: str) -> None:
    # Only needs to run once, the helpers set up their own publishing pipeline.
    keys = og.Controller.Keys
    graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": "/WristCameraGraph",
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("RgbPublisher", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("InfoPublisher", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "RenderProduct.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "RgbPublisher.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "InfoPublisher.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "RgbPublisher.inputs:renderProductPath"),
                ("RenderProduct.outputs:renderProductPath", "InfoPublisher.inputs:renderProductPath"),
                ("Context.outputs:context", "RgbPublisher.inputs:context"),
                ("Context.outputs:context", "InfoPublisher.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("Context.inputs:useDomainIDEnvVar", True),
                ("RenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(cam_path)]),
                ("RenderProduct.inputs:width", CAM_WIDTH),
                ("RenderProduct.inputs:height", CAM_HEIGHT),
                ("RgbPublisher.inputs:type", "rgb"),
                ("RgbPublisher.inputs:topicName", "wrist_cam/image_raw"),
                ("RgbPublisher.inputs:frameId", CAM_FRAME),
                ("RgbPublisher.inputs:useSystemTime", False),
                ("RgbPublisher.inputs:frameSkipCount", CAM_FRAME_SKIP),
                ("InfoPublisher.inputs:topicName", "wrist_cam/camera_info"),
                ("InfoPublisher.inputs:frameId", CAM_FRAME),
                ("InfoPublisher.inputs:useSystemTime", False),
                ("InfoPublisher.inputs:frameSkipCount", CAM_FRAME_SKIP),
            ],
        },
    )
    og.Controller.evaluate_sync(graph)


def main() -> None:
    app_utils.enable_extension("isaacsim.ros2.bridge")
    app_utils.enable_extension("isaacsim.asset.importer.urdf")
    simulation_app.update()

    stage_utils.set_stage_units(meters_per_unit=1.0)
    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=np.array([1.6, 1.6, 1.2]),
        target=np.array([0.0, 0.0, 0.4]),
    )

    assets_root = get_assets_root_path()
    if assets_root:
        stage_utils.add_reference_to_stage(
            assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd",
            "/background",
        )

    urdf_path = expand_urdf()
    usd_dir = tempfile.mkdtemp(prefix="ur5e_isaac_usd_")
    importer = URDFImporter(
        URDFImporterConfig(
            urdf_path=urdf_path,
            usd_path=usd_dir,
            fix_base=True,
            merge_fixed_joints=False,
            collision_from_visuals=False,
            joint_drive_type="force",
            joint_target_type="position",
            override_joint_stiffness=100000.0,
            override_joint_damping=10000.0,
            run_asset_transformer=False,
        )
    )
    usd_path = importer.import_urdf()
    robot_root = "/ur5e"
    stage_utils.add_reference_to_stage(usd_path, robot_root)
    simulation_app.update()
    robot_path = articulation_path(robot_root)
    print(f"UR5e articulation: {robot_path}")
    cam_path = add_wrist_camera(robot_root)
    if not args.no_cube:
        add_target_cube(args.cube_xyz, args.cube_size)

    commands_topic = args.joint_commands_topic.lstrip("/")
    states_topic = args.joint_states_topic.lstrip("/")
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ReadJointState", "isaacsim.sensors.physics.IsaacReadJointState"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ReadJointState.inputs:execIn"),
                ("ReadJointState.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("ReadJointState.outputs:jointNames", "PublishJointState.inputs:jointNames"),
                ("ReadJointState.outputs:jointPositions", "PublishJointState.inputs:jointPositions"),
                ("ReadJointState.outputs:jointVelocities", "PublishJointState.inputs:jointVelocities"),
                ("ReadJointState.outputs:jointEfforts", "PublishJointState.inputs:jointEfforts"),
                ("ReadJointState.outputs:jointDofTypes", "PublishJointState.inputs:jointDofTypes"),
                ("ReadJointState.outputs:stageMetersPerUnit", "PublishJointState.inputs:stageMetersPerUnit"),
                ("ReadJointState.outputs:sensorTime", "PublishJointState.inputs:sensorTime"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "SubscribeJointState.outputs:velocityCommand",
                    "ArticulationController.inputs:velocityCommand",
                ),
                (
                    "SubscribeJointState.outputs:effortCommand",
                    "ArticulationController.inputs:effortCommand",
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ArticulationController.inputs:robotPath", robot_path),
                ("ReadJointState.inputs:prim", [usdrt.Sdf.Path(robot_path)]),
                ("PublishJointState.inputs:topicName", states_topic),
                ("SubscribeJointState.inputs:topicName", commands_topic),
            ],
        },
    )
    # Optional on this Isaac Sim build. The joint graph above matches the
    # shipped ROS 2 bridge sample and must still come up if these inputs differ.
    for attr_path, value in (
        ("/ActionGraph/Context.inputs:useDomainIDEnvVar", True),
        ("/ActionGraph/PublishClock.inputs:topicName", "clock"),
    ):
        try:
            og.Controller.set(og.Controller.attribute(attr_path), value)
        except Exception as exc:
            print(f"Left {attr_path} at its default: {exc}")

    add_camera_graph(cam_path)
    simulation_app.update()
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    app_utils.play()
    simulation_app.update()
    if not args.no_cube:
        report_surface_below(args.cube_xyz[0], args.cube_xyz[1])

    while simulation_app.is_running():
        simulation_app.update()

    app_utils.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
