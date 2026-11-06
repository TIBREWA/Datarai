#!/usr/bin/env python3
"""Move the arm around the cube, save an image and the camera pose at each stop.

    ros2 run ur5e_isaac_bridge capture_wrist_images.py --out captures

To just aim the camera somewhere without saving:
    ros2 run ur5e_isaac_bridge capture_wrist_images.py --eye 0.5 0.3 0.5 --look-at 0 0 0.15 --no-save
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PlanningScene,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

GROUP = "ur_manipulator"
BASE = "base_link"
TOOL = "tool0"
CAM = "camera_optical_frame"
ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


MIN_CUBE_PIXELS = 3000  # the whole cube is about 9000+ px at these distances
MAX_IK_TRIES = 3


def red_pixels(bgr) -> int:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 40), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 40), (180, 255, 255))
    return cv2.countNonZero(mask)


def quat_to_matrix(x, y, z, w) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r: np.ndarray):
    # returns x, y, z, w
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    w = (r[k, j] - r[j, k]) / s
    return (q[0], q[1], q[2], w)


def transform_to_matrix(tf) -> np.ndarray:
    m = np.eye(4)
    q, t = tf.transform.rotation, tf.transform.translation
    m[:3, :3] = quat_to_matrix(q.x, q.y, q.z, q.w)
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def look_at_optical(eye, target, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """T_base_cam for a camera at eye looking at target, image kept upright."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    down = -np.asarray(world_up, float)
    x = np.cross(down, z)
    if np.linalg.norm(x) < 1e-6:  # straight down, cross product is zero
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.column_stack([x, y, z])
    m[:3, 3] = eye
    return m


def viewpoint(target, radius, azimuth_deg, elevation_deg):
    # az from +x in the base x-y plane, el up from horizontal
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    t = np.asarray(target, float)
    return t + radius * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


class WristCapture(Node):
    def __init__(self):
        super().__init__("wrist_capture", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.last_image: Image | None = None
        self.camera_info: CameraInfo | None = None
        self.joint_history: list[tuple[float, dict]] = []
        self.create_subscription(Image, "/wrist_cam/image_raw", self._on_image, 5)
        self.create_subscription(CameraInfo, "/wrist_cam/camera_info", self._on_info, 5)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 50)
        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.move = ActionClient(self, MoveGroup, "/move_action")

    def _on_image(self, msg):
        self.last_image = msg

    def _on_info(self, msg):
        self.camera_info = msg

    def _on_joints(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.joint_history.append((t, dict(zip(msg.name, msg.position))))
        del self.joint_history[:-400]

    def spin_until(self, predicate, timeout_s=30.0, what="condition"):
        deadline = time.monotonic() + timeout_s
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for {what}")
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_ready(self):
        self.get_logger().info("Waiting for camera, joint states, TF, IK and move_group...")
        self.spin_until(lambda: self.last_image is not None and self.camera_info is not None, 60, "camera topics")
        self.spin_until(lambda: len(self.joint_history) > 0, 30, "/joint_states")
        self.spin_until(lambda: self.tf_buffer.can_transform(TOOL, CAM, Time()), 30, "camera TF")
        if not self.ik.wait_for_service(timeout_sec=30) or not self.move.wait_for_server(timeout_sec=30):
            raise RuntimeError("move_group services are not available")
        self.scene.wait_for_service(timeout_sec=10)

    def current_joints(self) -> dict:
        return dict(self.joint_history[-1][1])

    def lookup(self, parent, child, stamp=Time()) -> np.ndarray:
        self.spin_until(lambda: self.tf_buffer.can_transform(parent, child, stamp), 10, f"TF {parent}->{child}")
        return transform_to_matrix(self.tf_buffer.lookup_transform(parent, child, stamp))

    def add_scene_objects(self, cube_xyz, cube_size, table_top_z):
        objects = []
        cube = CollisionObject()
        cube.header.frame_id = BASE
        cube.id = "target_cube"
        cube.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[cube_size] * 3))
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, cube_xyz)
        p.orientation.w = 1.0
        cube.primitive_poses.append(p)
        cube.operation = CollisionObject.ADD
        objects.append(cube)

        table = CollisionObject()
        table.header.frame_id = BASE
        table.id = "table"
        # base sits ~1.5 cm into the simple_room table, so keep the slab below z=0
        # or the base is always in collision
        table.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[2.0, 2.0, 0.02]))
        tp = Pose()
        tp.position.z = min(float(table_top_z), -0.005) - 0.01
        tp.orientation.w = 1.0
        table.primitive_poses.append(tp)
        table.operation = CollisionObject.ADD
        objects.append(table)

        req = ApplyPlanningScene.Request()
        req.scene = PlanningScene(is_diff=True)
        req.scene.world.collision_objects = objects
        fut = self.scene.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10)
        self.get_logger().info("Planning scene: added target_cube and table")

    def solve_ik(self, tool_pose: np.ndarray, extra_seeds: int = 8) -> list[list[float]]:
        # KDL gives the branch nearest its seed. OMPL kept failing on far branches,
        # so try a few seeds and return the distinct solutions, closest first.
        current = np.array([self.current_joints()[j] for j in ARM_JOINTS])
        rng = np.random.default_rng(0)
        seeds = [current] + [current + rng.uniform(-1.5, 1.5, 6) for _ in range(extra_seeds)]
        found = []
        for seed in seeds:
            sol = self._ik_once(tool_pose, seed)
            if sol is not None:
                found.append((float(np.max(np.abs(np.asarray(sol) - current))), sol))
        solutions = []
        for _, sol in sorted(found, key=lambda f: f[0]):
            if not any(np.max(np.abs(np.subtract(sol, other))) < 0.05 for other in solutions):
                solutions.append(sol)
        return solutions

    def _ik_once(self, tool_pose: np.ndarray, seed) -> list[float] | None:
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = GROUP
        r.ik_link_name = TOOL
        r.avoid_collisions = True
        r.timeout = Duration(seconds=0.2).to_msg()
        r.robot_state.joint_state.name = ARM_JOINTS
        r.robot_state.joint_state.position = [float(v) for v in seed]
        ps = PoseStamped()
        ps.header.frame_id = BASE
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, tool_pose[:3, 3])
        qx, qy, qz, qw = matrix_to_quat(tool_pose[:3, :3])
        ps.pose.orientation.x, ps.pose.orientation.y = float(qx), float(qy)
        ps.pose.orientation.z, ps.pose.orientation.w = float(qz), float(qw)
        r.pose_stamped = ps
        fut = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5)
        res = fut.result()
        if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return [sol[j] for j in ARM_JOINTS]

    def move_to_joints(self, joints, attempts: int = 2) -> bool:
        for attempt in range(attempts):
            if self._move_once(joints):
                return True
            self.get_logger().warn(f"motion attempt {attempt + 1}/{attempts} failed")
        return False

    def _move_once(self, joints) -> bool:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = GROUP
        req.num_planning_attempts = 10
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3
        c = Constraints()
        for name, value in zip(ARM_JOINTS, joints):
            c.joint_constraints.append(JointConstraint(
                joint_name=name, position=float(value), tolerance_above=1e-3, tolerance_below=1e-3, weight=1.0))
        req.goal_constraints.append(c)
        goal.planning_options.plan_only = False
        send = self.move.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=10)
        handle = send.result()
        if handle is None or not handle.accepted:
            return False
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res, timeout_sec=120)
        code = res.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"move_group failed with error code {code}")
        return code == MoveItErrorCodes.SUCCESS

    def wait_settled(self, window_s=0.5, tol_rad=2e-4, timeout_s=30.0) -> float:
        # checks positions, PhysX reports small velocities even when the arm is still
        def settled():
            if len(self.joint_history) < 2:
                return False
            t_end = self.joint_history[-1][0]
            window = [q for t, q in self.joint_history if t >= t_end - window_s]
            if self.joint_history[0][0] > t_end - window_s or len(window) < 5:
                return False
            spread = max(max(q[j] for q in window) - min(q[j] for q in window) for j in ARM_JOINTS)
            return spread < tol_rad
        self.spin_until(settled, timeout_s, "arm to settle")
        return self.joint_history[-1][0]

    def grab_after(self, t_settled: float) -> Image:
        def fresh():
            img = self.last_image
            return img is not None and img.header.stamp.sec + img.header.stamp.nanosec * 1e-9 > t_settled
        self.spin_until(fresh, 10, "a post-settle image")
        return self.last_image

    def joints_at(self, stamp_s: float) -> dict:
        return min(self.joint_history, key=lambda tq: abs(tq[0] - stamp_s))[1]


def camera_info_yaml(info: CameraInfo) -> dict:
    return {
        "image_width": info.width,
        "image_height": info.height,
        "camera_name": "wrist_cam",
        "frame_id": info.header.frame_id,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in info.k]},
        "distortion_model": info.distortion_model,
        "distortion_coefficients": {"rows": 1, "cols": len(info.d), "data": [float(v) for v in info.d]},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [float(v) for v in info.r]},
        "projection_matrix": {"rows": 3, "cols": 4, "data": [float(v) for v in info.p]},
    }


def default_views(cube):
    # (radius, azimuth, elevation) around the cube. az=180 is the robot side.
    # Planning fails on some of these now and then, so there are spares.
    return [
        (0.35, 180, 60),
        (0.35, 140, 45),
        (0.35, 220, 45),
        (0.30, 180, 35),
        (0.30, 180, 85),
        (0.30, 120, 55),
        (0.30, 240, 55),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="captures", help="output directory")
    ap.add_argument("--cube-xyz", type=float, nargs=3, default=[0.5, 0.0, 0.0403])
    ap.add_argument("--cube-size", type=float, default=0.05)
    ap.add_argument("--table-z", type=float, default=0.0153,
                    help="table top height")
    ap.add_argument("--eye", type=float, nargs=3, help="camera position, single view")
    ap.add_argument("--look-at", type=float, nargs=3, help="point to look at, single view")
    ap.add_argument("--no-save", action="store_true", help="move only")
    ap.add_argument("--count", type=int, default=5, help="images to capture")
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = WristCapture()
    log = node.get_logger()
    try:
        node.wait_ready()
        node.add_scene_objects(args.cube_xyz, args.cube_size, args.table_z)
        T_tool_cam = node.lookup(TOOL, CAM)

        if args.eye:
            target = args.look_at or args.cube_xyz
            views = [(np.asarray(args.eye), np.asarray(target), "manual")]
        else:
            views = [(viewpoint(args.cube_xyz, r, az, el), np.asarray(args.cube_xyz), f"r={r} az={az} el={el}")
                     for r, az, el in default_views(args.cube_xyz)]

        out = os.path.abspath(args.out)
        if not args.no_save:
            os.makedirs(os.path.join(out, "images"), exist_ok=True)
            with open(os.path.join(out, "camera_info.yaml"), "w") as f:
                yaml.safe_dump(camera_info_yaml(node.camera_info), f, sort_keys=False)
            with open(os.path.join(out, "scene.yaml"), "w") as f:
                yaml.safe_dump({"frame": BASE, "cube_center": [float(v) for v in args.cube_xyz],
                                "cube_size": float(args.cube_size)}, f, sort_keys=False)
            poses_file = open(os.path.join(out, "poses.jsonl"), "w")

        saved = 0
        for i, (eye, target, label) in enumerate(views):
            if saved >= args.count:
                break
            T_base_cam_goal = look_at_optical(eye, target)
            T_base_tool_goal = T_base_cam_goal @ np.linalg.inv(T_tool_cam)
            log.info(f"[{i}] {label}: camera at {np.round(eye, 3).tolist()} looking at {np.round(target, 3).tolist()}")
            solutions = node.solve_ik(T_base_tool_goal)
            if not solutions:
                log.error(f"[{i}] no collision-free IK solution, skipping")
                continue
            img = None
            for joints in solutions[:MAX_IK_TRIES]:
                if not node.move_to_joints(joints):
                    log.error(f"[{i}] motion failed")
                    continue
                t_settled = node.wait_settled()
                frame = node.grab_after(t_settled)
                # some arm shapes put the forearm in front of the lens
                if args.eye or red_pixels(node.bridge.imgmsg_to_cv2(frame, "bgr8")) >= MIN_CUBE_PIXELS:
                    img = frame
                    break
                log.warn(f"[{i}] arm is blocking the cube, trying the next IK solution")
            if img is None:
                log.error(f"[{i}] no usable image, skipping")
                continue
            stamp = Time.from_msg(img.header.stamp)
            stamp_s = img.header.stamp.sec + img.header.stamp.nanosec * 1e-9
            T_base_cam = node.lookup(BASE, CAM, stamp)
            err_mm = 1000 * np.linalg.norm(T_base_cam[:3, 3] - T_base_cam_goal[:3, 3])
            q_now = node.joints_at(stamp_s)
            q_err = max(abs(q_now[j] - g) for j, g in zip(ARM_JOINTS, joints))
            log.info(f"[{i}] settled at t={t_settled:.3f}s, image t={stamp_s:.3f}s, "
                     f"max joint error vs IK goal {1000 * q_err:.1f} mrad, "
                     f"camera position error vs goal {err_mm:.1f} mm")
            if args.no_save:
                continue
            name = f"images/pose_{saved:02d}.png"
            cv2.imwrite(os.path.join(out, name), node.bridge.imgmsg_to_cv2(img, "bgr8"))
            q = node.joints_at(stamp_s)
            qx, qy, qz, qw = matrix_to_quat(T_base_cam[:3, :3])
            record = {
                "index": saved,
                "image": name,
                "stamp": {"sec": img.header.stamp.sec, "nanosec": img.header.stamp.nanosec},
                "joint_names": ARM_JOINTS,
                "joint_positions": [round(q[j], 6) for j in ARM_JOINTS],
                "camera_pose": {
                    "parent_frame": BASE,
                    "child_frame": CAM,
                    "translation": [round(v, 6) for v in T_base_cam[:3, 3]],
                    "quaternion_xyzw": [round(v, 6) for v in (qx, qy, qz, qw)],
                },
                "view": {"label": label, "eye": [round(v, 4) for v in eye], "look_at": [round(v, 4) for v in target]},
            }
            poses_file.write(json.dumps(record) + "\n")
            poses_file.flush()
            saved += 1
            log.info(f"[{i}] saved {name}")

        if not args.no_save:
            poses_file.close()
            log.info(f"Saved {saved} images to {out}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
