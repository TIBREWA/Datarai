#!/usr/bin/env python3
"""Draw TF frame origins on a camera frame. If the mount is right, dots land on the links.

    python3 snap_tf_overlay.py out.png [frame ...]
"""
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

CAM = "camera_optical_frame"


def tf_to_matrix(t):
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
    return m


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tf_overlay.png"
    frames = sys.argv[2:] or ["base_link", "shoulder_link", "upper_arm_link", "forearm_link",
                              "wrist_1_link", "wrist_2_link"]
    rclpy.init()
    node = Node("snap_tf_overlay", parameter_overrides=[Parameter("use_sim_time", value=True)])
    buf = Buffer()
    TransformListener(buf, node)
    imgs, infos = [], []
    node.create_subscription(Image, "/wrist_cam/image_raw", imgs.append, 2)
    node.create_subscription(CameraInfo, "/wrist_cam/camera_info", infos.append, 2)
    while len(imgs) < 3 or not infos:
        rclpy.spin_once(node, timeout_sec=0.5)
    img_msg, info = imgs[-1], infos[-1]
    stamp = Time.from_msg(img_msg.header.stamp)
    k = np.array(info.k).reshape(3, 3)
    img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")

    for f in frames:
        while not buf.can_transform(CAM, f, stamp, Duration(seconds=0.1)):
            rclpy.spin_once(node, timeout_sec=0.1)
        p = tf_to_matrix(buf.lookup_transform(CAM, f, stamp))[:3, 3]
        if p[2] <= 0:
            print(f"{f:16s} behind camera (z={p[2]:+.3f})")
            continue
        u, v, _ = k @ (p / p[2])
        print(f"{f:16s} cam xyz [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}] -> pixel ({u:.1f}, {v:.1f})")
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (int(round(u)), int(round(v))), 6, (0, 0, 255), 2)
            cv2.putText(img, f, (int(u) + 8, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(out, img)
    print("saved", out)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
