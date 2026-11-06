#!/usr/bin/env python3
"""Print size, frame_id and sim-time rate of the /wrist_cam topics."""
import statistics

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image

N = 40


def main():
    rclpy.init()
    node = Node("check_camera_topics",
                parameter_overrides=[Parameter("use_sim_time", value=True)])
    imgs, infos, clock = [], [], []
    node.create_subscription(Image, "/wrist_cam/image_raw", imgs.append, 10)
    node.create_subscription(CameraInfo, "/wrist_cam/camera_info", infos.append, 10)
    node.create_subscription(Clock, "/clock", clock.append, 10)
    while len(imgs) < N or not infos:
        rclpy.spin_once(node, timeout_sec=1.0)

    img, info = imgs[-1], infos[-1]
    stamps = [m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in imgs]
    dts = [b - a for a, b in zip(stamps, stamps[1:])]
    sim_now = clock[-1].clock.sec + clock[-1].clock.nanosec * 1e-9
    print(f"image      {img.width}x{img.height} {img.encoding}  frame_id={img.header.frame_id}")
    print(f"info       {info.width}x{info.height}  frame_id={info.header.frame_id}  model={info.distortion_model}")
    print(f"K          fx={info.k[0]:.2f} fy={info.k[4]:.2f} cx={info.k[2]:.2f} cy={info.k[5]:.2f}")
    print(f"rate (sim) {1.0 / statistics.mean(dts):.2f} Hz   dt min/max {min(dts)*1e3:.1f}/{max(dts)*1e3:.1f} ms")
    print(f"stamp      last image {stamps[-1]:.3f} s vs /clock {sim_now:.3f} s (sim time, not wall)")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
