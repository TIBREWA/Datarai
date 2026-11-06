#!/usr/bin/env python3
"""Project the cube into the saved images and draw it. Reads only the saved files.

The red centroid is the centre of the visible faces, not the real cube centre,
so expect a few px of difference.

    python3 verify_projection.py captures
"""
import json
import os
import sys

import cv2
import numpy as np
import yaml


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project(k, t_cam_base, pts_base):
    pts = np.c_[pts_base, np.ones(len(pts_base))] @ t_cam_base.T
    xyz = pts[:, :3]
    uvw = xyz @ k.T
    return uvw[:, :2] / uvw[:, 2:3], xyz[:, 2]


def red_centroid(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 40), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 40), (180, 255, 255))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None, 0
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cents[i], int(stats[i, cv2.CC_STAT_AREA])


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "captures"
    info = yaml.safe_load(open(os.path.join(root, "camera_info.yaml")))
    scene = yaml.safe_load(open(os.path.join(root, "scene.yaml")))
    k = np.array(info["camera_matrix"]["data"]).reshape(3, 3)
    c = np.array(scene["cube_center"])
    h = scene["cube_size"] / 2
    corners = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)]) + c
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]

    os.makedirs(os.path.join(root, "annotated"), exist_ok=True)
    rows = []
    for line in open(os.path.join(root, "poses.jsonl")):
        rec = json.loads(line)
        pose = rec["camera_pose"]
        t_base_cam = np.eye(4)
        t_base_cam[:3, :3] = quat_to_matrix(*pose["quaternion_xyzw"])
        t_base_cam[:3, 3] = pose["translation"]
        t_cam_base = np.linalg.inv(t_base_cam)

        img = cv2.imread(os.path.join(root, rec["image"]))
        found, area = red_centroid(img)  # do this before drawing on the image
        (uv,), (depth,) = project(k, t_cam_base, c[None])
        corner_uv, _ = project(k, t_cam_base, corners)
        for a, b in edges:
            cv2.line(img, tuple(np.round(corner_uv[a]).astype(int)), tuple(np.round(corner_uv[b]).astype(int)),
                     (0, 255, 0), 1, cv2.LINE_AA)
        p = tuple(np.round(uv).astype(int))
        cv2.drawMarker(img, p, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.circle(img, p, 4, (0, 255, 0), -1)

        err = None
        if found is not None:
            err = float(np.linalg.norm(found - uv))
            cv2.circle(img, tuple(np.round(found).astype(int)), 3, (255, 0, 255), -1)
        label = f"pose {rec['index']}: proj ({uv[0]:.1f},{uv[1]:.1f})  depth {depth:.3f} m"
        label2 = f"red centroid {'n/a' if err is None else f'({found[0]:.1f},{found[1]:.1f})  diff {err:.1f} px'}"
        for j, txt in enumerate((label, label2)):
            cv2.putText(img, txt, (8, 20 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (8, 20 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out = os.path.join(root, "annotated", os.path.basename(rec["image"]))
        cv2.imwrite(out, img)
        inside = 0 <= uv[0] < info["image_width"] and 0 <= uv[1] < info["image_height"]
        rows.append((rec["index"], uv, depth, inside, area, err))

    print(f"{'pose':>4} {'proj u':>8} {'proj v':>8} {'depth m':>8} {'in img':>6} {'red px':>7} {'diff px':>8}")
    for i, uv, d, inside, area, err in rows:
        print(f"{i:>4} {uv[0]:>8.1f} {uv[1]:>8.1f} {d:>8.3f} {str(inside):>6} {area:>7} "
              f"{'n/a' if err is None else f'{err:.1f}':>8}")
    errs = [r[5] for r in rows if r[5] is not None]
    if errs:
        print(f"projected centre vs red centroid: mean {np.mean(errs):.1f} px, max {np.max(errs):.1f} px")
    print(f"annotated images in {os.path.join(root, 'annotated')}")


if __name__ == "__main__":
    main()
