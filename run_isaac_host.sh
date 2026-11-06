#!/usr/bin/env bash
# Runs Isaac on the host when ROS is in a container (Fedora + toolbox here).
# Then in the container:
#   ros2 launch ur5e_isaac_bridge ur5e_isaac.launch.py launch_isaac:=false
set -euo pipefail

ROS_CONTAINER="${ROS_CONTAINER:-ros2}"
ISAACSIM="${ISAACSIM:-$HOME/isaacsim}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$REPO/ros2_ws"
PKG="$WS/src/ur5e_isaac_bridge"

if [[ ! -f "$WS/install/setup.bash" ]]; then
  echo "Build the workspace first (see README)." >&2
  exit 1
fi

URDF="$(mktemp -t ur5e_isaac_XXXX.urdf)"
toolbox run -c "$ROS_CONTAINER" bash -c "
  source /opt/ros/jazzy/setup.bash && source '$WS/install/setup.bash' &&
  xacro '$PKG/urdf/ur_isaac.urdf.xacro' ur_type:=ur5e name:=ur force_abs_paths:=true
" > "$URDF"
echo "[run_isaac_host] URDF expanded to $URDF"

# The bridge only picks a distro on Ubuntu, otherwise it falls back to humble and fails
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ISAACSIM/exts/isaacsim.ros2.core/jazzy/lib"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export PYTHONUNBUFFERED=1

exec "$ISAACSIM/python.sh" "$PKG/scripts/spawn_ur5e.py" --urdf "$URDF" "$@"
