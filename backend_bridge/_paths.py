"""Make the pure-Python ROS package modules importable even when the workspace is not sourced."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for pkg in ('redshift_autonomy', 'redshift_sim'):
    p = str(ROOT / 'ros2_ws' / 'src' / pkg)
    if p not in sys.path:
        sys.path.append(p)
