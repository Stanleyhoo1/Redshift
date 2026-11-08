"""SDF models for the Ground Rover and the Aerial Scout, plus their geometry/extrinsics specs.

`ROVER_SPEC` / `SCOUT_SPEC` are the single source of truth for robot geometry: the launch file passes
them to autonomy nodes as ROS parameters, and `tau_max_nm` is written into the wheel joint effort
limits so Gazebo's physics stalls exactly where `physics_model` says the rover cannot climb.
"""
from __future__ import annotations

import math

ROVER_SPEC = {
    'mass_kg': 60.0, 'wheel_radius_m': 0.20, 'track_m': 0.85, 'wheelbase_m': 0.60, 'cg_height_m': 0.35,
    'wheels': 4, 'tau_max_nm': 20.8, 'base_height_m': 0.28, 'radius_m': 0.6,
    # longitudinal grip vs. lateral scrub (fdir1 = wheel axle): skid-steer turning stays within the torque limit
    'wheel_mu': 0.9, 'wheel_mu_lateral': 0.2,
    'lidar_xyz': [0.25, 0.0, 0.62], 'lidar_rings': 16, 'lidar_samples': 360,
    'lidar_vfov': [-0.2618, 0.1745], 'lidar_range': [0.3, 25.0],
    'camera_xyz': [0.45, 0.0, 0.50], 'camera_pitch_rad': 0.2094, 'camera_hfov': 1.396,
    'camera_w': 320, 'camera_h': 240, 'camera_hz': 10.0, 'camera_far': 40.0,
    'v_max': 1.0, 'w_max': 1.0,
}

SCOUT_SPEC = {
    'mass_kg': 2.0, 'camera_xyz': [0.0, 0.0, -0.08], 'camera_pitch_rad': math.pi / 2,
    'camera_hfov': 1.571, 'camera_w': 160, 'camera_h': 120, 'camera_hz': 5.0, 'camera_far': 40.0,
    'cruise_agl_m': 12.0, 'v_max': 4.0, 'vz_max': 2.0,
    'v_air_max': 18.0,      # air-relative limit: survey guidance <= v_max, plus wind feed-forward in a storm
}


ROVER_WHEEL_MASS_KG = 3.0
ROVER_AXLE_Z = -0.08           # wheel axles below base_link


def rover_com_z(spec: dict = ROVER_SPEC) -> float:
    """Height of the whole rover's centre of mass (chassis + wheels) above its base_link origin."""
    wheels_m = spec['wheels'] * ROVER_WHEEL_MASS_KG
    base_m = spec['mass_kg'] - wheels_m
    return (base_m * (spec['cg_height_m'] - spec['base_height_m']) + wheels_m * ROVER_AXLE_Z) / spec['mass_kg']


# Rover paint (docs/protocol.md §8.1 roles): rover_0 is the white scout rover with an orange nose stripe, every other
# rover is the amber "heavy recovery" rover with a dark stripe. Shared by the SDF and the mock renderer's bodies.
LIVERY = {
    'scout': {'chassis': (0.88, 0.88, 0.86), 'stripe': (1.0, 0.35, 0.1), 'stripe_emissive': (0.9, 0.25, 0.05)},
    'recovery': {'chassis': (0.95, 0.64, 0.10), 'stripe': (0.12, 0.12, 0.13), 'stripe_emissive': (0.0, 0.0, 0.0)},
}


def rover_livery(name: str) -> dict:
    return LIVERY['scout' if name == 'rover_0' else 'recovery']


def _box_inertia(m, x, y, z):
    return (m / 12 * (y * y + z * z), m / 12 * (x * x + z * z), m / 12 * (x * x + y * y))


def _inertial(m, ixx, iyy, izz, pose='0 0 0 0 0 0'):
    return (f'<inertial><pose>{pose}</pose><mass>{m:.4f}</mass><inertia><ixx>{ixx:.5f}</ixx><ixy>0</ixy>'
            f'<ixz>0</ixz><iyy>{iyy:.5f}</iyy><iyz>0</iyz><izz>{izz:.5f}</izz></inertia></inertial>')


def _mat(r, g, b, emissive=None, spec=0.2):
    e = f'<emissive>{emissive[0]} {emissive[1]} {emissive[2]} 1</emissive>' if emissive else ''
    return (f'<material><ambient>{r * 0.6:.3f} {g * 0.6:.3f} {b * 0.6:.3f} 1</ambient>'
            f'<diffuse>{r} {g} {b} 1</diffuse><specular>{spec} {spec} {spec} 1</specular>{e}</material>')


def _rgbd(name, topic, pose, hfov, w, h, hz, far):
    return f'''
      <sensor name="{name}" type="rgbd_camera">
        <pose>{pose}</pose>
        <always_on>1</always_on>
        <update_rate>{hz}</update_rate>
        <topic>{topic}</topic>
        <camera>
          <horizontal_fov>{hfov}</horizontal_fov>
          <image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>{far}</far></clip>
          <depth_camera><clip><near>0.1</near><far>{far}</far></clip></depth_camera>
        </camera>
      </sensor>'''


def _imu(topic, hz=100):
    return f'''
      <sensor name="imu" type="imu">
        <always_on>1</always_on>
        <update_rate>{hz}</update_rate>
        <topic>{topic}</topic>
      </sensor>'''


def _odom_pub(name, hz=30):
    return f'''
    <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
      <odom_frame>world</odom_frame>
      <robot_base_frame>{name}/base_link</robot_base_frame>
      <odom_topic>{name}/odom</odom_topic>
      <odom_publish_frequency>{hz}</odom_publish_frequency>
      <dimensions>3</dimensions>
    </plugin>'''


def rover_sdf(name: str, pose, spec: dict = ROVER_SPEC) -> str:
    x, y, z, yaw = pose
    s = spec
    lv = rover_livery(name)
    stripe_e = lv['stripe_emissive'] if any(lv['stripe_emissive']) else None
    r = s['wheel_radius_m']
    half_track = s['track_m'] / 2
    half_base = s['wheelbase_m'] / 2
    axle_z = ROVER_AXLE_Z
    wheel_m = ROVER_WHEEL_MASS_KG
    base_m = s['mass_kg'] - 4 * wheel_m
    bx, by, bz = 1.0, 0.62, 0.22
    ixx, iyy, izz = _box_inertia(base_m, bx, by, bz)
    lx, ly, lz = s['lidar_xyz']
    cx, cy, cz = s['camera_xyz']
    rmin, rmax = s['lidar_range']
    vmin, vmax = s['lidar_vfov']
    n_h = s['lidar_samples']
    wheel_iyy = 0.5 * wheel_m * r * r
    wheel_ixx = wheel_m * (3 * r * r + 0.14 ** 2) / 12
    wheels = []
    joints = []
    for wname, wx, wy in (('front_left', half_base, half_track), ('front_right', half_base, -half_track),
                          ('rear_left', -half_base, half_track), ('rear_right', -half_base, -half_track)):
        wheels.append(f'''
    <link name="{wname}_wheel">
      <pose relative_to="base_link">{wx} {wy} {axle_z} 0 0 0</pose>
      {_inertial(wheel_m, wheel_ixx, wheel_iyy, wheel_ixx)}
      <collision name="collision">
        <geometry><sphere><radius>{r}</radius></sphere></geometry>
        <surface><friction><ode><mu>{s['wheel_mu_lateral']}</mu><mu2>{s['wheel_mu']}</mu2><fdir1>0 1 0</fdir1></ode></friction></surface>
      </collision>
      <visual name="tire">
        <pose>0 0 0 1.5708 0 0</pose>
        <geometry><cylinder><radius>{r}</radius><length>0.14</length></cylinder></geometry>
        {_mat(0.08, 0.08, 0.09)}
      </visual>
      <visual name="hub">
        <pose>0 {0.072 if wy > 0 else -0.072} 0 1.5708 0 0</pose>
        <geometry><cylinder><radius>{r * 0.55}</radius><length>0.01</length></cylinder></geometry>
        {_mat(0.85, 0.45, 0.12, emissive=(0.35, 0.15, 0.02))}
      </visual>
    </link>''')
        joints.append(f'''
    <joint name="{wname}_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>{wname}_wheel</child>
      <axis><xyz>0 1 0</xyz><limit><effort>{s['tau_max_nm']}</effort></limit></axis>
    </joint>''')
    return f'''
  <model name="{name}">
    <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.4f}</pose>
    <link name="base_link">
      {_inertial(base_m, ixx, iyy, izz, pose=f'0 0 {s["cg_height_m"] - s["base_height_m"]:.3f} 0 0 0')}
      <collision name="chassis">
        <geometry><box><size>{bx} {by} {bz}</size></box></geometry>
      </collision>
      <visual name="chassis">
        <geometry><box><size>{bx} {by} {bz}</size></box></geometry>
        {_mat(*lv['chassis'], spec=0.5)}
      </visual>
      <visual name="deck">
        <pose>-0.05 0 0.12 0 0 0</pose>
        <geometry><box><size>0.82 0.58 0.02</size></box></geometry>
        {_mat(0.10, 0.12, 0.22, spec=0.8)}
      </visual>
      <visual name="stripe">
        <pose>0.505 0 0 0 0 0</pose>
        <geometry><box><size>0.01 0.5 0.06</size></box></geometry>
        {_mat(*lv['stripe'], emissive=stripe_e)}
      </visual>
      <visual name="mast">
        <pose>{lx} 0 {(lz + 0.11) / 2:.3f} 0 0 0</pose>
        <geometry><cylinder><radius>0.03</radius><length>{lz - 0.11:.3f}</length></cylinder></geometry>
        {_mat(0.3, 0.3, 0.32)}
      </visual>
      <visual name="lidar_head">
        <pose>{lx} {ly} {lz} 0 0 0</pose>
        <geometry><cylinder><radius>0.06</radius><length>0.07</length></cylinder></geometry>
        {_mat(0.1, 0.1, 0.1, emissive=(0.0, 0.3, 0.35))}
      </visual>
      <visual name="camera_head">
        <pose>{cx - 0.03:.3f} {cy} {cz} 0 {s['camera_pitch_rad']} 0</pose>
        <geometry><box><size>0.08 0.16 0.06</size></box></geometry>
        {_mat(0.15, 0.15, 0.16)}
      </visual>
      {_imu(f'{name}/imu')}
      <sensor name="lidar" type="gpu_lidar">
        <pose>{lx} {ly} {lz} 0 0 0</pose>
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <topic>{name}/lidar</topic>
        <lidar>
          <scan>
            <horizontal><samples>{n_h}</samples><resolution>1</resolution>
              <min_angle>{-math.pi:.5f}</min_angle><max_angle>{math.pi - 2 * math.pi / n_h:.5f}</max_angle></horizontal>
            <vertical><samples>{s['lidar_rings']}</samples><resolution>1</resolution>
              <min_angle>{vmin}</min_angle><max_angle>{vmax}</max_angle></vertical>
          </scan>
          <range><min>{rmin}</min><max>{rmax}</max><resolution>0.01</resolution></range>
          <noise><type>gaussian</type><mean>0</mean><stddev>0.01</stddev></noise>
        </lidar>
      </sensor>
      {_rgbd('camera', f'{name}/camera', f'{cx} {cy} {cz} 0 {s["camera_pitch_rad"]} 0', s['camera_hfov'],
             s['camera_w'], s['camera_h'], s['camera_hz'], s['camera_far'])}
    </link>
    {''.join(wheels)}
    {''.join(joints)}
    <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
      <left_joint>front_left_wheel_joint</left_joint>
      <left_joint>rear_left_wheel_joint</left_joint>
      <right_joint>front_right_wheel_joint</right_joint>
      <right_joint>rear_right_wheel_joint</right_joint>
      <wheel_separation>{s['track_m']}</wheel_separation>
      <wheel_radius>{r}</wheel_radius>
      <max_linear_velocity>1.2</max_linear_velocity>
      <min_linear_velocity>-1.2</min_linear_velocity>
      <max_angular_velocity>1.5</max_angular_velocity>
      <min_angular_velocity>-1.5</min_angular_velocity>
      <max_linear_acceleration>1.5</max_linear_acceleration>
      <min_linear_acceleration>-2.0</min_linear_acceleration>
      <topic>{name}/cmd_vel</topic>
      <odom_topic>{name}/wheel_odom</odom_topic>
      <frame_id>{name}/odom</frame_id>
      <child_frame_id>{name}/base_link</child_frame_id>
      <odom_publish_frequency>20</odom_publish_frequency>
    </plugin>
    {_odom_pub(name)}
  </model>'''


def scout_sdf(name: str, pose, spec: dict = SCOUT_SPEC) -> str:
    x, y, z, yaw = pose
    m = spec['mass_kg']
    ixx, iyy, izz = _box_inertia(m, 0.5, 0.5, 0.1)
    cx, cy, cz = spec['camera_xyz']
    arms = []
    for k, (ax, ay) in enumerate(((0.25, 0.25), (0.25, -0.25), (-0.25, 0.25), (-0.25, -0.25))):
        arms.append(f'''
      <visual name="arm_{k}">
        <pose>{ax / 2} {ay / 2} 0 0 0 {math.atan2(ay, ax):.4f}</pose>
        <geometry><box><size>0.36 0.03 0.02</size></box></geometry>
        {_mat(0.2, 0.2, 0.22)}
      </visual>
      <visual name="rotor_{k}">
        <pose>{ax} {ay} 0.03 0 0 0</pose>
        <geometry><cylinder><radius>0.11</radius><length>0.005</length></cylinder></geometry>
        {_mat(0.3, 0.9, 1.0, emissive=(0.0, 0.5, 0.6))}
      </visual>''')
    return f'''
  <model name="{name}">
    <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.4f}</pose>
    <link name="base_link">
      {_inertial(m, ixx, iyy, izz)}
      <collision name="body">
        <geometry><box><size>0.5 0.5 0.1</size></box></geometry>
      </collision>
      <visual name="body">
        <geometry><box><size>0.22 0.22 0.08</size></box></geometry>
        {_mat(0.9, 0.9, 0.92, spec=0.6)}
      </visual>
      <visual name="beacon">
        <pose>0 0 0.05 0 0 0</pose>
        <geometry><sphere><radius>0.03</radius></sphere></geometry>
        {_mat(1.0, 0.3, 0.1, emissive=(1.0, 0.3, 0.05))}
      </visual>
      {''.join(arms)}
      {_imu(f'{name}/imu')}
      {_rgbd('camera', f'{name}/camera', f'{cx} {cy} {cz} 0 {spec["camera_pitch_rad"]:.5f} 0', spec['camera_hfov'],
             spec['camera_w'], spec['camera_h'], spec['camera_hz'], spec['camera_far'])}
    </link>
    <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
      <topic>{name}/cmd_vel</topic>
    </plugin>
    {_odom_pub(name)}
  </model>'''
