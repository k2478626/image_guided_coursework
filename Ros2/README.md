# ROS2 Neuro Arm Description and MoveIt Configuration

This folder contains the URDF description and MoveIt configuration package for the 7-DOF neuro arm manipulator used in the surgical planning pipeline.

## Contents

- **URDF**: Neuro arm geometry, joint limits, and link definitions (`neuro_arm.urdf`).  
- **MoveIt Config**: Planning pipelines, joint limits, controllers, and visualization settings for MoveIt and RViz2.  
- **Launch Files**: Launch scripts for MoveIt planning and simulation.

## Usage

1. Build the ROS2 workspace:

```bash
colcon build --symlink-install
source install/setup.bash

2. Launch MoveIt and Rviz:
```bash
ros2 launch my_neuro_arm_in_moveit demo.launch.py

This will bring up MoveIt's planning environment in RViz2
