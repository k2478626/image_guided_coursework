# Image-Guided Surgical Planning and Robotic Simulation

This repository contains the code, configuration files, and test scripts for an integrated pipeline that combines medical image processing with robotic simulation. The workflow consists of:

✅ **3D Slicer PathPlanning Module** – An interactive module for safe trajectory planning within medical images.  
✅ **OpenIGTLink Communication Scripts** – Real-time data streaming between 3D Slicer and ROS2.  
✅ **ROS2 Integration** – URDF description, MoveIt planning configuration, and RViz visualization for robotic control.

Each component is carefully modularized, validated, and designed to be reproducible.  

## Repository Structure

- **/Slicer/PathPlanning** – 3D Slicer module code, UI file, and unit tests.  
- **/Ros2/my_neuro_arm_in_moveit** – URDF, MoveIt configuration package, and launch files.  
- **/Ros2/igtl_bridge** – ROS2 listener nodes for OpenIGTLink data reception and transformation.  
- **/Validation** – Supporting scripts and logs demonstrating module testing.

## Setup and Usage

Please see the individual README.md files within each folder for detailed instructions on building and running each component.

## Reference

This repository accompanies the master’s project report:  
**“Integration of Medical Image Processing and Robotic Simulation for Surgical Planning”**  
Author: Andrea Walker  
King’s College London, 2025
