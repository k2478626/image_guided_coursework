#!/usr/bin/env python3

# ----------------------------
# my_robot_goal.py
# ----------------------------

import rclpy  # ROS 2 Python API
from moveit_py import MoveItPy  # MoveIt Python API

def main():
    # Initialize ROS 2 Python client
    rclpy.init()

    # Create a node to interface with ROS 2
    node = rclpy.create_node("my_robot_goal")

    # Initialize MoveItPy for your planning group
    # Change "arm" to your actual planning group name, e.g., "neuro_arm"
    moveit = MoveItPy(node=node, joint_model_group_name="arm")

    # Example: move to a random valid state
    moveit.move_to_random_state()
    moveit.execute()

    # Shutdown ROS 2
    rclpy.shutdown()

if __name__ == "__main__":
    main()
