#!/usr/bin/env python3 
# ----------------------------
# my_neuro_arm_goal_node.py
# ----------------------------

import rclpy  # ROS 2 Python API
from rclpy.node import Node
from moveit_py import MoveItPy  # MoveIt Python API
from geometry_msgs.msg import PoseStamped

from ros2_igtl_bridge.msg import PointArray

class MyNeuroArmGoal(Node):
    def __init__(self):
        super().__init__('my_neuro_arm_goal_node')
        self.moveit = MoveItPy(node=self, joint_model_group_name="arm_group")

        self.entry_point = None
        self.target_point = None 

        self.create_subscription(
            PointArray, "/IGTL_POINT_IN", self.point_callback, 10
        )

    def point_callback(self, msg):
        name = msg.name
        point = msg.pointdata[0]
        self.get_logger().info(f"Received point: {name} at {point.x}, {point.y}, {point.z}")
    
        if name == "Entry":
            self.entry_point = point
        elif name == "Target":
            self.target_point = point
    
        #Once both are received, move from entry to target (optimal trajectory)
        if self.entry_point and self.target_point:
            self.move_to_point(self.entry_point)
            self.move_to_point(self.target_point)
            #Reset points to avoid re-execution
            self.entry_point = None
            self.target_point = None
    
    def move_to_point(self, point):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        pose.pose.position.z = point.z
        pose.pose.orientation.w = 1.0 #Neutral orientation
    
        self.get_logger().info(f"Moving to point: {pose.pose.position}")
        self.moveit.move_to_pose(pose)
        self.moveit.execute()

def main(args=None):
    rclpy.init(args=args)
    node = MyNeuroArmGoal()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    
