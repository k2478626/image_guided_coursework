from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    #spawning joint state broadcaster that publishes joint states
    jsb_broadcaster_spawner = Node(
         package="controller_manager",
         executable="spawner",
         name="spawner_joint_state_broadcaster",
         arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
         output="screen",
    )
    
    #spawning the arm_group_controler that moveit will send the trajectories to
    arm_group_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="spawner_arm_group_controller",
        arguments=["arm_group_controller", "--controller-manager", "/controller_manager"],
    )
    
    #Launch spawners
    return LaunchDescription([
        jsb_broadcaster_spawner,
        arm_group_spawner,
    ])
