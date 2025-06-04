from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_static_virtual_joint_tfs_launch


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("neuro_arm", package_name="my_neuro_arm_in_moveit")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )
    
    return generate_static_virtual_joint_tfs_launch(moveit_config)
