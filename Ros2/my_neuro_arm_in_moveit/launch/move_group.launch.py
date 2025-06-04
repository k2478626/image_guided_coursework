from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch

#This function generates a trajectory for the arm_controller 
def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("neuro_arm", package_name="my_neuro_arm_in_moveit")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )
    
    return generate_move_group_launch(moveit_config)
    
    

