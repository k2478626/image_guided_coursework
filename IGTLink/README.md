# ROS2 OpenIGTLink Bridge

This package implements listener nodes for receiving trajectory data from 3D Slicer via OpenIGTLink.

## Features

- Receives `POSITION` and `TRANSFORM` messages from 3D Slicer.
- Parses incoming trajectory data for integration with ROS2 and MoveIt.
- Transforms data from Slicer’s RAS frame to ROS2’s ENU (base_link) frame for accurate spatial alignment.

## Usage

After launching 3D Slicer’s **OpenIGTLinkIF** connector:

1. Build the ROS2 workspace:

    ```bash
    colcon build --symlink-install
    source install/setup.bash
    ```

2. Run the server node or launch the .launch.py file:

    ```bash
    ros2 launch ros2_igtl_bridge bridge.launch.py
    ```

3. Confirm real-time updates:

    ```bash
    ros2 topic echo /IGTL_POINT_IN
    ```

You should see updates in real time when fiducial markers are moved within 3D Slicer, confirming accurate and reliable data transfer.

## Reference

For complete implementation details, see the report 
