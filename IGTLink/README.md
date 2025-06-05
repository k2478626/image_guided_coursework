# ROS2 OpenIGTLink Bridge

This package implements listener nodes for receiving trajectory data from 3D Slicer via OpenIGTLink.

---

## Features

- Receives `POSITION` and `TRANSFORM` messages from 3D Slicer.
- Parses incoming trajectory data for integration with ROS2 and MoveIt2.
- Transforms data from Slicer’s RAS frame to ROS2’s ENU (base_link) frame for accurate spatial alignment.

---

## 3D Slicer ↔ ROS2 Connection Setup

Here’s how to connect 3D Slicer to your ROS2-IGTLink-Bridge in real time.

---

### 1. Build and Source the ROS2 Workspace

colcon build --symlink-install
source install/setup.bash

---

### 2. Launch the ROS2-IGTLink-Bridge in Server Mode

ros2 launch ros2_igtl_bridge bridge.launch.py

The bridge will listen on port 18944 for incoming OpenIGTLink messages.

---

### 3. Configure 3D Slicer’s OpenIGTLinkIF

In 3D Slicer:

1. Open the OpenIGTLinkIF module.
2. Create a new Connector:
   - Type: Client  
   - Hostname: IP address of your ROS2 machine (e.g., 192.168.1.117).  
   - Port: 18944  
3. Click Active to start the connection.

You should see the connector status turn green (indicating an active connection).

---

### 4 Register and Send Trajectory Data

1. Go to the Markups module in 3D Slicer.
2. Select or create a Markups Fiducial Node (e.g., TrajectoryPoints) containing your entry and target points.
3. In the OpenIGTLinkIF module, add this node to the Outgoing Nodes list.

⚠️ If the + button in the GUI isn’t working, do it programmatically:

connectorNode = slicer.util.getNode('IGTLConnector')
fiducialNode = slicer.util.getNode('TrajectoryPoints')
connectorNode.RegisterOutgoingMRMLNode(fiducialNode)
connectorNode.Start()

Move points in the 3D view to trigger updates, or manually push updates:

connectorNode.PushNode(fiducialNode)

---

### 5 Verify Real-Time Updates in ROS2

In your ROS2 terminal:

ros2 topic echo /IGTL_POINT_IN

As you move or edit points in 3D Slicer, you should see real-time updates streamed into ROS2!

---

## Important Notes

- Coordinate Frames:  
  3D Slicer uses RAS (Right-Anterior-Superior).  
  ROS2 typically uses ENU (East-North-Up) or LPS.  
  Make sure you account for any axis flips when integrating with your robot (e.g., flipping Y and Z).

- Topic for Trajectories:  
  The entry and target points from 3D Slicer are published on /IGTL_POINT_IN as a list of 3D points.

- Name Field:  
  The name field in the ROS2 messages is set by the Markups Fiducial Node name in Slicer.  
  Rename it in Slicer if needed for clarity (e.g., EntryAndTarget).

---

## Real-Time Validation in Slicer

To validate real-time point updates directly in Slicer:

1. In the Markups module, watch the R, A, S coordinates table.
2. Or use this Python script in the Slicer Python Interactor:

def onMarkupPointModified(caller, event):
    numPoints = fiducialNode.GetNumberOfFiducials()
    for i in range(numPoints):
        pos = [0, 0, 0]
        fiducialNode.GetNthFiducialPosition(i, pos)
        print(f"Point {i}: X={pos[0]}, Y={pos[1]}, Z={pos[2]}")

fiducialNode = slicer.util.getNode('TrajectoryPoints')
fiducialNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, onMarkupPointModified)

This prints live XYZ updates in the Slicer console as you move points!

---

## Reference

For detailed implementation details, see the accompanying report or documentation.  
This pipeline can be extended to move the robot in real-time using the /IGTL_POINT_IN data.
