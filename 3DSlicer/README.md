# PathPlanning Module for 3D Slicer

This module allows users to plan safe surgical trajectories within medical images. It features:

- Selection of entry and target points using fiducial markers.
- Definition of target regions, avoid regions, and cortex structures.
- Constraint-based trajectory optimization (length, approach angle, obstacle clearance).
- Visualization of planned trajectories in 3D and slice views.
- Comprehensive unit tests for robust validation.

## Setup

1. Copy the `PathPlanning` module folder into your 3D Slicer extensions directory.  
2. Start 3D Slicer and navigate to the **Modules** drop-down to find **PathPlanning**.  
3. Load your medical image data and segmented labelmaps.  
4. Use the module’s GUI to plan and visualize trajectories.

## Unit Testing

From the 3D Slicer Python console:

```python
import PathPlanning
PathPlanning.PathPlanningTest().runTest()
