import logging
import os
from typing import Annotated, Optional, List

import vtk
import time
import slicer
import numpy as np
import unittest
import qt
import SimpleITK as sitk
import sitkUtils as su
from slicer.ScriptedLoadableModule import *
from slicer.i18n import tr as _
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper, WithinRange
from slicer.util import VTKObservationMixin
from scipy.spatial import cKDTree
from slicer import vtkMRMLMarkupsFiducialNode, vtkMRMLLabelMapVolumeNode


#
# PathPlanning
#

class PathPlanning(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("PathPlanning")
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = ["Andrea Walker"]
        self.parent.helpText = _("""Plan safe trajectories avoiding anatomical structures""")
        self.parent.acknowledgementText = _("""Developed using 3D Slicer and parameterNodeWrapper.""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # PathPlanning1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="PathPlanning",
        sampleName="PathPlanning1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "PathPlanning1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="PathPlanning1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="PathPlanning1",
    )

    # PathPlanning2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="PathPlanning",
        sampleName="PathPlanning2",
        thumbnailFileName=os.path.join(iconsPath, "PathPlanning2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="PathPlanning2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="PathPlanning2",
    )


#
# PathPlanningParameterNode
#


@parameterNodeWrapper
class PathPlanningParameterNode:
    """
    The parameters needed by module.

    entryFiducials -
    targetFiducials -
    targetRegion -
    cortex -
    avoidRegions -
    """

    entryFiducials: vtkMRMLMarkupsFiducialNode
    targetFiducials: vtkMRMLMarkupsFiducialNode
    targetRegion: vtkMRMLLabelMapVolumeNode
    cortex: Optional[vtkMRMLLabelMapVolumeNode] = None
    avoidRegions: List[vtkMRMLLabelMapVolumeNode]
    maxTrajectoryLengthMm: Annotated[float, WithinRange(0.0, 200.0)]

#
# PathPlanningWidget
#


class PathPlanningWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/PathPlanning.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = PathPlanningLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.avoidSelectors = []
        self.addAvoidSelector()
        self.ui.addAvoidButton.connect("clicked(bool)", self.addAvoidSelector)
        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def addAvoidSelector(self):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        selector.setMRMLScene(slicer.mrmlScene)
        selector.noneEnabled = True
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.setToolTip("Select a region to avoid")

        self.ui.avoidRegionContainer.layout().addWidget(selector)
        self.avoidSelectors.append(selector)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, parameterNode):
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = parameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

    def _checkCanApply(self, caller=None, event=None) -> None:
        if (self._parameterNode.entryFiducials and
                self._parameterNode.targetFiducials and
                self._parameterNode.targetRegion):
            self.ui.applyButton.toolTip = _("Compute best trajectory")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select all required input nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        pn = self._parameterNode
        avoidNodes = [sel.currentNode() for sel in self.avoidSelectors if sel.currentNode()]
        best_entry, best_target, best_dist = self.logic.find_best_trajectory(
            pn.entryFiducials, pn.targetFiducials,
            {
                "target_in": pn.targetRegion,
                "avoid": avoidNodes,
                "cortex": pn.cortex,
                "max_length": pn.maxTrajectoryLengthMm
            }
        )
        if best_entry and best_target:
            self.logic.visualize_trajectory(best_entry, best_target)
            print("Best trajectory:")
            print("Entry:", best_entry)
            print("Target:", best_target)
            print("Distance to nearest obstacle:", best_dist)
        else:
            slicer.util.errorDisplay("No valid trajectory found.")

#
# PathPlanningLogic
#


class PathPlanningLogic(ScriptedLoadableModuleLogic):

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return PathPlanningParameterNode(super().getParameterNode())

    def ras_to_ijk(self, volumeNode, ras):
        mat = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(mat)
        ras = list(ras) + [1]
        ijk = [0, 0, 0, 0]
        mat.MultiplyPoint(ras, ijk)
        return tuple(int(round(c)) for c in ijk[:3])

    def check_point_in_label(self, volumeNode, ras):
        arr = slicer.util.arrayFromVolume(volumeNode)
        ijk = self.ras_to_ijk(volumeNode, ras)
        try:
            return arr[ijk[2], ijk[1], ijk[0]] > 0
        except:
            return False

    def sample_line(self, entry, target, num=100):
        return [tuple(entry[i] + (target[i] - entry[i]) * t for i in range(3)) for t in np.linspace(0, 1, num)]

    def convert_labelmap_to_model(self, labelmapNode):
        seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, seg)
        seg.CreateDefaultDisplayNodes()
        seg.GetSegmentation().CreateRepresentation("Closed surface")

        # Get a valid subject hierarchy parent
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        sceneItemID = shNode.GetSceneItemID()

        # Create a folder to contain the exported model
        folderID = shNode.CreateFolderItem(sceneItemID, "TempModelExport")

        # ✅ Pass the required second argument
        slicer.modules.segmentations.logic().ExportAllSegmentsToModels(seg, folderID)

        model = slicer.util.getNodesByClass("vtkMRMLModelNode")[-1]
        slicer.mrmlScene.RemoveNode(seg)
        return model

    def get_kdtree_and_normals(self, modelNode):
        poly = modelNode.GetMesh()

        # ✅ Compute normals if missing
        if poly.GetPointData().GetNormals() is None:
            normalsFilter = vtk.vtkPolyDataNormals()
            normalsFilter.SetInputData(poly)
            normalsFilter.ConsistencyOn()
            normalsFilter.AutoOrientNormalsOn()
            normalsFilter.SplittingOff()
            normalsFilter.Update()
            poly = normalsFilter.GetOutput()
            modelNode.SetAndObserveMesh(poly)

        points = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())])
        normals = np.array([poly.GetPointData().GetNormals().GetTuple(i) for i in range(poly.GetNumberOfPoints())])

        return cKDTree(points), normals

    def check_angle(self, entry, target, kd, norms):
        vec = np.array(target) - np.array(entry)
        vec /= np.linalg.norm(vec)
        _, idx = kd.query(entry)
        normal = norms[idx] / np.linalg.norm(norms[idx])
        angle = np.degrees(np.arccos(np.clip(np.dot(vec, normal), -1.0, 1.0)))
        return abs(90 - angle)

    def compute_distance_map(self, labelmapNode):
        img = su.PullVolumeFromSlicer(labelmapNode)
        filt = sitk.DanielssonDistanceMapImageFilter()
        dist = filt.Execute(img)
        return su.PushVolumeToSlicer(dist, None, "DistanceMap")

    def measure_path_distance(self, distNode, entry, target, num=100):
        arr = slicer.util.arrayFromVolume(distNode)
        samples = self.sample_line(entry, target, num)
        distances = []
        for pt in samples:
            ijk = self.ras_to_ijk(distNode, pt)
            try:
                d = float(arr[ijk[2], ijk[1], ijk[0]])
            except:
                d = 0.0
            distances.append(d)
        return distances, min(distances)

    def combine_avoid_regions(self, avoid_nodes):
        """
        Given a list of vtkMRMLLabelMapVolumeNode (avoid_nodes),
        pull each into SimpleITK, perform a logical OR across their binary arrays,
        then push the combined binary image back into Slicer as a new LabelMap node.
        Returns that new “combined avoid” node.
        """
        if len(avoid_nodes) == 1:
            return avoid_nodes[0]

        # Pull first avoid region into SimpleITK
        base_sitk = su.PullVolumeFromSlicer(avoid_nodes[0])
        base_array = sitk.GetArrayFromImage(base_sitk)  # Z,Y,X

        combined_mask = (base_array > 0)

        for node in avoid_nodes[1:]:
            sitk_img = su.PullVolumeFromSlicer(node)
            arr = sitk.GetArrayFromImage(sitk_img)
            combined_mask = np.logical_or(combined_mask, arr > 0)

        combined_uint8 = combined_mask.astype(np.uint8)
        combined_sitk = sitk.GetImageFromArray(combined_uint8)
        combined_sitk.CopyInformation(base_sitk)

        combined_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "CombinedAvoid"
        )
        su.PushVolumeToSlicer(combined_sitk, combined_node)
        return combined_node

    def check_trajectory(self, entry, target, constraints, kd, norms):
        #1) ensure target lies within targetRegion
        if constraints["target_in"] and not self.check_point_in_label(constraints["target_in"], target):
            return False, "Target outside region"
        #2) ensure no sample point intersects any avoid region
        for avoid in constraints["avoid"]:
            for pt in self.sample_line(entry, target):
                if self.check_point_in_label(avoid, pt):
                    return False, "Hit avoid region"
        #3) angle vs cortex normal check
        if kd and norms is not None:
            angle = self.check_angle(entry, target, kd, norms)
            if angle < 55:
                return False, f"Angle too oblique: {angle:.1f}°"
        #4) length constraint
        if "max_length" in constraints:
            length = np.linalg.norm(np.array(target) - np.array(entry))
            if length > constraints["max_length"]:
                return False, f"Trajectory too long ({length:.1f}mm > {constraints['max_length']:.1f}mm)"
        return True, "Valid"

    def find_best_trajectory(self, entryNode, targetNode, constraints, num=100):
        total_start = time.time()

        entries = self.get_fiducial_points(entryNode)
        targets = self.get_fiducial_points(targetNode)

        kd, norms = None, None
        if constraints["cortex"]:
            t0 = time.time()
            model = self.convert_labelmap_to_model(constraints["cortex"])
            kd, norms = self.get_kdtree_and_normals(model)
            print(f"⏱️ Cortex mesh + KDTree: {time.time() - t0:.2f}s")

        distNode = None
        if constraints["avoid"]:
            t0 = time.time()
            combined_avoid = self.combine_avoid_regions(constraints["avoid"])
            distNode = self.compute_distance_map(combined_avoid)
            print(f"⏱️ Distance map creation: {time.time() - t0:.2f}s")

        best = (None, None, -np.inf)
        eval_start = time.time()

        for e in entries:
            for t in targets:
                valid, reason = self.check_trajectory(e, t, constraints, kd, norms)
                if not valid:
                    continue
                _, d = self.measure_path_distance(distNode, e, t, num) if distNode else ([], 0)
                if d > best[2]:
                    best = (e, t, d)

        print(f"⏱️ Trajectory evaluation loop: {time.time() - eval_start:.2f}s")
        print(f"✅ Total planning time: {time.time() - total_start:.2f}s")

        return best

    def visualize_trajectory(self, entry, target):
        t0 = time.time()
        self.create_line(entry, target)
        self.create_fiducials(entry, target)
        print(f"🖼️ Visualization created in {time.time() - t0:.2f}s")

    def get_fiducial_points(self, node):
        return [list(node.GetNthControlPointPosition(i)) for i in range(node.GetNumberOfControlPoints())]

    def create_line(self, entry, target, name="TrajectoryLine"):
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(entry)
        pts.InsertNextPoint(target)
        line = vtk.vtkCellArray()
        line.InsertNextCell(2)
        line.InsertCellPoint(0)
        line.InsertCellPoint(1)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(line)
        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        model.SetAndObservePolyData(poly)
        disp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        model.SetAndObserveDisplayNodeID(disp.GetID())
        disp.SetColor(1, 0, 0)
        disp.SetLineWidth(3)

    def create_fiducials(self, entry, target, name="TrajectoryPoints"):
        fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        fid.AddControlPoint(*entry)
        fid.SetNthControlPointLabel(0, "Entry")
        fid.AddControlPoint(*target)
        fid.SetNthControlPointLabel(1, "Target")
        fid.GetDisplayNode().SetSelectedColor(0, 1, 0)
        fid.GetDisplayNode().SetTextScale(2.0)

#
# PathPlanningTest
#


class PathPlanningTest(ScriptedLoadableModuleTest):
    """
    Test suite for the PathPlanning module.
    Covers core logic, edge cases, and end-to-end GUI execution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from PathPlanning import PathPlanningLogic
        self.logic = PathPlanningLogic()

    def setUp(self):
        """
        Called before each test: clear the scene and re‐instantiate logic.
        """
        slicer.mrmlScene.Clear()
        from PathPlanning import PathPlanningLogic
        self.logic = PathPlanningLogic()

    def runTest(self):
        """
        Run all tests in sequence. For manual invocation of individual tests,
        call setUp() first, then the desired test_...() method.
        """
        self.setUp()
        self.test_ras_to_ijk_roundtrip()
        self.test_trajectory_validity()
        self.test_distance_map_creation()
        self.test_multiple_avoid_regions()
        self.test_max_trajectory_length_enforced()
        self.test_target_in_region_constraint()
        self.test_angle_constraint()
        self.test_with_real_geometry()

    def _create_labelmap_from_array(self, array, name="TempLabelmap"):
        """
        Helper: take a 3D NumPy array (dtype=uint8) and push it into a newly created
        vtkMRMLLabelMapVolumeNode in the scene. Returns that LabelMapVolumeNode.
        """
        # Convert NumPy → SimpleITK image
        sitk_image = sitk.GetImageFromArray(array.astype(np.uint8))
        sitk_image.SetSpacing((1.0, 1.0, 1.0))
        sitk_image.SetOrigin((0.0, 0.0, 0.0))

        # Create an empty LabelMapVolumeNode in the scene, with the given name
        labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name)

        # Push the SimpleITK image *into* that labelNode
        su.PushVolumeToSlicer(sitk_image, labelNode)

        return labelNode

    def test_ras_to_ijk_roundtrip(self):
        """
        Instead of assuming an identity transform, verify that a known IJK coordinate,
        when converted → RAS via the volume’s IJKToRASMatrix and back → IJK via ras_to_ijk,
        yields the original (i, j, k).
        """
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        volNode = self._create_labelmap_from_array(arr, name="RoundtripVolume")

        # Pick a valid voxel coordinate within [0..9]
        i, j, k = 2, 3, 4

        # Get the volume’s IJK→RAS 4×4 matrix
        ijkToRAS = vtk.vtkMatrix4x4()
        volNode.GetIJKToRASMatrix(ijkToRAS)

        # Compute RAS = M * [i, j, k, 1]^T
        ras = [0.0, 0.0, 0.0]
        for row in range(3):
            ras[row] = (
                    ijkToRAS.GetElement(row, 0) * i +
                    ijkToRAS.GetElement(row, 1) * j +
                    ijkToRAS.GetElement(row, 2) * k +
                    ijkToRAS.GetElement(row, 3)
            )

        # Now run ras_to_ijk; we expect (i, j, k) exactly
        ijk_back = self.logic.ras_to_ijk(volNode, tuple(ras))
        self.assertEqual(ijk_back, (i, j, k))
        print("[PASS] test_ras_to_ijk_roundtrip")

    def test_trajectory_validity(self):
        """
        Test a simple, straight trajectory with no constraints (should be valid).
        """
        entry = (0.0, 0.0, 0.0)
        target = (0.0, 0.0, 10.0)
        constraints = {
            "target_in": None,
            "avoid": [],
            "cortex": None,
            "max_length": 20.0
        }
        valid, reason = self.logic.check_trajectory(entry, target, constraints, None, None)
        self.assertTrue(valid)
        self.assertEqual(reason, "Valid")
        print("[PASS] test_trajectory_validity")

    def test_distance_map_creation(self):
        """
        Create a simple labelmap (all zeros except a single voxel at center),
        then compute its distance map. Verify the output exists and the center’s
        distance is 0 (because that voxel was “1” in the labelmap).
        """
        arr = np.zeros((20, 20, 20), dtype=np.uint8)
        arr[10, 10, 10] = 1
        labelmapNode = self._create_labelmap_from_array(arr, name="AvoidRegionTest")

        distNode = self.logic.compute_distance_map(labelmapNode)
        self.assertIsNotNone(distNode)

        distances, minDist = self.logic.measure_path_distance(distNode, (10, 10, 10), (10, 10, 10))
        self.assertEqual(minDist, 0.0)
        print("[PASS] test_distance_map_creation")

    def test_multiple_avoid_regions(self):
        """
        Test a trajectory when there is a single LabelMapVolume whose “1”s
        appear far from the z-axis, so a straight path along z-axis remains valid.
        """
        arr = np.zeros((40, 40, 40), dtype=np.uint8)
        arr[30:33, 30:33, 30:33] = 1
        arr[35:38, 35:38, 35:38] = 1
        avoidNode = self._create_labelmap_from_array(arr, name="AvoidRegionsCombined")

        entry = (0.0, 0.0, 0.0)
        target = (0.0, 0.0, 10.0)
        constraints = {
            "target_in": None,
            "avoid": [avoidNode],
            "cortex": None,
            "max_length": 20.0
        }
        valid, reason = self.logic.check_trajectory(entry, target, constraints, None, None)
        self.assertTrue(valid)
        print("[PASS] test_multiple_avoid_regions")

    def test_max_trajectory_length_enforced(self):
        """
        Ensure trajectories longer than max_length are rejected.
        """
        entry = (0.0, 0.0, 0.0)
        target = (0.0, 0.0, 100.0)  # length = 100 mm
        constraints = {
            "target_in": None,
            "avoid": [],
            "cortex": None,
            "max_length": 50.0
        }
        valid, reason = self.logic.check_trajectory(entry, target, constraints, None, None)
        self.assertFalse(valid)
        self.assertIn("Trajectory too long", reason)
        print("[PASS] test_max_trajectory_length_enforced")

    def test_target_in_region_constraint(self):
        """
        Check that if the target point lies outside a small “1”-voxel region, the trajectory is invalid.
        We mark only arr[5,5,5]=1, then request a target at (0,0,0), which is outside.
        """
        arr = np.zeros((10, 10, 10), dtype=np.uint8)
        arr[5, 5, 5] = 1
        targetRegionNode = self._create_labelmap_from_array(arr, name="TargetRegionTest")

        entry = (0.0, 0.0, 0.0)
        target = (0.0, 0.0, 0.0)
        constraints = {
            "target_in": targetRegionNode,
            "avoid": [],
            "cortex": None,
            "max_length": 20.0
        }
        valid, reason = self.logic.check_trajectory(entry, target, constraints, None, None)
        self.assertFalse(valid)
        self.assertEqual(reason, "Target outside region")
        print("[PASS] test_target_in_region_constraint")

    def test_angle_constraint(self):
        """
        Create a “flat” slab at z=5 (arr[5,:,:]=1), build a model from that slab.
        Then a path from (0,0,0)→(0,10,0) is oblique to the slab’s normal.
        check_trajectory should reject it as “Angle too oblique.”
        """
        slab = np.zeros((20, 20, 20), dtype=np.uint8)
        slab[5, :, :] = 1
        cortexNode = self._create_labelmap_from_array(slab, name="CortexTest")

        modelNode = self.logic.convert_labelmap_to_model(cortexNode)
        kd_tree, normals = self.logic.get_kdtree_and_normals(modelNode)

        entry = (0.0, 0.0, 0.0)
        target = (0.0, 10.0, 0.0)
        constraints = {
            "target_in": None,
            "avoid": [],
            "cortex": cortexNode,
            "max_length": 50.0
        }
        valid, reason = self.logic.check_trajectory(entry, target, constraints, kd_tree, normals)
        self.assertFalse(valid)
        self.assertIn("Angle too oblique", reason)
        print("[PASS] test_angle_constraint")

    def test_with_real_geometry(self):
        """
        Build a 30×30×30 volume with a spherical “target” and an off-axis spherical “avoid”:
        • Target sphere centered at (15,15,15), radius=5
        • Avoid sphere centered at (25,25,25), radius=3
        Check a path from (0,0,0)→(15,15,15) is valid and
        measure distance to avoid sphere: minDist>0.
        """
        shape = (30, 30, 30)
        arr_target = np.zeros(shape, dtype=np.uint8)
        center_t = np.array([15, 15, 15])
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    if np.linalg.norm(np.array([z, y, x]) - center_t) <= 5:
                        arr_target[z, y, x] = 1
        targetNode = self._create_labelmap_from_array(arr_target, name="SphereTarget")

        arr_avoid = np.zeros(shape, dtype=np.uint8)
        center_a = np.array([25, 25, 25])
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    if np.linalg.norm(np.array([z, y, x]) - center_a) <= 3:
                        arr_avoid[z, y, x] = 1
        avoidNode = self._create_labelmap_from_array(arr_avoid, name="SphereAvoid")

        entry = (0.0, 0.0, 0.0)
        target_pt = (15.0, 15.0, 15.0)
        constraints = {
            "target_in": targetNode,
            "avoid": [avoidNode],
            "cortex": None,
            "max_length": 50.0
        }
        valid, reason = self.logic.check_trajectory(entry, target_pt, constraints, None, None)
        self.assertTrue(valid, f"Expected valid path, but got '{reason}'")

        distNode = self.logic.compute_distance_map(avoidNode)
        distances, minDist = self.logic.measure_path_distance(distNode, entry, target_pt, num=50)
        self.assertGreater(minDist, 0.0)
        print("[PASS] test_with_real_geometry")



