import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Tuple

from compas.geometry import oriented_bounding_box_numpy

class BoundingBox:
    def __init__(self, corner_points: List[np.ndarray]) -> None:
        """
        Initialize the BoundingBox object.

        Args:
        corner_points (List[np.ndarray]): List of 8 corner points of the bounding box.
        """

        self.corner_points = corner_points

        #compute center, scale and orientation (as quaternion and as Rotation matrix)
        self.center, self.scale, self.quaternion, self.R = self.box_properties(corner_points)

    def box_properties(self, points: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray, np.ndarray]:
        """
        Compute center, scale, orientation (as quaternion), and rotation matrix from the 8 corners of a box.

        Args:
        points (List[np.ndarray]): List of 8 corner points of the bounding box.

        Returns:
        Tuple[np.ndarray, Tuple[float, float, float], np.ndarray, np.ndarray]: The center, scale, orientation of the box (as quaternion), and rotation matrix.
        """

        # Ensure the input is a numpy array
        points = np.array(points)

        # Compute the center of the box
        center = np.mean(points, axis=0)

        # Compute the scale of the box in each direction
        scale_x = np.linalg.norm(points[1] - points[0])
        scale_y = np.linalg.norm(points[3] - points[0])
        scale_z = np.linalg.norm(points[4] - points[0])

        # Compute the orientation of the box
        u1 = (points[1] - points[0]) / scale_x
        u2 = (points[3] - points[0]) / scale_y
        u3 = (points[4] - points[0]) / scale_z

        # Construct the rotation matrix
        rot_matrix = np.array([u1, u2, u3]).T

        # Convert the rotation matrix to a quaternion
        rot = Rotation.from_matrix(rot_matrix)
        quaternion = rot.as_quat()

        return center, (scale_x, scale_y, scale_z), quaternion, rot_matrix


def compute_bb(points: np.ndarray) -> BoundingBox:
    """
    Compute an oriented bounding box for a set of points.

    Args:
    points (np.ndarray): Array of points.

    Returns:
    BoundingBox: Oriented bounding box for the points.
    """
    return BoundingBox(oriented_bounding_box_numpy(points))

def compute_bb_on_plane(points: np.ndarray, plane) -> BoundingBox:
    """
    Project the points onto the plane and compute an oriented bounding box.

    Args:
    points (np.ndarray): Array of points.
    plane (Plane): Plane object.

    Returns:
    BoundingBox: Oriented bounding box extendet to the plane.
    """
    projected_points = plane.project_points_onto_plane(points)
    combined = np.concatenate((np.asarray(projected_points), np.asarray(points)))
    return compute_bb(combined)