import numpy as np
import open3d as o3d
from typing import List, Tuple, Union

from .boundingbox_utils import BoundingBox, compute_bb, compute_bb_on_plane
from .pointcloud_utils import db_clustering


class Plane:
    def __init__(self, plane_model: List[float], plane_bb: BoundingBox, cloud: o3d.geometry.PointCloud) -> None:
        """
        Initialize Plane object.

        Args:
        plane_model (List[float]): List of parameters for plane equation [a, b, c, d].
        plane_bb (BoundingBox): Bounding box of the detected plane.
        cloud (o3d.geometry.PointCloud): Point cloud of all inliers.
        """

        self.a, self.b, self.c, self.d = plane_model
        self.bb = plane_bb
        self.cloud = cloud

    def get_plane_equation(self) -> List[float]:
        """
        Get the plane equation parameters.
        """

        return self.a, self.b, self.c, self.d

    def normalize(self) -> None:
        """
        Normalize the plane equation parameters.
        Returns: The normalized plane equation parameters.
        """

        magnitude = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        a = self.a / magnitude
        b = self.b / magnitude
        c = self.c / magnitude
        d = self.d / magnitude
        return a, b, c, d
    
    def project_points_onto_plane(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Projects a set of points onto the plane.

        Args:
        points (np.ndarray): A numpy array of points to be projected.

        Returns:
        List[np.ndarray]: A list of numpy arrays representing the points projected onto the plane.
        """

        # Get normalized plane equation
        a, b, c, d = self.normalize()

        # Compute the projection of each point onto the plane
        projected_points = []
        for point in points:
            x, y, z = point

            # Compute the distance from the point to the plane
            distance = a * x + b * y + c * z + d

            # Project the point onto the plane
            projected_point = np.array([x - a * distance, y - b * distance, z - c * distance])
            projected_points.append(projected_point)

        return projected_points
    
    def detect_objects_on_plane(self, cloud: o3d.geometry.PointCloud, threshold: float=0.01, eps: float=0.025, min_points: int=50) -> List[BoundingBox]:
        """
        Detect objects present on the plane using DBSCAN for clustering and computing bounding boxes for each cluster.

        Args:
        cloud (o3d.geometry.PointCloud): Input point cloud.
        threshold (float): The distance used to offset the plane in the normal direction.
        eps (float): Density parameter that is used to find neighbouring points.
        min_points (int):  Minimum number of points to form a cluster.

        Returns:
        List[BoundingBox]: List of bounding boxes on the plane.
        """

        # Remove points below the plane
        print("Remove points below the plane")
        cloud = self.remove_points_below_plane(cloud, threshold)

        # Cluster the points above the plane
        print("Cluster points spatially")
        clusters, _ = db_clustering(cloud, eps, min_points)
        print(str(len(clusters))+" clusters detected")

        # Compute bounding boxes for each cluster
        print("compute bounding boxes for each cluster")
        plane_eq = self.get_plane_equation()
        bbs = [compute_bb_on_plane(cluster.points, plane_eq) for cluster in clusters]

        # Get x and y coordinates of the plane bb corners
        plane_x = [corner[0] for corner in self.bb.corner_points]
        plane_y = [corner[1] for corner in self.bb.corner_points]

        # Check if bb is ontop of plane
        bbs_on_plane = []
        for bb in bbs:
            if bb.center[0] < min(plane_x) or \
               bb.center[0] > max(plane_x) or \
               bb.center[1] < min(plane_y) or \
               bb.center[1] > max(plane_y):
                continue

            bbs_on_plane.append(bb)
        return bbs_on_plane
            

    def remove_points_below_plane(self, cloud: o3d.geometry.PointCloud, threshold: float=0.01) -> o3d.geometry.PointCloud:
        """
        Remove points that are below the plane.

        Args:
        cloud (o3d.geometry.PointCloud): The input point cloud.
        threshold (float): The distance used to offset the plane in the normal direction.

        Returns:
        o3d.geometry.PointCloud: Point cloud containing all points that are above the plane.
        """
        
        points = cloud.points

        # Select points above the plane
        condition = self.a * points[:, 0] + self.b * points[:, 1] + self.c * points[:, 2] + self.d > threshold

        # Create Open3D point cloud from the points above the plane
        above_plane = o3d.geometry.PointCloud()
        above_plane.points = o3d.utility.Vector3dVector(points[condition])
        return above_plane




def detect_plane(cloud: o3d.geometry.PointCloud, distance_threshold: float=0.02,
                 ransac_n: int=3, num_iterations: int=1000, probability: float=0.99999999, horizontal_threshold: float=0.1,
                 plane_min_size: Tuple[float, float]=(0.3, 0.3)) -> Union[Tuple[Plane, BoundingBox], None]:
    """
    Detect the largest horizontal plane in an Open3D point cloud.

    Args:
    cloud (o3d.geometry.PointCloud): Input point cloud for plane detection.
    distance_threshold (float): Max distance a point can be from the plane model, and still be considered an inlier.
    ransac_n (int): Number of initial points to be considered inliers in each iteration.
    num_iterations (int): The maximum number of iterations for finding a plane.
    probability (float):  Expected probability of finding the optimal plane.
    horizontal_threshold (float): Defines what 'horizontal' means. If the x and y parts of the normal vector are lower than this, the plane is considered as horizontal.
    plane_min_size (Tuple[float, float]): Minimum size of the plane to be considered valid.

    Returns:
    Union[Tuple[Plane, BoundingBox], None]: Plane object representing detected plane and its bounding box extended to the floor (useful for moveit planning scene, if it's a table for example). Returns None if no plane is found.
    """

    while len(cloud.points) >= ransac_n:  # While there are enough points to compute a plane

        # Detect the largest plane in the cloud using open3d's RANSAC method
        plane_model, index = cloud.segment_plane(distance_threshold, ransac_n, num_iterations, probability)
        inlier_cloud = cloud.select_by_index(index)
        
        # Remove detected plane from cloud to avoid detecting the same plane again
        cloud = cloud.select_by_index(index, invert=True)
        [a, b, c, d] = plane_model

        # The normal vector should point in the z axis; otherwise, it is not a horizontal plane
        # This only holds when the point cloud is in an frame aligned with the world frame like base_link or map
        if abs(a) > horizontal_threshold or abs(b) > horizontal_threshold:
            continue

        # Cluster all points inside the plane using open3d DBSCAN method
        clusters, _ = db_clustering(inlier_cloud, eps=0.04, min_points=100)

        # Compute Bounding Boxes for each cluster
        # Return the first cluster where the scale fits the minimum size
        for cluster in clusters:

            # Bounding box of the plane
            bb = compute_bb(cluster.points)
            
            # Check if the bounding box is large enough
            if bb.scale[0] > plane_min_size[0] and bb.scale[1] > plane_min_size[1]:

                # Bounding box extendet to the floor, good for moveit planning scene (if its a table for example))
                bb_floor = compute_bb_on_plane(cluster.points, [0.0, 0.0, 1.0, 0.0])
                return Plane(plane_model, bb, cluster), bb_floor

    return None