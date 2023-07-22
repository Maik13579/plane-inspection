import numpy as np
import open3d as o3d

from typing import List, Tuple

def db_clustering(cloud: o3d.geometry.PointCloud, eps: float=0.05, min_points: int=20) -> Tuple[List[o3d.geometry.PointCloud], o3d.geometry.PointCloud]:
    """
    Cluster a point cloud using Open3D's DBSCAN method. Returns clusters and noise points.

    Args:
    cloud (o3d.geometry.PointCloud): Input point cloud for clustering.
    eps (float): Density parameter that is used to find neighbouring points.
    min_points (int):  Minimum number of points to form a cluster.

    Returns:
    Tuple[List[o3d.geometry.PointCloud], o3d.geometry.PointCloud]: List of point clouds representing each cluster and a point cloud of noise points.
    """
    clusters = []

    # Perform DBSCAN clustering
    labels = np.array(cloud.cluster_dbscan(eps=eps, min_points=min_points))

    # Select noise points (label == -1)
    noise = cloud.select_by_index(list(np.where(labels == -1)[0]))
    
    # Create a point cloud for each cluster
    for i in range(labels.max()+1):
        cluster = cloud.select_by_index(list(np.where(labels == i)[0]))
        clusters.append(cluster)
    
    return clusters, noise