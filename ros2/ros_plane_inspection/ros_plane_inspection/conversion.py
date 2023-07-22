
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from random import randrange

import numpy as np
from typing import List, Tuple

from plane_inspection import box_properties

def create_marker_from_bb(bb: List[np.ndarray], ns: str, frame_id: str, marker_id: int=None, color: Tuple[float, float, float]=(0.0, 0.0, 1.0)) -> Marker:
    """
    Create a visualization_msgs/Marker message from the 8 corner points of a bounding box.

    Args:
    bb (List[np.ndarray]): List of 8 corner points of the bounding box.
    ns (str): Namespace for the marker message.
    frame_id (str): The name of the coordinate frame the bounding box is associated with.
    marker_id (int, optional): The id of the marker message. Defaults to a random integer.
    color (Tuple[float, float, float], optional): The color of the marker message. Defaults to blue.

    Returns:
    visualization_msgs/Marker: Marker message representing the bounding box.
    """
    # Initializing the marker
    marker = Marker()
    marker.id = marker_id if marker_id is not None else randrange(2**16)
    marker.ns = ns
    marker.type = marker.CUBE
    marker.header.frame_id = frame_id
    marker.color.a = 0.5
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]

    #Add corner points of bb
    for corner in bb:
        p = Point()
        p.x = corner[0]
        p.y = corner[1]
        p.z = corner[2]
        marker.points.append(p)

    # Extracting the bounding box properties
    center, scale, q = box_properties(bb)

    # Setting the position, scale, and orientation of the marker based on the bounding box properties
    marker.pose.position.x = center[0]
    marker.pose.position.y = center[1]
    marker.pose.position.z = center[2]
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]

    return marker
