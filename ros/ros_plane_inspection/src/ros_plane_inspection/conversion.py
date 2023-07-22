
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from random import randrange

from typing import Tuple

from plane_inspection import BoundingBox

def create_marker_from_bb(bb: BoundingBox, ns: str, frame_id: str, marker_id: int=None, color: Tuple[float, float, float]=(0.0, 0.0, 1.0, 0.5)) -> Marker:
    """
    Create a visualization_msgs/Marker message from a BoundingBox object.

    Args:
    bb (BoundingBox): BoundingBox object.
    ns (str): Namespace for the marker message.
    frame_id (str): The name of the coordinate frame the bounding box is associated with.
    marker_id (int, optional): The id of the marker message. Defaults to a random integer.
    color (Tuple[float, float, float, float], optional): The RGBA color of the marker message. Defaults to blue with 0.5 alpha.

    Returns:
    visualization_msgs/Marker: Marker message representing the bounding box.
    """
    # Initializing the marker
    marker = Marker()
    marker.id = marker_id if marker_id is not None else randrange(2**16)
    marker.ns = ns
    marker.type = marker.CUBE
    marker.header.frame_id = frame_id
    marker.color.a = color[3]
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]

    #Add corner points of bb
    for corner in bb.corner_points:
        p = Point()
        p.x = corner[0]
        p.y = corner[1]
        p.z = corner[2]
        marker.points.append(p)

    # Setting the position, scale, and orientation of the marker based on the bounding box properties
    marker.pose.position.x = bb.center[0]
    marker.pose.position.y = bb.center[1]
    marker.pose.position.z = bb.center[2]
    marker.scale.x = bb.scale[0]
    marker.scale.y = bb.scale[1]
    marker.scale.z = bb.scale[2]
    marker.pose.orientation.x = bb.quaternion[0]
    marker.pose.orientation.y = bb.quaternion[1]
    marker.pose.orientation.z = bb.quaternion[2]
    marker.pose.orientation.w = bb.quaternion[3]

    return marker
