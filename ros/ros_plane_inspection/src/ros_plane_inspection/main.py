#!/usr/bin/env python3

import rospy
import open3d as o3d
import numpy as np
import cv2

from ros_plane_inspection_interfaces.srv import InspectPlane, InspectPlaneResponse, InspectPlaneRequest
from sensor_msgs.msg import PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from visualization_msgs.msg import MarkerArray
import tf2_ros
from vision_msgs.msg import Detection2DArray, Detection2D
from visualization_msgs.msg import Marker
from typing import Dict

from ros_plane_inspection.conversion import create_marker_from_bb
from plane_inspection import detect_plane

# Constants defining the frame and topic to subscribe to
FRAME = 'base_footprint'
POINTCLOUD_TOPIC = '/xtion/depth/points'
CAMERA_FRAME = 'xtion_rgb_optical_frame'
CAMERA_INFO_TOPIC = '/xtion/rgb/camera_info'

# Ordered frameworks with highest priority being the first element
OBJECT_RECOGNITION = [
    {"framework": "yolov8", "topic": "/yolov8_ros/detections"},
    {"framework": "yolov7_ycb", "topic": "/yolov7_ros/detection"},
    {"framework": "mask2former", "topic": "/maskformer2"},
]


class PlaneInspection:
    """
    This class performs a Plane Inspection on a 3D Point Cloud utilizing ROS and Open3D.
    The class includes ROS publishers, subscribers, and a service that detects and publishes the largest
    horizontal plane in a point cloud, along with bounding boxes around clusters.
    """
    def __init__(self, pointcloud_topic: str = POINTCLOUD_TOPIC, camera_info_topic: str = CAMERA_INFO_TOPIC):
        """
        The constructor for the PlaneInspection class initializes the ROS node, Transform Listener and Buffer for 
        Cloud Transformation, ROS publishers, ROS subscribers and ROS service. Finally, it starts the ROS node.
        """
        
        # Initialize the ROS node
        rospy.init_node("plane_inspection")
        
        # Initialize Transform Listener and Buffer for Cloud Transformation
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize ROS Publishers
        self.pub = rospy.Publisher('/plane_inspection/markers', MarkerArray, queue_size=10)

        # Initialize ROS Subscriber
        self.cloud_msg = None
        rospy.Subscriber(pointcloud_topic, PointCloud2, lambda msg: setattr(self, 'cloud_msg', msg))
        
        self.K = None
        self.D = None
        rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_cb)

        # Initialize Object Recognition Subscriber
        for obj_rec in OBJECT_RECOGNITION:
            setattr(self, obj_rec["framework"], None)
            rospy.Subscriber(obj_rec["topic"], Detection2DArray, lambda msg, obj_rec=obj_rec: setattr(self, obj_rec["framework"], msg))
    
        # Initialize ROS Service
        rospy.Service('/plane_inspection/inspect_plane', InspectPlane, self.service_inspect_plane)

        # Keep the node running
        rospy.spin()

    def camera_info_cb(self, msg: CameraInfo) -> None:
        """
        The callback function for the camera_info topic.
        This function saves the camera matrix and distortion coefficients.
        """
        self.K = np.array(msg.K).reshape((3, 3))
        self.D = msg.D

    def service_inspect_plane(self, req: InspectPlaneRequest) -> InspectPlaneResponse:
        """
        The callback function for the inspect_plane service.
        This function performs plane detection, point filtering, clustering, and bounding box calculation. 

        Parameters:
        req (InspectPlaneRequest): The request message which includes the following parameters: base_footprint_frame, 
                                min_height, max_height, distance_threshold, ransac_n, num_iterations, horizontal_threshold,
                                plane_min_size_x, plane_min_size_y, eps, and min_points.

        Returns:
        InspectPlaneResponse: The response message which includes the detected plane, the plane's equation coefficients, 
                            and bounding boxes for known and unknown objects.

        Note:
        If no object recognition is available, all objects are added as unknown.
        """
        res = InspectPlaneResponse()
        marker_array = MarkerArray()
        
        # Check if point cloud is available
        if self.cloud_msg is None:
            rospy.loginfo('No pointcloud available!')
            return res

        # Reset detected objects
        self.detected_objects = {}

        # Initialize request parameters with provided values or defaults
        base_footprint = req.base_footprint_frame if req.base_footprint_frame != '' else FRAME
        min_height = req.min_height if req.min_height != 0.0 else 0.3
        max_height = req.max_height if req.max_height != 0.0 else 2.0
        distance_threshold = req.distance_threshold if req.distance_threshold != 0.0 else 0.01
        ransac_n = req.ransac_n if req.ransac_n != 0 else 3
        num_iterations = req.num_iterations if req.num_iterations != 0 else 1000
        horizontal_threshold = req.horizontal_threshold if req.horizontal_threshold != 0.0 else 0.1
        plane_min_size_x = req.plane_min_size_x if req.plane_min_size_x != 0.0 else 0.1
        plane_min_size_y = req.plane_min_size_y if req.plane_min_size_y != 0.0 else 0.1
        plane_min_size = (plane_min_size_x, plane_min_size_y)
        eps = req.eps if req.eps != 0.0 else 0.025
        min_points = req.min_points if req.min_points != 0 else 50

        # Get the point cloud data
        scan = self.get_scan_cloud(base_footprint, min_height, max_height)
        
        # Detect the largest horizontal plane
        rospy.loginfo('Detect plane:')
        plane, plane_bb_floor = detect_plane(scan, distance_threshold,
                                             ransac_n, num_iterations,
                                             horizontal_threshold,
                                             plane_min_size)
        if plane is None:
            rospy.loginfo('No plane detected!')
            return res

        # Convert bounding boxes to marker messages
        plane_marker = create_marker_from_bb(plane.bb, 'plane', base_footprint, color=(1.0, 0.0, 0.0, 0.5))
        plane_marker_floor = create_marker_from_bb(plane_bb_floor, 'plane_floor', base_footprint, color=(1.0, 0.0, 0.0, 0.5))

        # Add plane markers to response
        res.plane = plane_marker
        res.plane_floor = plane_marker_floor
        marker_array.markers.append(plane_marker)
        marker_array.markers.append(plane_marker_floor)
        res.a = plane.a
        res.b = plane.b
        res.c = plane.c
        res.d = plane.d
        rospy.loginfo(f"Plane equation: {plane.a:.2f}x + {plane.b:.2f}y + {plane.c:.2f}z + {plane.d:.2f} = 0")

        # Detect objects on plane
        bbs = plane.detect_objects_on_plane(scan, distance_threshold, eps, min_points)

        # Convert bbs to marker arrays
        marker_bbs = {}
        for bb in bbs:
            marker = create_marker_from_bb(bb, 'unkown_objects', base_footprint, color=(0.0, 0.0, 1.0, 0.5))
            marker.header.stamp = rospy.Time.now()
            marker_bbs[marker.id] = marker


        # Recognize objects
        detected_cluster_ids = []
        if self.obj_rec_available() and self.K is not None:
            rospy.loginfo('Recognize objects:')
            detections = self.rec_objects(marker_bbs)
    
            #add known objects
            for key, cluster_id in detections.items():
                framework, id = key.rsplit('_', 1) #split key in framework and is

                rospy.loginfo(str(framework)+' detected object with id: '+str(id))
                marker_bbs[cluster_id].ns = str(framework)
                marker_bbs[cluster_id].id = int(id)

                #change color to green
                marker_bbs[cluster_id].color.b = 0.0
                marker_bbs[cluster_id].color.g = 1.0

                detected_cluster_ids.append(cluster_id)
                marker_array.markers.append(marker_bbs[cluster_id])
                res.objects_bb.append(marker_bbs[cluster_id])

        #add unkown object
        for id, marker in marker_bbs.items():
            if id not in detected_cluster_ids:
                marker_array.markers.append(marker)
                res.unkown_objects_bb.append(marker)

        self.pub.publish(marker_array)
        rospy.loginfo('#'*50)
        rospy.loginfo('Final Result:')
        rospy.loginfo(f"Plane equation: {plane.a:.2f}x + {plane.b:.2f}y + {plane.c:.2f}z + {plane.d:.2f} = 0")
        rospy.loginfo('Unkown objects: '+str(len(res.unkown_objects_bb)))
        rospy.loginfo('Known objects: '+str(len(res.objects_bb)))
        return res

    def get_scan_cloud(self, target_frame: str, min_height: float=0.3, max_height:float=2.0) -> o3d.geometry.PointCloud:
        """
        This method transforms the point cloud to the target frame, filters points that are not within the defined height 
        range (min_height to max_height), and creates an Open3D PointCloud object.

        Parameters:
        target_frame (str): The target frame to which the point cloud will be transformed.
        min_height (float, optional): The minimum height of points. Points below this height will be filtered out. Defaults to 0.3.
        max_height (float, optional): The maximum height of points. Points above this height will be filtered out. Defaults to 2.0.

        Returns:
        o3d.geometry.PointCloud: An Open3D PointCloud object with points within the specified height range and transformed to the target frame.
        """

        # Transform the point cloud to target frame
        scan_msg = self.transform_cloud(self.cloud_msg, target_frame)

        points = list(pc2.read_points(scan_msg, skip_nans=True))

        # Remove points that don't lie within the defined height range
        points = [p[:3] for p in points if p[2] >= min_height and p[2] <= max_height]

        # Create Open3D point cloud
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(points)
        return o3d_cloud
    
    def transform_cloud(self, cloud: PointCloud2, target_frame: str) -> PointCloud2:
        """
        This method transforms a ROS PointCloud2 message to a target frame using tf2.

        Parameters:
        cloud (PointCloud2): A ROS PointCloud2 message that will be transformed.
        target_frame (str): The target frame to which the point cloud will be transformed.

        Returns:
        PointCloud2: A transformed PointCloud2 message in the target frame.
        """

        trans = self.tf_buffer.lookup_transform(target_frame,
                                                cloud.header.frame_id,
                                                rospy.Time(0),
                                                rospy.Duration(1.0))
        return do_transform_cloud(cloud, trans)
    
    def transform_pose(self, pose: PoseStamped, target_frame: str) -> PoseStamped:
        """
        This method transforms a ROS PoseStamped message to a target frame using tf2.

        Parameters:
        pose (PoseStamped): A ROS PoseStamped message that will be transformed.
        target_frame (str): The target frame to which the pose will be transformed.

        Returns:
        PoseStamped: A transformed PoseStamped message in the target frame.
        """

        trans = self.tf_buffer.lookup_transform(target_frame,
                                                pose.header.frame_id,
                                                rospy.Time(0),
                                                rospy.Duration(1.0))
        return do_transform_pose(pose, trans)
    
    def obj_rec_available(self) -> bool: 
        """
        This method checks if there is at least one object recognition framework running.

        Returns:
        bool: True if there is at least one object recognition framework running, else False.
        """

        for obj_rec in OBJECT_RECOGNITION:
            if getattr(self, obj_rec['framework']) is not None:
                return True
        return False

    def rec_objects(self, bbs: Dict[int, Marker]) -> Dict[str, int]:
        """
        This method iterates over the available object recognition frameworks to recognize objects.
        If there are multiple objects in a row and all fit into the detected 2D bounding box,
        it chooses the one with the smallest x value (the one nearest to the robot).

        Parameters:
        bbs (Dict[int, Marker]): A dictionary where the key is the id of the bounding box and the value 
                                is a marker representing the bounding box of an object.

        Returns:
        detections (Dict[str, int]): A dictionary where the key is a combination of the framework name and 
                                    the id of the detected object, and the value is the id of the bounding box.
        """

        detections = {}
        framework = ""
        for id, marker in bbs.items():
            for obj_rec in OBJECT_RECOGNITION:
                framework = obj_rec['framework']
                detection = self.identify_object(marker, getattr(self, framework))
                if detection is not None:
                    break
            
            if detection is not None: #object detected
                detection_key = f"{framework}_{detection.results[0].id}"

                #check if there is already a detection
                if detection_key in detections.keys():

                    #if there are multiply clusters with the same detection take the one which is closer
                    if marker.pose.position.x > bbs[detections[detection_key]].pose.position.x:
                        continue
                detections[detection_key] = id
        return detections

    def identify_object(self, marker: Marker, det_msg: Detection2DArray) -> Detection2D:
        """
        Identify the object inside the bounding box using YOLO detection results.
        
        Parameters:
        marker (Marker): A Marker representing the bounding box of an object.
        det_msg (Detection2DArray): The message containing detection results from YOLO or another object detection framework.
        
        Returns:
        detection (Detection2D): The Detection2D corresponding to the identified object. 
                                    Returns None if no object could be identified.

        TODO use object recognition pipeline :)
        """
        if det_msg is None:
            return None
        
        # Transform center of object to camera frame
        pose = PoseStamped()
        pose.pose = marker.pose
        pose.header = marker.header
        pose = self.transform_pose(pose, CAMERA_FRAME).pose

        # Get center coords
        center = np.float64([pose.position.x, pose.position.y, pose.position.z])

        # Compute pixel
        imgpt, _ = cv2.projectPoints(center, np.zeros(3), np.zeros(3), self.K, self.D)

        # Find closest bounding box from yolo
        min_distance = float('inf')
        min_index = -1
        
        for index, detection in enumerate(det_msg.detections):
            distance = np.sqrt((detection.bbox.center.x - imgpt[0,0,0])**2 + (detection.bbox.center.y - imgpt[0,0,1])**2)
            if distance < min_distance:
                min_distance = distance
                min_index = index

        if min_index == -1: 
            return None
        
        detection = det_msg.detections[min_index]
        
        # Check if pixel is inside the bounding box
        if abs(detection.bbox.center.x - imgpt[0,0,0]) > detection.bbox.size_x/2:
            return None
        if abs(detection.bbox.center.y - imgpt[0,0,1]) > detection.bbox.size_y/2:
            return None
        return detection
    


if __name__ == '__main__':
    # Instantiate the PlaneInspection class
    import sys

    if len(sys.argv) > 1:
        PlaneInspection(sys.argv[1])
    elif len(sys.argv) > 2:
        PlaneInspection(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3:
        FRAME = sys.argv[3]
        PlaneInspection(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 4:
        FRAME = sys.argv[3]
        CAMERA_FRAME = sys.argv[4]
        PlaneInspection(sys.argv[1], sys.argv[2])
    else:
        PlaneInspection()
