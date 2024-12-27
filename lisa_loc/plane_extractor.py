import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d

def field_to_dtype(field):
    if field.datatype == 1:
        return np.int8
    elif field.datatype == 2:
        return np.uint8
    elif field.datatype == 3:
        return np.int16
    elif field.datatype == 4:
        return np.uint16
    elif field.datatype == 5:
        return np.int32
    elif field.datatype == 6:
        return np.uint32
    elif field.datatype == 7:
        return np.float32
    elif field.datatype == 8:
        return np.float64
    else:
        return None
    
def point_cloud2_to_array(cloud_msg):
    ''' 
    Converts a ros PointCloud2 message to a numpy recordarray    
    '''
    # Construct a numpy record array
    field_names = [field.name for field in cloud_msg.fields]
    field_dtypes = [field_to_dtype(field) for field in cloud_msg.fields]
    field_offsets = [field.offset for field in cloud_msg.fields]

    array_dtype = np.dtype({ 'names': field_names,
                             'formats': field_dtypes,
                             'offsets': field_offsets,
                             'itemsize': cloud_msg.point_step})
    
    point_cloud = np.frombuffer(cloud_msg.data, dtype=array_dtype)

    xyz = np.array([point_cloud['x'], point_cloud['y'], point_cloud['z']]).T
    intensity = np.array(point_cloud['intensity'])
    
    return xyz, intensity

class PlaneExtractor(Node):
    def __init__(self):
        super().__init__('point_cloud_extractor')

        self.declare_parameter("lidar_topic", "/lidar_center/points")
        lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value

        self.logger = self.get_logger()
        self.logger.info("LiSa Localisation Node Started")

        self._lidar_subscription = self.create_subscription(PointCloud2, 'non_ground', self.lidar_callback, qos_profile_sensor_data)
        self.planar_publisher = self.create_publisher(PointCloud2, 'planar', qos_profile_sensor_data)

    def lidar_callback(self, msg):
        xyz, intensity = point_cloud2_to_array(msg)   
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz)
        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)

        indices = get_planar_points(point_cloud)

        point_cloud.select_by_index(indices)
        planar_points = np.asarray(point_cloud.points)

        planar_points[:, 2] = 0.0

        planar_msg = pc2.create_cloud_xyz32(header=msg.header, points=planar_points)
        self.planar_publisher.publish(planar_msg)

def get_planar_points(pointcloud):
    try:
        pointcloud.estimate_normals()
        planes = pointcloud.detect_planar_patches()
        indices = []
        for plane in planes:
            # print(np.asarray(plane.get_box_points()))
            # print(is_thin_vertical_cuboid(np.asarray(plane.get_box_points())))
            if plane.get_max_bound()[2] < 5:
                indices += plane.get_point_indices_within_bounding_box(pointcloud.points)
        return indices
    except Exception as e:
        print(e)
        return None      

def main(args=None):
    rclpy.init(args=args)
    plane_extractor = PlaneExtractor()

    executor = MultiThreadedExecutor()
    executor.add_node(plane_extractor)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        plane_extractor.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
