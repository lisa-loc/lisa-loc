import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import pypatchworkpp
import numpy as np
import small_gicp

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
    ring = np.array(point_cloud['ring'])
    
    return xyz, intensity, ring

class ScanToModelMatchingOdometry(object):
  def __init__(self, num_threads):
    self.num_threads = num_threads
    self.T_last_current = np.identity(4)
    self.T_world_lidar = np.identity(4)
    self.target = small_gicp.GaussianVoxelMap(1.0)
    self.target.set_lru(horizon=100, clear_cycle=10)

  def estimate(self, raw_points):
    downsampled, tree = small_gicp.preprocess_points(raw_points, 0.25, num_threads=self.num_threads)

    if self.target.size() == 0:
      self.target.insert(downsampled)
      return self.T_world_lidar

    result = small_gicp.align(self.target, downsampled, self.T_world_lidar @ self.T_last_current, num_threads=self.num_threads)

    self.T_last_current = np.linalg.inv(self.T_world_lidar) @ result.T_target_source
    self.T_world_lidar = result.T_target_source
    self.target.insert(downsampled, self.T_world_lidar)

    return self.T_world_lidar


class GroundExtractor(Node):
    def __init__(self):
        super().__init__('ground_extractor')

        self.logger = self.get_logger()

        self.logger.info("Ground Extractor Node Started")

        self.declare_parameter("lidar_topic", "/lidar_center/points")
        lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value

        self.lidar_subscription = self.create_subscription(PointCloud2, lidar_topic, self.lidar_callback, qos_profile_sensor_data)

        self.slam_points_publisher = self.create_publisher(PointCloud2, '/points', qos_profile_sensor_data)
        self.ground_publisher = self.create_publisher(PointCloud2, '/ground', qos_profile_sensor_data)
        self.non_ground_publisher = self.create_publisher(PointCloud2, '/non_ground', qos_profile_sensor_data)
        self.wall_publisher = self.create_publisher(PointCloud2, '/wall', qos_profile_sensor_data)

        self.buffer = []
        self.odom_buffer = []

        params = pypatchworkpp.Parameters()
        params.verbose = False   
        self.PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    def calculate_distance(self, matrices):
        distances = []
        for i in range(1, len(matrices)):
            diff = matrices[i][:3, 3] - matrices[i-1][:3, 3]
            distances.append(np.linalg.norm(diff))
        return sum(distances)

    def lidar_callback(self, msg):
        xyz, intensity, ring = point_cloud2_to_array(msg)
        # self.logger.info(f"Received {len(xyz)} points")

        # print(len(xyz), len(intensity), len(ring))

        point_cloud = np.column_stack((xyz, intensity))
        self.buffer.append(point_cloud)

        scan_model = ScanToModelMatchingOdometry(8)

        if len(self.buffer) < 15:    
            return
        else:
            self.buffer.pop(0)
        
        # print(len(self.buffer))

        inner_buffer = self.buffer[::-1]
        # print(inner_buffer)
        inner_point_cloud = None
        
        for i in range(len(inner_buffer)):
            T = scan_model.estimate(inner_buffer[i])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(inner_buffer[i][:,:3])
            pcd.transform(T)
            self.odom_buffer.append(T)
            if inner_point_cloud is None:
                inner_point_cloud = np.array(pcd.points)
            else:
                inner_point_cloud = np.vstack((inner_point_cloud, np.array(pcd.points)))

        # distance = self.calculate_distance(self.odom_buffer)
        # print(distance)
        # self.odom_buffer = []
        # if distance > 5:
        #     self.buffer.pop(0)
        
        self.PatchworkPLUSPLUS.estimateGround(point_cloud)

        ground_idx = self.PatchworkPLUSPLUS.getGroundIndices()
        nonground_idx = self.PatchworkPLUSPLUS.getNongroundIndices()

        fields = ['x', 'y', 'z', 'intensity']

        walls = point_cloud[nonground_idx]
        walls = walls[walls[:,2] > 15]
        walls[:2] = 0

        msg_fields = [field for field in msg.fields if field.name in fields]

        if self.slam_points_publisher.get_subscription_count() != 0:
            slam_msg = pc2.create_cloud(msg.header, msg_fields, points=point_cloud)
            self.slam_points_publisher.publish(slam_msg)

        if self.ground_publisher.get_subscription_count() != 0:
            ground_msg = pc2.create_cloud(msg.header, msg_fields, points=point_cloud[ground_idx])
            # ground_msg = pc2.create_cloud(msg.header, msg_fields, points=ground)
            self.ground_publisher.publish(ground_msg)

        if self.non_ground_publisher.get_subscription_count() != 0:
            non_ground_msg = pc2.create_cloud(msg.header, msg_fields, points=point_cloud[nonground_idx])
            self.non_ground_publisher.publish(non_ground_msg)
        
        if self.wall_publisher.get_subscription_count() != 0:
            wall_msg = pc2.create_cloud(msg.header, msg_fields, points=walls)
            self.wall_publisher.publish(wall_msg)

def main(args=None):
    rclpy.init(args=args)
    ground_extractor = GroundExtractor()
    executor = MultiThreadedExecutor()
    executor.add_node(ground_extractor)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        ground_extractor.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()