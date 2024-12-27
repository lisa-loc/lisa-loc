import rclpy
import os
import numpy as np
import open3d as o3d
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import rclpy.time
from sensor_msgs.msg import PointCloud2, Imu
from geometry_msgs.msg import PointStamped
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from scipy.spatial.transform import Rotation as R

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


class SaveToFolder(Node):
    def __init__(self):
        super().__init__('SaveToFolder')

        self.logger = self.get_logger()
        self.logger.info("Save To Folder Node Started")

        # self.subscription = self.create_subscription(
        #     PointCloud2,
        #     '/lidar_center/points',
        #     self.callback,
        #     qos_profile_sensor_data)
        
        # self.gps_subscription = self.create_subscription(
        #     PointStamped,
        #     '/anavs/solution/pos_llh',
        #     self.gps_callback,
        #     qos_profile_sensor_data)
        

        lidar_subscriber = Subscriber(self, PointCloud2, '/lidar_center/points', qos_profile=qos_profile_sensor_data)
        gps_subscriber = Subscriber(self, PointStamped, '/anavs/solution/pos_llh', qos_profile=qos_profile_sensor_data)
        imu_subscriber = Subscriber(self, Imu, '/imu/data', qos_profile=qos_profile_sensor_data)

        self.ts = ApproximateTimeSynchronizer([lidar_subscriber, gps_subscriber, imu_subscriber], 10, 1)

        self.ts.registerCallback(self.callback)

        self.last_gps_msg = None

        self.count = 0
        self.folder = "/home/lisa/datasets/dai2"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.lidar_save_folder = os.path.join(self.folder, "velodyne_points", "data_pcd")
        self.gps_save_folder = os.path.join(self.folder, "oxts")

        if not os.path.exists(self.lidar_save_folder):
            os.makedirs(self.lidar_save_folder)
        
        if not os.path.exists(self.gps_save_folder):
            os.makedirs(self.gps_save_folder)

    def callback(self, lidar_msg, gps_msg, imu_msg):
        self.logger.info(f"Processing: {self.count}")
        self.save_lidar_data(lidar_msg)
        self.save_gps_data(gps_msg, imu_msg)
        self.count += 1

    def save_gps_data(self, gps_msg, imu_msg):
        r = R.from_quat([imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w]).as_euler('xyz', degrees=False)
        
        gps_file = open(os.path.join(self.folder, "oxts","gps_lat_long.txt"), "a")
        gps_file.writelines(f"{gps_msg.point.x} {gps_msg.point.y} {r[2]}\n")
        gps_file.close()               

    def save_lidar_data(self, msg):
        xyz, intensity = point_cloud2_to_array(msg) 
        
        # Save the point cloud
        filename = os.path.join(self.folder, "velodyne_points/data_pcd", f"{self.count}.pcd")

        intensity = intensity.reshape(-1, 1)
        
        point_cloud = o3d.t.geometry.PointCloud()
        point_cloud.point.positions = o3d.core.Tensor(xyz, dtype=o3d.core.float32)
        point_cloud.point.intensities = o3d.core.Tensor(intensity, dtype=o3d.core.float32)

        o3d.t.io.write_point_cloud(filename, point_cloud)

def main(args=None):
    rclpy.init(args=args)

    save_to_folder = SaveToFolder()

    rclpy.spin(save_to_folder)

    save_to_folder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()