import rclpy
import os
import numpy as np
import open3d as o3d
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import rclpy.time
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped

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

        self.subscription = self.create_subscription(
            PointCloud2,
            '/lidar_center/points',
            self.callback,
            qos_profile_sensor_data)
        
        self.gps_subscription = self.create_subscription(
            PointStamped,
            '/anavs/solution/pos_llh',
            self.gps_callback,
            qos_profile_sensor_data)

        self.subscription

        self.last_gps_msg = None

        self.count = 0
        self.folder = "/home/lisa/lisa_ws/data/point_clouds"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def save_to_folder(self, msg):
        if not self.last_gps_msg:
            return
        
        xyz, intensity = point_cloud2_to_array(msg)
        

        # Create a folder to save the point cloud
        

        # with open(os.path.join(folder, "gps.txt"), "w") as gps_file:
        #     timestamp = rclpy.time.Time.from_msg(self.last_gps_msg.header.stamp)
        #     gps_file.writelines(f"{timestamp.nanoseconds}, {self.last_gps_msg.point.x}, {self.last_gps_msg.point.y}, {self.last_gps_msg.point.z}")   

        gps_file = open(os.path.join(self.folder, "gps.txt"), "a")
        timestamp = rclpy.time.Time.from_msg(self.last_gps_msg.header.stamp)
        gps_file.writelines(f"{self.count}, {timestamp.nanoseconds}, {self.last_gps_msg.point.x}, {self.last_gps_msg.point.y}, {self.last_gps_msg.point.z}\n")
        gps_file.close()       

        # Save the point cloud
        filename = os.path.join(self.folder, f"{self.count}.pcd")

        intensity = intensity.reshape(-1, 1)
        
        point_cloud = o3d.t.geometry.PointCloud()
        point_cloud.point.positions = o3d.core.Tensor(xyz, dtype=o3d.core.float32)
        point_cloud.point.intensities = o3d.core.Tensor(intensity, dtype=o3d.core.float32)

        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(xyz)

        # o3d.io.write_point_cloud(filename, point_cloud)

        o3d.t.io.write_point_cloud(filename, point_cloud)

        timestamp_file = open(os.path.join(self.folder, "timestamps.txt"), "a")
        timestamp = rclpy.time.Time.from_msg(msg.header.stamp)
        timestamp_file.writelines(f"{self.count},{timestamp.nanoseconds}\n")

        self.count += 1


    def callback(self, msg):
        self.logger.info("Received Point Cloud Message")
        if not self.last_gps_msg:
            self.logger.info("No GPS message received")
            return
        
        self.save_to_folder(msg)
        

    def gps_callback(self, msg):
        if not self.last_gps_msg:
            gps_file = open(os.path.join(self.folder, "gps.txt"), "w")
            gps_file.write("")
            gps_file.close()
        self.last_gps_msg = msg

def main(args=None):
    rclpy.init(args=args)

    save_to_folder = SaveToFolder()

    rclpy.spin(save_to_folder)

    save_to_folder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()