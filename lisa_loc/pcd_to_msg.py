import os
import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from pypcd4 import PointCloud
import re
import time

def numerical_sort_key(file_name):
  """Extracts the numeric prefix from a filename and returns it as an integer."""
  match = re.match(r'(\d+)\.pcd', file_name)
  if match:
    return int(match.group(1))
  else:
    return 0 

class PCDToMSG(Node):

    def __init__(self):
        super().__init__('PCDToMSG')

        # Get the logger
        self.logger = self.get_logger()
        self.logger.info("PCD To MSG")

        # Get the parameters from the parameter server
        self.declare_parameter("lidar_topic", "/lidar_center/points")
        self.declare_parameter("pointcloud_folder", "/home/lisa/datasets/dai1/velodyne_points/data_pcd/")

        self.lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        self.pointcloud_folder = self.get_parameter("pointcloud_folder").get_parameter_value().string_value

        # Log the topics
        self.logger.info("Lidar topic: %s" % self.lidar_topic)

        # Create Subscriptions
        self._lidar_publisher = self.create_publisher(PointCloud2, '/lidar_center/points', qos_profile_sensor_data)

        self.files = self._list_folder(self.pointcloud_folder)
        self.files.sort(key=numerical_sort_key)

        self.count = 0
        
        self._main()

    def _list_folder(self, folder):
        files = os.listdir(folder)
        # print(files)
        files.sort(key=numerical_sort_key) 
        return files

    def _main(self):
        for i in range(len(self.files)-1):
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'lidar'
            pointcloud = PointCloud.from_path(self.pointcloud_folder + self.files[i]).to_msg(header)
            self._lidar_publisher.publish(pointcloud)
            time.sleep(0.1)



def main(args=None):
    rclpy.init(args=args)
    pcd_to_msg = PCDToMSG()
    try:
        rclpy.spin(pcd_to_msg)
    finally:
        pcd_to_msg.destroy_node()
        rclpy.shutdown()
    

if __name__ == '__main__':
    main()