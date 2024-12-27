import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2

class IMUSubscriebr(Node):

    def __init__(self):
        super().__init__('LiSaLocalisation')

        # self.imu_subscriber = self.create_subscription(
        #     Imu,
        #     '/imu/data',
        #     self.imu_callback,
        #     10)
        self.count = 0
        self.lidar_subscriber = self.create_subscription( PointCloud2,"non_ground", self.lidar_callback, 10)
        self.lidar_gps_publisher = self.create_publisher( PointCloud2, "non_ground_gps", 10)
        
    def imu_callback(self, msg):
        orientation = msg.orientation
        r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w]).as_euler('xyz', degrees=False)
        print(r)

    def lidar_callback(self, msg):
        msg.header.frame_id = 'gps'
        self.lidar_gps_publisher.publish(msg)
        print(self.count)
        self.count += 1


def main(args=None):
    rclpy.init(args=args)
    imu_subscriber = IMUSubscriebr()
    rclpy.spin(imu_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()