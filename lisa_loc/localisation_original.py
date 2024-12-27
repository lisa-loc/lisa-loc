import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import PointCloud2, Imu, NavSatFix
from pypcd4 import PointCloud
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
import os
import utils.file
import utils.map
import utils.registration
import time
import open3d as o3d
import numpy as np
import small_gicp
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import pypatchworkpp
import math

class Localisation(Node):
    def __init__(self):
        super().__init__('LiSaLocalisation')
        self.logger = self.get_logger()

        self.declare_parameter("lidar_topic", "lidar")
        self.declare_parameter("lidar_data_folder", "/home/lisa/datasets/2011_09_26_drive_0091_sync/2011_09_26/2011_09_26_drive_0091_sync/velodyne_points/data_pcd/")
        self.declare_parameter("lidar_buffer_size", 5)
        self.declare_parameter("gps_data_folder", "/home/lisa/datasets/2011_09_26_drive_0091_sync/2011_09_26/2011_09_26_drive_0091_sync/oxts/")
        self.declare_parameter("gps_file", "gps_lat_long.txt")
        self.declare_parameter("map_data_folder", "/home/lisa/datasets/2011_09_26_drive_0091_sync/2011_09_26/2011_09_26_drive_0091_sync/map/mapbox_images_pcd")

        self.lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        self.lidar_data_folder = self.get_parameter("lidar_data_folder").get_parameter_value().string_value
        self.lidar_buffer_size = self.get_parameter("lidar_buffer_size").get_parameter_value().integer_value
        self.gps_data_folder = self.get_parameter("gps_data_folder").get_parameter_value().string_value
        self.gps_file = self.get_parameter("gps_file").get_parameter_value().string_value
        self.map_data_folder = self.get_parameter("map_data_folder").get_parameter_value().string_value
        self.dai = False

        # kitti 84
        # self.lidar_data_folder = "/home/lisa/datasets/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/velodyne_points/data_pcd/"
        # self.gps_data_folder = "/home/lisa/datasets/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/oxts/"
        # self.map_data_folder = "/home/lisa/datasets/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/map/mapbox_images_pcd"

        # kitti 93
        # self.lidar_data_folder = "/home/lisa/datasets/2011_09_26_drive_0093_sync/2011_09_26/2011_09_26_drive_0093_sync/velodyne_points/data_pcd/"
        # self.gps_data_folder = "/home/lisa/datasets/2011_09_26_drive_0093_sync/2011_09_26/2011_09_26_drive_0093_sync/oxts/"
        # self.map_data_folder = "/home/lisa/datasets/2011_09_26_drive_0093_sync/2011_09_26/2011_09_26_drive_0093_sync/map/mapbox_images_pcd"

        # kitti 71
        # self.lidar_data_folder = "/home/lisa/datasets/2011_09_29_drive_0071_sync/2011_09_29/2011_09_29_drive_0071_sync/velodyne_points/data_pcd/"
        # self.gps_data_folder = "/home/lisa/datasets/2011_09_29_drive_0071_sync/2011_09_29/2011_09_29_drive_0071_sync/oxts/"
        # self.map_data_folder = "/home/lisa/datasets/2011_09_29_drive_0071_sync/2011_09_29/2011_09_29_drive_0071_sync/map/mapbox_images_pcd"

        # kitti 20
        # self.lidar_data_folder = "/home/lisa/datasets/2011_09_30_drive_0020_sync/2011_09_30/2011_09_30_drive_0020_sync/velodyne_points/data_pcd/"
        # self.gps_data_folder = "/home/lisa/datasets/2011_09_30_drive_0020_sync/2011_09_30/2011_09_30_drive_0020_sync/oxts/"
        # self.map_data_folder = "/home/lisa/datasets/2011_09_30_drive_0020_sync/2011_09_30/2011_09_30_drive_0020_sync/map/mapbox_images_pcd"

        # dai
        # self.lidar_data_folder = "/home/lisa/datasets/dai1/velodyne_points/data_pcd/"
        # self.gps_data_folder = "/home/lisa/datasets/dai1/oxts/"
        # self.map_data_folder = "/home/lisa/datasets/dai1/map/mapbox_images_pcd"
        # self.dai = True

        self.logger.info(f"Lidar topic: {self.lidar_topic}")
        self.logger.info(f"Lidar data folder: {self.lidar_data_folder}")
        self.logger.info(f"Lidar buffer size: {self.lidar_buffer_size}")
        
        self.lidar_filenames = os.listdir(self.lidar_data_folder)
        self.lidar_filenames.sort(key=utils.file.numerical_sort_key)

        self.total_lidar_frames = len(self.lidar_filenames)

        self.gps_buffer = []

        self.gps_initialised = False

        self.current_transform = np.identity(4)

        self.lidar_publisher = self.create_publisher(PointCloud2, 'lidar', 10)
        self.non_ground_publisher = self.create_publisher(PointCloud2, 'non_ground', 10)
        self.non_ground_filtered_publisher = self.create_publisher(PointCloud2, 'non_ground_filtered', 10)
        self.map_publisher = self.create_publisher(PointCloud2, 'map', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.mgrs_tf_broadcaster = TransformBroadcaster(self)

        params = pypatchworkpp.Parameters()
        params.verbose = False

        self.PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

        self.lidar_frames_buffer = []

        self.T_world_lidar = np.identity(4)

        self.main()

    def _load_lidar_files(self, window_index):
        if (len(self.lidar_frames_buffer) == 0):
            for index in range(0, window_index+1):
                cloud = PointCloud.from_path(os.path.join(self.lidar_data_folder + self.lidar_filenames[index])).numpy()
                self.lidar_frames_buffer.append(cloud)
                self.logger.info(f"Loaded {self.lidar_filenames[index]}")
        else:
            self.lidar_frames_buffer.pop(0)
            cloud = PointCloud.from_path(os.path.join(self.lidar_data_folder + self.lidar_filenames[window_index])).numpy()
            self.lidar_frames_buffer.append(cloud)
            self.logger.info(f"Loaded {self.lidar_filenames[window_index]}")

    def _scan_to_scan_matching(self):
        concatenated_clouds = None
        ground_clouds = None
        non_ground_clouds = None
        odometry = utils.registration.ScanToScanMatchingOdometry(16)

        inner_lidar_frames = self.lidar_frames_buffer[::-1]

        for i in range(0, len(inner_lidar_frames)):
        # for index, value in enumerate(indices):
            T = odometry.estimate(inner_lidar_frames[i][:,:3])
            inner_cloud_pcd = o3d.geometry.PointCloud()
            inner_cloud_pcd.points = o3d.utility.Vector3dVector(inner_lidar_frames[i][:,:3])
            inner_cloud_pcd.transform(T)

            inner_cloud_pcd = np.asarray(inner_cloud_pcd.points)
            intensities = inner_lidar_frames[i][:,3].reshape(-1, 1)
            inner_cloud_pcd = np.concatenate((inner_cloud_pcd, intensities), axis=1)

            self.PatchworkPLUSPLUS.estimateGround(inner_cloud_pcd)
            ground_idx = self.PatchworkPLUSPLUS.getGroundIndices()
            non_ground_idx = self.PatchworkPLUSPLUS.getNongroundIndices()

            # print(f"ground_idx: {len(ground_idx)}")
            # print(f"non_ground_idx: {len(non_ground_idx)}")

            ground = inner_cloud_pcd[ground_idx]

            non_ground = inner_cloud_pcd[non_ground_idx]
            non_ground = non_ground[non_ground[:, 2] > 0.75]
            non_ground[:, 2] = 0

            if concatenated_clouds is None:
                concatenated_clouds = inner_cloud_pcd
                ground_clouds = ground
                non_ground_clouds = non_ground
            else:
                concatenated_clouds = np.concatenate((concatenated_clouds, inner_cloud_pcd), axis=0)
                ground_clouds = np.concatenate((ground_clouds, ground), axis=0)
                non_ground_clouds = np.concatenate((non_ground_clouds, non_ground), axis=0)

        T_first_second  = odometry.T_fist_to_second

        return concatenated_clouds, ground_clouds, non_ground_clouds, T_first_second
    
    def _publish_concatenated_clouds(self, concatenated_clouds):
        concatenated_clouds_o3d = o3d.geometry.PointCloud()
        concatenated_clouds_o3d.points = o3d.utility.Vector3dVector(concatenated_clouds)
        concatenated_clouds_o3d = concatenated_clouds_o3d.voxel_down_sample(voxel_size=0.07)

        concatenated_clouds = np.asarray(concatenated_clouds_o3d.points)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "lidar"

        lidar_msg = pc2.create_cloud_xyz32(header, concatenated_clouds[:, :3])
        self.lidar_publisher.publish(lidar_msg)

    def _publish_non_ground_clouds(self, non_ground_clouds):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "lidar"

        non_ground_msg = pc2.create_cloud_xyz32(header, non_ground_clouds)
        self.non_ground_publisher.publish(non_ground_msg)

    def _publish_non_ground_filtered_clouds(self, non_ground_clouds):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "lidar"

        non_ground_msg = pc2.create_cloud_xyz32(header, non_ground_clouds)
        self.non_ground_filtered_publisher.publish(non_ground_msg)

    def _publish_map(self, pcd_map):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        map_msg = pc2.create_cloud_xyz32(header, pcd_map)
        self.map_publisher.publish(map_msg)

    def _read_gps_data(self, window_index):
        if len(self.gps_buffer) == 0:
            with open(os.path.join(self.gps_data_folder, self.gps_file), 'r') as f:
                self.gps_buffer = f.readlines()
        current_gps = self.gps_buffer[window_index].split(" ")
        lat = float(current_gps[0])
        lon = float(current_gps[1])
        rotation = float(current_gps[2])

        mgrs_x, mgrs_y = utils.map.get_mgrs_from_lat_long(lat, lon)

        return lat, lon, rotation, mgrs_x, mgrs_y

    def _load_map(self, lat, long):
        x, y = utils.map.deg2num(lat, long, 21)

        x = int(int(x)/10)*10
        y = int(int(y)/10)*10

        if self.dai == True:
            y += 2

        mgrs_x, mgrs_y = utils.map.get_mgrs_from_lat_long(lat, long)
        
        pcd_map = utils.map.load_map_pcd1(x, y, 21, self.map_data_folder)
        if self.dai == True:
            pcd_map = pcd_map[np.logical_and(pcd_map[:,0] > mgrs_x - 150, pcd_map[:,0] < mgrs_x + 150)]
            pcd_map = pcd_map[np.logical_and(pcd_map[:,1] > mgrs_y - 150, pcd_map[:,1] < mgrs_y + 150)]
        else:
            pcd_map = pcd_map[np.logical_and(pcd_map[:,0] > mgrs_x - 100, pcd_map[:,0] < mgrs_x + 100)]
            pcd_map = pcd_map[np.logical_and(pcd_map[:,1] > mgrs_y - 100, pcd_map[:,1] < mgrs_y + 100)]

        pcd_map_o3d = o3d.geometry.PointCloud()
        pcd_map_o3d.points = o3d.utility.Vector3dVector(pcd_map)
        # pcd_map_o3d = pcd_map_o3d.voxel_down_sample(voxel_size=0.5)

        # print(f"map_shape: {pcd_map.shape}")

        pcd_map = np.asarray(pcd_map_o3d.points)
        
        return pcd_map
    
    def _process_concatenated_clouds(self, concatenated_clouds, mgrs_x, mgrs_y, rotation):
        
        concatenated_clouds_o3d = o3d.geometry.PointCloud()
        concatenated_clouds_o3d.points = o3d.utility.Vector3dVector(concatenated_clouds[:,:3])
        # concatenated_clouds_o3d = concatenated_clouds_o3d.voxel_down_sample(voxel_size=0.2)
        # concatenated_clouds_o3d = concatenated_clouds_o3d.translate(np.array([mgrs_x, mgrs_y, 0]))
        # rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, rotation))
        # concatenated_clouds_o3d = concatenated_clouds_o3d.rotate(rotation_matrix, center=(mgrs_x, mgrs_y, 0))

        # concatenated_clouds = np.concatenate((np.asarray(concatenated_clouds_o3d.points), concatenated_clouds[:,3].reshape(-1, 1)), axis=1)

        return np.asarray(concatenated_clouds_o3d.points)
    
    def _transform_pointcloud(self, pointcloud, T):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.transform(T)

        return np.asarray(pcd.points)
    
    def _filter_pointcloud(self, pointcloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)

        labels = pcd.cluster_dbscan(eps=1, min_points=5, print_progress=False)

        labels_np = np.asarray(labels)

        clusterd_points = None

        for label in np.unique(labels_np):
            indices = np.where(labels_np == label)

            if label == -1:
                continue

            cluster_points = pointcloud[labels_np == label]

            if cluster_points.shape[0] < 5:
                continue

            hull = ConvexHull(cluster_points[:, :2])
            # print(hull.equations)
            area = hull.volume
            perimeter = hull.area
            circularity = 4 * np.pi * area / perimeter**2

            

            # if circularity < 0.4:
            #     if clusterd_points is None:
            #         clusterd_points = cluster_points
                           
            #     else:
            #         clusterd_points = np.concatenate((clusterd_points, cluster_points), axis=0)

            if circularity < 0.7:
                if clusterd_points is None:
                    clusterd_points = cluster_points
                           
                else:
                    clusterd_points = np.concatenate((clusterd_points, cluster_points), axis=0)

        pcd.points = o3d.utility.Vector3dVector(clusterd_points)
        pcd.voxel_down_sample(voxel_size=1.0)
        cluster_points = np.asarray(pcd.points)
        return clusterd_points
                    

    def _publish_transform(self, T, frame_id="lidar", broadcaster=None):
        if broadcaster is None:
            broadcaster = self.tf_broadcaster
        transform_msg = TransformStamped()
        transform_msg.header.stamp = self.get_clock().now().to_msg()
        transform_msg.header.frame_id = 'map'
        transform_msg.child_frame_id = frame_id
        transform_msg.transform.translation.x = T[0, 3]
        transform_msg.transform.translation.y = T[1, 3]
        transform_msg.transform.translation.z = T[2, 3]

        r = R.from_matrix(T[:3, :3])

        transform_msg.transform.rotation.x = r.as_quat()[0]
        transform_msg.transform.rotation.y = r.as_quat()[1]
        transform_msg.transform.rotation.z = r.as_quat()[2]
        transform_msg.transform.rotation.w = r.as_quat()[3]

        broadcaster.sendTransform(transform_msg)

    def main(self):
        for window_index in range(self.lidar_buffer_size - 1, self.total_lidar_frames):
            # if window_index - self.lidar_buffer_size >= 2:
            #     break
            self._load_lidar_files(window_index)
            concatenated_clouds, ground_clouds, non_ground_clouds, T_first_second = self._scan_to_scan_matching()

            lat, long, rotation, mgrs_x, mgrs_y = self._read_gps_data(window_index)
            pcd_map = self._load_map(lat, long)

            # print(f"map_shape: {pcd_map.shape}")

            r = R.from_euler('xyz', [0, 0, rotation], degrees=False).as_matrix()
            T_mgrs = np.identity(4)
            T_mgrs[:3, :3] = r
            T_mgrs[0, 3] = mgrs_x
            T_mgrs[1, 3] = mgrs_y

            T_mgrs_start = np.copy(T_mgrs)
            # T_mgrs_start[0, 3] -= 2
            # T_mgrs_start[1, 3] -= 2

            if not self.gps_initialised:
                self.T_world_lidar = self.T_world_lidar @ T_mgrs_start
                self.gps_initialised = True            
            else:
                self.T_world_lidar = self.T_world_lidar @ np.linalg.inv(T_first_second)
                r = R.from_matrix(self.T_world_lidar[:3, :3]).as_euler('xyz', degrees=False)
                r[0] = 0
                r[1] = 0
                r = R.from_euler('xyz', r).as_matrix()
                self.T_world_lidar[:3, :3] = r
                self.T_world_lidar[2, 3] = 0
                # pass

            concatenated_clouds = self._process_concatenated_clouds(concatenated_clouds, mgrs_x, mgrs_y, rotation)
            ground_clouds = self._process_concatenated_clouds(ground_clouds, mgrs_x, mgrs_y, rotation)
            non_ground_clouds = self._process_concatenated_clouds(non_ground_clouds, mgrs_x, mgrs_y, rotation)
            
            non_ground_filtered = self._filter_pointcloud(non_ground_clouds)
            print(f"non_ground_translated filtered: {non_ground_filtered.shape}")
            # non_ground_translated = self._transform_pointcloud(non_ground_filtered, self.T_world_lidar)
            # print(f"non_ground_translated: {non_ground_translated.shape}")      

            # if self.dai == True:
            #     non_ground_clouds = non_ground_filtered

            result = None
            for _ in range(0, 5):
                # if self.dai == True:
                result = small_gicp.align(pcd_map, non_ground_filtered, registration_type="GICP",  init_T_target_source= self.T_world_lidar , max_correspondence_distance=1, max_iterations=1000)
                # else:
                # result = small_gicp.align(pcd_map, non_ground_clouds, registration_type="ICP",  init_T_target_source= self.T_world_lidar , max_correspondence_distance=1, max_iterations=1000)
                 

                self.T_world_lidar = np.copy(result.T_target_source) 

            # result = small_gicp.align(pcd_map, non_ground_filtered, registration_type="GICP",  init_T_target_source= self.T_world_lidar , max_correspondence_distance=1, max_iterations=1000)
            # self.T_world_lidar = np.copy(result.T_target_source) 

            print(f"result: {result.T_target_source}")
            print(f"mgrs: {mgrs_x}, {mgrs_y}")
            print(f"last to current: {np.linalg.inv(self.T_world_lidar) @ result.T_target_source}")  
            # pcd_map_o3d = o3d.geometry.PointCloud()
            # pcd_map_o3d.points = o3d.utility.Vector3dVector(pcd_map)

            # translated_pcd_o3d = o3d.geometry.PointCloud()
            # translated_pcd_o3d.points = o3d.utility.Vector3dVector(non_ground_translated)

            # o3d.io.write_point_cloud(f"/home/lisa/datasets/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/velodyne_points/pcd_map/{window_index}.pcd", pcd_map_o3d)
            # o3d.io.write_point_cloud(f"/home/lisa/datasets/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/velodyne_points/translated_pcd/{window_index}.pcd", translated_pcd_o3d)

            # pcd_map_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # translated_pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # reg_result = o3d.pipelines.registration.registration_icp(pcd_map_o3d, translated_pcd_o3d, 0.1, init=self.T_world_lidar,
            #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

            # self.T_world_lidar = self.T_world_lidar @ np.linalg.inv(result.T_target_source)
            # print(f"last to current: {np.linalg.inv(self.T_world_lidar) @ result.T_target_source}")
            # self.T_world_lidar = np.copy(reg_result.transformation)
            # self.T_world_lidar = self.T_world_lidar @ np.linalg.inv(reg_result.transformation)
            # self.T_world_lidar = self.T_world_lidar @ reg_result.transformation

            # print(result.T_target_source)
            # print(reg_result.transformation)

            self._publish_transform(self.T_world_lidar)
            self._publish_transform(T_mgrs, frame_id="gps", broadcaster=self.mgrs_tf_broadcaster)

            if self.non_ground_filtered_publisher.get_subscription_count() > 0:
                self._publish_non_ground_filtered_clouds(non_ground_filtered[:,:3])

            if self.lidar_publisher.get_subscription_count() > 0:
                self._publish_concatenated_clouds(concatenated_clouds[:,:3])  

            if self.map_publisher.get_subscription_count() > 0:
                self._publish_map(pcd_map)

            if self.non_ground_publisher.get_subscription_count() > 0:
                self._publish_non_ground_clouds(non_ground_clouds[:,:3])
            

def main(args=None):
    rclpy.init(args=args)
    lisa_loc = Localisation()
    try:
        rclpy.spin(lisa_loc)
    finally:
        lisa_loc.destroy_node()
        rclpy.shutdown()    

if __name__ == '__main__':
    main()
