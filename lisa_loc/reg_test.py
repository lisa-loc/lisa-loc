import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import small_gicp

map_folder = "/home/lisa/datasets/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/velodyne_points/pcd_map/"
pcd_folder = "/home/lisa/datasets/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/velodyne_points/translated_pcd/"

pcd_map = o3d.io.read_point_cloud(f"{map_folder}20.pcd")
translated_pcd = o3d.io.read_point_cloud(f"{pcd_folder}20.pcd")
translated_pcd = translated_pcd.voxel_down_sample(voxel_size=0.5)

pcd_map_points = np.asarray(pcd_map.points)
translated_pcd_points = np.asarray(translated_pcd.points)

result = small_gicp.align(pcd_map_points, translated_pcd_points, max_correspondence_distance=0.1,  registration_type="VGICP", max_iterations=1000)

print(result)

reg_result = o3d.pipelines.registration.registration_icp(pcd_map, translated_pcd, 0.1, 
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

print(np.asarray(reg_result.correspondence_set))
print(reg_result.fitness)
print(reg_result.inlier_rmse)
print(reg_result.transformation)

# pcd_map.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# translated_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# # pcd_map = pcd_map.voxel_down_sample(voxel_size=0.5)
# # translated_pcd = translated_pcd.voxel_down_sample(voxel_size=0.5)

# translated_pcd = translated_pcd.remove_statistical_outlier(nb_neighbors=6, std_ratio=1.0)[0]

# pre_inlier_rmse = 1000

# for _ in range(10):
#     reg_result = o3d.pipelines.registration.registration_icp(pcd_map, translated_pcd, 0.1, 
#                 estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#                 criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
    
#     print(reg_result.inlier_rmse)

#     print(reg_result.transformation)


#     # pcd_map_points = np.asarray(pcd_map.points)
#     # translated_pcd_points = np.asarray(translated_pcd.points)                          
                                                            
#     if reg_result.inlier_rmse < pre_inlier_rmse:
#         pre_inlier_rmse = reg_result.inlier_rmse
#         # T = reg_result.transformation
#         translated_pcd = translated_pcd.transform(reg_result.transformation)
#         # translated_pcd = translated_pcd.translate([0.2, 0.2, 0])


# # pcd_map_points = np.asarray(pcd_map.points)
# # translated_pcd_points = np.asarray(translated_pcd.points)

# # for _ in range(10):

# #     result = small_gicp.align(pcd_map_points, translated_pcd_points, max_correspondence_distance=0.1,  registration_type="VGICP", max_iterations=1000)

translated_pcd = translated_pcd.transform(result.T_target_source)

o3d.visualization.draw_geometries([translated_pcd, pcd_map])



