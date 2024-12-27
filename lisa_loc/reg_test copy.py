import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import small_gicp

map_folder = "/home/lisa/datasets/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/velodyne_points/pcd_map/"
pcd_folder = "/home/lisa/datasets/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/velodyne_points/translated_pcd/"

pcd_map = o3d.io.read_point_cloud(f"{map_folder}20.pcd")
# pcd_map.voxel_down_sample(voxel_size=1)
translated_pcd = o3d.io.read_point_cloud(f"{pcd_folder}20.pcd")

# pcd_map_points = np.asarray(pcd_map.points) 
# translated_pcd_points = np.asarray(translated_pcd.points)
# print(translated_pcd_points.shape)

# labels = translated_pcd.cluster_dbscan(eps=2, min_points=10, print_progress=True)

# labels_np = np.asarray(labels)

# # print(np.unique(labels_np))

# colors = np.ones_like(translated_pcd_points)

# filtered_labels = []

# clusterd_points = None
# colored_points = None

# for label in np.unique(labels_np):
#     indices = np.where(labels_np == label)
#     colors[indices] = np.random.rand(3)

#     if label == -1:
#         continue

#     cluster_points = translated_pcd_points[labels_np == label]
#     hull = ConvexHull(cluster_points[:, :2])
#     # print(hull.equations)
#     area = hull.volume
#     perimeter = hull.area
#     circularity = 4 * np.pi * area / perimeter**2

    

#     if circularity < 0.5:
#         print(circularity)
#         if clusterd_points is None:
#             clusterd_points = cluster_points
#             colored_points = colors[indices]          
#         else:
#             clusterd_points = np.concatenate((clusterd_points, cluster_points), axis=0)
#             colored_points = np.concatenate((colored_points, colors[indices]), axis=0)

    
    

# max_label = labels.max().item()
# colors = plt.get_cmap("tab20")(
#         labels.numpy() / (max_label if max_label > 0 else 1))
# colors = o3d.core.Tensor(colors[:, :3], o3d.core.float32)
# colors[labels < 0] = 0
# translated_pcd.colors = colors

# translated_pcd_points = translated_pcd_points[labels_np == np.asarray(filtered_labels)]
# colors = colors[labels_np == np.asarray(filtered_labels)]
# translated_pcd.points = o3d.utility.Vector3dVector(translated_pcd_points)
# translated_pcd.points = o3d.utility.Vector3dVector(clusterd_points)
# translated_pcd.colors = o3d.utility.Vector3dVector(colored_points)
# translated_pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([translated_pcd])

# o3d.io.write_point_cloud(f"{pcd_folder}1_filtered.pcd", translated_pcd)

# clusterd_points_copy = np.copy(clusterd_points)

# for i in range(1, 10):
#     inner_clustered_points = np.copy(clusterd_points_copy)
#     inner_clustered_points[:, 2] = i  / 1.25
#     clusterd_points = np.concatenate((clusterd_points, inner_clustered_points), axis=0)

# pcd_map_points_copy = np.copy(pcd_map_points)

# for i in range(1, 10):
#     inner_pcd_map_points = np.copy(pcd_map_points_copy)
#     inner_pcd_map_points[:, 2] = i / 1.25
#     pcd_map_points = np.concatenate((pcd_map_points, inner_pcd_map_points), axis=0)

# print(pcd_map_points.shape)

# pcd_map.points = o3d.utility.Vector3dVector(pcd_map_points)
# pcd_map = pcd_map.voxel_down_sample(voxel_size=0.5)

# pcd_map_points = np.asarray(pcd_map.points)

# print(pcd_map_points.shape)
# print(clusterd_points.shape)

# translated_pcd.points = o3d.utility.Vector3dVector(clusterd_points)

pcd_map.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
translated_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

pcd_map = pcd_map.voxel_down_sample(voxel_size=0.25)
translated_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
translated_pcd = translated_pcd.voxel_down_sample(voxel_size=0.25)


reg_result = o3d.pipelines.registration.registration_icp(pcd_map, translated_pcd, 1, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
T=reg_result.transformation
print(reg_result.transformation)

reg_result = o3d.pipelines.registration.registration_icp(pcd_map, translated_pcd, 0.1, 
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

print(reg_result.transformation)

pcd_map_points = np.asarray(pcd_map.points)
translated_pcd_points = np.asarray(translated_pcd.points)


result = small_gicp.align(pcd_map_points, translated_pcd_points, registration_type='PLANE_ICP', downsampling_resolution=0.25, num_threads=12, max_correspondence_distance=0.5,max_iterations=1000)

print(result)


# temp_pcd = o3d.geometry.PointCloud()
# temp_pcd.points = o3d.utility.Vector3dVector(np.asarray(translated_pcd.points))
                                                        
                                                        

translated_pcd = translated_pcd.transform(T)
# print(reg_result.transformation)

# print(translated_pcd.get_center())

o3d.visualization.draw_geometries([translated_pcd, pcd_map])

# o3d.io.write_point_cloud(f"{pcd_folder}29_filtered.pcd", translated_pcd)
# o3d.io.write_point_cloud(f"{pcd_folder}29_map.pcd", pcd_map)



#  9.99999416e-01  1.08113609e-03  0.00000000e+00 -2.96651589e+01
# -1.08113609e-03  9.99999416e-01  0.00000000e+00  6.65953638e+01
#  0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
#  0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00