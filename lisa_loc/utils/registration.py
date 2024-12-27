import small_gicp
import numpy as np

class ScanToScanMatchingOdometry(object):
  def __init__(self, num_threads):
    self.num_threads = num_threads
    self.T_last_current = np.identity(4)
    self.T_world_lidar = np.identity(4)
    self.target = None
    self.T_fist_to_second = None
  
  def estimate(self, raw_points):
    downsampled, tree = small_gicp.preprocess_points(raw_points, 0.25, num_threads=self.num_threads)
    
    if self.target is None:
      self.target = (downsampled, tree)
      return self.T_world_lidar

    result = small_gicp.align(self.target[0], downsampled, self.target[1], self.T_last_current, num_threads=self.num_threads)
    
    self.T_last_current = result.T_target_source
    self.T_world_lidar = self.T_world_lidar @ result.T_target_source
    self.target = (downsampled, tree)

    if self.T_fist_to_second is None:
      self.T_fist_to_second = result.T_target_source
    
    return self.T_world_lidar
  
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
  
class ScanToMapMatchingOdometry(object):
  def __init__(self, num_threads):
    self.num_threads = num_threads
    self.T_last_current = np.identity(4)
    self.T_world_lidar = np.identity(4)
    self.target = None