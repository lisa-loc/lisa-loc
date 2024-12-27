from argparse import ArgumentParser
import os
from scipy.spatial.transform import Rotation as R
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("folder_path", type=str, help="Path to the folder")
    args = parser.parse_args()

    gps_buffer = []

    # 9.999976e-01 7.553071e-04 -2.035826e-03 -7.854027e-04 9.998898e-01 -1.482298e-02 2.024406e-03 1.482454e-02 9.998881e-01

    # rotation_matrix = np.array([[9.999976e-01, 7.553071e-04, -2.035826e-03],
    #                             [-7.854027e-04, 9.998898e-01, -1.482298e-02],
    #                             [2.024406e-03, 1.482454e-02, 9.998881e-01]])

    rotation_matrix = np.identity(3)

    rotation_matrix_euler = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)
    rotation_matrix_euler[0] = 0.0
    rotation_matrix_euler[1] = 0.0

    rotation_matrix = R.from_euler("xyz", rotation_matrix_euler, degrees=False).as_matrix()
    # print(rotation_matrix)

    T_imu_velo = np.identity(4)
    # T_imu_velo[0, 3] = -8.086759e-01
    # T_imu_velo[1, 3] = 3.195559e-01 
    T_imu_velo[0, 3] = 1.6
    T_imu_velo[0:3, 0:3] = rotation_matrix

    # print(T_imu_velo)
    

    gps_file = os.path.join(args.folder_path, "gps.txt")
    with open(gps_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(" ")
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])

            x_rot = float(line[4])
            y_rot = float(line[5])
            z_rot = float(line[6])
            w_rot = float(line[7])

            rotation_imu = R.from_quat([x_rot, y_rot, z_rot, w_rot]).as_matrix()

            T_imu = np.identity(4)
            T_imu[0, 3] = x
            T_imu[1, 3] = y
            T_imu[2, 3] = z
            T_imu[0:3, 0:3] = rotation_imu

            T_velo = T_imu @ T_imu_velo

            rotation_imu_quat = R.from_matrix(T_velo[0:3, 0:3]).as_quat()
            x = T_velo[0, 3]
            y = T_velo[1, 3]
            z = T_velo[2, 3]

            # x = x + 0.81
            # y = y - 0.32
            line[1] = str(x)
            line[2] = str(y)
            line[3] = str(z)

            line[4] = str(rotation_imu_quat[0])
            line[5] = str(rotation_imu_quat[1])
            line[6] = str(rotation_imu_quat[2])
            line[7] = str(rotation_imu_quat[3])

            # print(line)
            gps_buffer.append(" ".join(line))

    # print(gps_buffer)

    gps_translated = os.path.join(args.folder_path, "gps_translated1.txt")
    with open(gps_translated, "w") as f:
        for line in gps_buffer:
            f.write(line + "\n")

            