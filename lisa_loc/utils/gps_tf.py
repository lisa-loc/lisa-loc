from argparse import ArgumentParser
import os
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("folder_path", type=str, help="Path to the folder")
    args = parser.parse_args()

    gps_buffer = []

    gps_file = os.path.join(args.folder_path, "gps.txt")
    with open(gps_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(" ")
            x = float(line[1])
            y = float(line[2])
            x = x + 0.81
            y = y - 0.32
            line[1] = str(x)
            line[2] = str(y)
            gps_buffer.append(" ".join(line))

    gps_translated = os.path.join(args.folder_path, "gps_translated.txt")
    with open(gps_translated, "w") as f:
        f.writelines(gps_buffer)

            