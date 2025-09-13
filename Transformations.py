# Functions to obtain the transformation matrix to locate every object in the space
# Absolute coordinate system located at the robot's base

import numpy as np


def homog(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()  # (3, 1) to (3, ) Conversion to avoid numpy errors caused by the return of pupil_apriltags

    return T

def pos_abs_base(T_cam_base: np.ndarray, obj_det_cam: dict):
    # This operation transforms the relative position obtained by the camera to the base reference
    obj_det_cam = {k: [T_cam_base @ np.array(v_i).reshape(-1, 1) for v_i in v_list] for k, v_list in obj_det_cam.items()}

    # Converting the column vectors obtained before into lists again
    obj_det_cam = {k: [list(v_i.flatten()) for v_i in v_list] for k, v_list in obj_det_cam.items()}

    return obj_det_cam

def rotation_x(T: np.ndarray) -> np.ndarray:
    rot_x = np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])

    T_rot_x = np.dot(T, rot_x)

    return T_rot_x

def adjust(obj_dict):
    """
    This adjustment is a translation of the coordinate system of the camera to compensate the difference between
    the coordinate system of the depthai library (middle of the camera) and the pupil_apriltags library (located 20 mm
    to the right and 40 mm down of the camera's center if watching directly to the lenses)
    """

    adjusted_obj_dict = {}

    for key, positions in obj_dict.items():
        adjusted_positions = []
        for pos in positions:
            x_adj = pos[0] + 0.020
            y_adj = -1 * (pos[1] - 0.040)
            adjusted_positions.append([x_adj, y_adj, pos[2], pos[3]])
        adjusted_obj_dict[key] = adjusted_positions

    return adjusted_obj_dict


def main():
    R_tag_base = np.eye(3)  # Rotation matrix, assuming coordinate systems from base and tag align perfectly
    t_tag_base = np.array([0.0173, -0.3215, 0]) # Apriltag location coordinate system for the robot's base in meters
    T_tag_base = homog(R_tag_base, t_tag_base)
    np.set_printoptions(precision=4, suppress=True)
    print("T_tag_base =\n", T_tag_base)
    T_cam_base = np.eye(4)
    T_cam_base[:3, 3] = np.array([2, 2, 2])
    d = {'hi': [[1, 1, 1, 1],[4, 4, 4, 1], [2, 3, 4, 1]], 'hi1': [[2, 2, 2, 1]], 'hi2': [[3, 3, 3, 1], [6, 6, 6, 1]]}
    e = pos_abs_base(T_cam_base, d)
    print("e =\n", e)


if __name__ == "__main__":
    main()
