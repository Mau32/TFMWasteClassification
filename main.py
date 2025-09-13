# main script for the TFM project

import cv2
import numpy as np
from ultralytics import YOLO
from DataCollection import object_labels, CameraDataCol
from YOLOv8ObjDetector import CameraGenObjDet, obj_detect_img
from ApriltagDetector import CameraAprilTagDet
from Transformations import homog, pos_abs_base, rotation_x, adjust
from RobotControl import RobotMotion


def main():
    while True:
        act = input("Type 'capture', 'dataset', 'detect', or 'exit' for the required command: ").lower()

        # OAK-D-LITE Camera parameters
        width, height = 1920, 1080
        fx, fy = 1498.367322, 1497.377563

        img = "Image1.jpg"   # For Image
        model = YOLO("Yolo-Weights/yolov8l_PROPUESTA_I.pt") # YOLO model location
        objects = ['Carton', 'Envases', 'Vidrio']   # Labels for every object to detect in "PROPUESTA I"
        # objects = ['Carton', 'Latas', 'Plastico', 'Vidrio']   # Labels for every object to detect in "PROPUESTA II"
        color = (0, 255, 0)  # Rectangle color to show BGR

        # Apriltag
        R_tag_base = np.eye(3)  # Rotation matrix, assuming coordinate systems from base and tag align perfectly
        t_tag_base = np.array([-0.0170, -0.320, 0]) # Apriltag location coordinate system for the robot's base in meters
        tag_size = 0.015    # Tag size in meters (actual size of the apriltag)

        # Robot parameters
        robot_ip = "192.168.1.102"
        frequency = 500.0
        port = 50003
        init_pose = [-0.130, -0.192, 0.360, 2.074, 2.295, -0.093]   # Pose for the initial position of the robot
        # init_pose = [-0.011, -0.223, 0.360, 2.074, 2.295, -0.093]  # Pose for the initial position of the robot
        deposit_loc = [0.130, -0.245, 0.050, 1.0]  # Location for the deposition of waste
        safe_height = 0.200
        off_height = 0.000  # Security factor to avoid hitting the base while descending the gripper

        # Capturing images
        if act == 'capture':
            camera = CameraDataCol(width, height, fx, fy)
            camera.init_rgb()
            camera.img_capture()

        # Dataset setup
        if act == 'dataset':
            dataset_name, labels = object_labels()
            camera = CameraDataCol(width, height, fx, fy)
            camera.init_rgb()
            camera.img_collection(dataset_name, labels)
            print(f'The dataset for {labels} has been created')
            # Run "label-studio" on the terminal

        # Object detection
        if act == 'detect':
            img_act = input("Type 'yes' or 'no' if there is an image to analyze: ").lower()

            if img_act == 'yes' and img is not None:
                obj_detect_img(img, model, objects, color)  # Object detection inside an image

            elif img_act == 'no':
                #Initial position
                print("The robot will move to the initial position...")
                robot = RobotMotion(robot_ip, frequency, port, init_pose)
                robot.move_to_initial() # The robot is positioned to start detecting
                print("Robot ready to start detecting objects")

                # Object detection
                camera = CameraGenObjDet(width, height, fx, fy)
                camera.init_rgb_depth()
                obj_det_cam = camera.obj_detect(model, objects, color)
                print(f'Objects detected: {obj_det_cam}')

                # AprilTag detection
                camera = CameraAprilTagDet(width, height, fx, fy)
                camera.init_rgb()
                tag_id, R_tag_cam, t_tag_cam = camera.apriltag_detect(tag_size)
                print(f'AprilTag ID: {tag_id} detected')

                # Rotate 180 degrees over the x-axis to get it properly aligned (library issue)
                T_tag_cam = rotation_x(homog(R_tag_cam, t_tag_cam))

                # Getting the transformation matrix T_cam_base and T_tag_base
                T_cam_tag = np.linalg.inv(T_tag_cam)
                T_tag_base = homog(R_tag_base, t_tag_base)

                # Transformation matrix used to locate the object from base and the actual location of the objects
                T_cam_base = T_tag_base @ T_cam_tag
                np.set_printoptions(precision=4, suppress=True)
                print("T_tag_base =\n", T_tag_base)
                print("T_tag_cam =\n", T_tag_cam)
                print("T_cam_tag =\n", T_cam_tag)
                print("T_cam_base =\n", T_cam_base)
                adjusted_obj_det_cam = adjust(obj_det_cam)
                print(f'Objects detected (adjusted): {adjusted_obj_det_cam}')
                obj_det_base = pos_abs_base(T_cam_base, adjusted_obj_det_cam)
                print(f'Location of every object detected from the base: {obj_det_base}')

                # Robot movement
                while obj_det_base:
                    print(f'The following materials have been found: {list(obj_det_base.keys())}')
                    cls_sel = input("Please, type the desired material, exactly as listed above to be picked and then, "
                                    "press Enter or type 'stop' to stop the classification process: ")

                    if cls_sel.lower() == 'stop':
                        print("Stopping the program for the robot's movement...")
                        robot.shutdown()
                        break

                    if cls_sel not in obj_det_base:
                        print("Try a valid object class or material that was detected")
                        continue

                    for obj in obj_det_base[cls_sel]:
                        robot.descend(obj, safe_height, off_height)
                        robot.shutdown()
                        robot.ascend(obj, safe_height)
                        robot.descend(deposit_loc, safe_height, off_height)
                        robot.shutdown()
                        robot.ascend(deposit_loc, safe_height)
                        robot.shutdown()

                    del obj_det_base[cls_sel]   # This deletes the class once the classification is over


        # Stop main
        if act == 'exit':
            print("Ending the execution of the main program...")
            cv2.destroyAllWindows()
            break

        # Else cases
        elif act not in ['capture', 'dataset', 'detect', 'exit']:
            print("Try a valid command")
            continue


if __name__ == "__main__":
    main()
