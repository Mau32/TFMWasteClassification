from rtde_control import RTDEControlInterface as RTDEControl

class RobotMotion:
    def __init__(self, robot_ip: str, frequency: float, port: int, initial_pose: list):
        # Initialize RTDE control interface
        self.rtde_c = RTDEControl(robot_ip, frequency, RTDEControl.FLAG_USE_EXT_UR_CAP, port)
        self.initial_pose = initial_pose  # [X, Y, Z, RX, RY, RZ]

    def move_to_initial(self):
        # Move to initial safe position
        print("Moving to initial safe position...")
        self.rtde_c.moveL(self.initial_pose, 0.2, 0.15)

    def descend(self, object_loc: list, safe_height: float, off_height: float):
        # Descend in two steps: above the object, then to contact
        x, y, z, _ = object_loc

        # Step 1: approach above the object
        pose_above = [x, y, z + safe_height, self.initial_pose[3], self.initial_pose[4], self.initial_pose[5]]
        print("Descending - step 1 (safe height):", pose_above)
        self.rtde_c.moveL(pose_above, 0.2, 0.15)

        # Step 2: descend to object's height
        pose_contact = [x, y, z + off_height, self.initial_pose[3], self.initial_pose[4], self.initial_pose[5]]
        print("Descending - step 2 (to object):", pose_contact)
        self.rtde_c.moveL(pose_contact, 0.2, 0.15)

    def ascend(self, object_loc: list, safe_height: float):
        # Ascend in two steps: retract a bit, then return to initial
        x, y, z, _ = object_loc

        # Step 1: lift slightly from current pose
        pose_above = [x, y, z + safe_height, self.initial_pose[3], self.initial_pose[4], self.initial_pose[5]]
        print("Ascending - step 1 (safe height):", pose_above)
        self.rtde_c.moveL(pose_above, 0.2, 0.15)

        # Step 2: return to initial pose
        print("Ascending - step 2 (return to initial):", self.initial_pose)
        self.rtde_c.moveL(self.initial_pose, 0.2, 0.15)

    def shutdown(self):
        # Stop script and release control
        print("Shutting down robot communication.")
        self.rtde_c.stopScript()


# main
def main():
    robot_ip = "192.168.1.102"
    frequency = 500.0
    port = 50003
    init_pose = [-0.130, -0.192, 0.360, 2.074, 2.295, -0.093]  # Pose for the initial position of the robot
    robot = RobotMotion(robot_ip, frequency, port, init_pose)
    robot.move_to_initial()
    print("Robot movement ended")


if __name__ == "__main__":
    main()