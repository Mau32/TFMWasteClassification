# Script for the detection of AprilTags using a OAK-D-LITE camera by Luxonis

import cv2
from pupil_apriltags import Detector
import depthai as dai
import time


# Camera exception handling
class CameraException(Exception):

    def __init__(self, msg: str):
        msg = 'Camera Exception: ' + msg
        super().__init__(msg)


# Camera config
class CameraConfig():
    # Constructor
    def __init__(self, width: int, height: int, fx: float, fy: float) -> None:
        """ Camera config
            Resolution: pixels
            Camera center: pixels
            Focal length: pixels
            """
        self.resolution = [width, height]
        self.f = [fx, fy]
        self.c = [width/2, height/2]


class CameraAprilTagDet(CameraConfig):
    # Constructor
    def __init__(self, width: int, height: int, fx: float, fy: float) -> None:
        super().__init__(width, height, fx, fy)

        self.pipeline = None
        self.device = None

    # Init rgb
    def init_rgb(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()
        # self.device = dai.Device()

        camRgb = self.pipeline.create(dai.node.ColorCamera)

        sync = self.pipeline.create(dai.node.Sync)
        xOut = self.pipeline.create(dai.node.XLinkOut)
        xOut.input.setBlocking(False)

        # Properties RGB CAM
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        if self.resolution[1] == 720:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        elif self.resolution[1] == 1080:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        elif self.resolution[1] == 2160:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        elif self.resolution[1] == 3120:
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)

        camRgb.setFps(35)   # Set rate at which camera should produce frames (cap is 35 fps)

        try:
            calibData = dai.Device().readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lensPosition:
                camRgb.initialControl.setManualFocus(lensPosition)
        except:
            raise CameraException("Lens calibration failure")

        camRgb.isp.link(sync.inputs["rgb"])
        sync.out.link(xOut.input)
        xOut.setStreamName("out")

        print('RGB Camera initialized')
        return

    def apriltag_detect(self, tag_size: float):
        with dai.Device(self.pipeline) as self.device:
            print('Camera working')

            q = self.device.getOutputQueue(name="out", maxSize=4, blocking=False)

            # Initialize AprilTag detector
            detector = Detector(
                families="tag36h11",
                nthreads=4,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=True,
                decode_sharpening=0.25,
                debug=False
            )

            # Camera parameters
            camera_params = (self.f[0], self.f[1], self.c[0], self.c[1])
            counter = 0

            while self.device.isPipelineRunning():
                inMessage = q.get()
                inColor = inMessage["rgb"]
                frame = inColor.getCvFrame()

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale for AprilTag detection
                detections = detector.detect(gray_frame, estimate_tag_pose=True, camera_params=camera_params,
                                             tag_size=tag_size) # AprilTag detection with pose

                for det in detections:
                    corners = det.corners.astype(int)   # Corners describing the square shape
                    tag_id = det.tag_id # Tag ID number
                    t = det.pose_t  # (3, 1) Translation
                    R = det.pose_R  # (3, 3) Rotation

                    print(f'Detecting: AprilTag ID: {tag_id}')
                    print("Detecting translation vector t_tag_cam =\n", t)
                    print("Detecting rotation matrix R_tag_cam =\n", R)

                    # AprilTag drawing
                    for i in range(4):
                        pt1 = tuple(corners[i])
                        pt2 = tuple(corners[(i + 1) % 4])
                        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

                    # Show ID and location on the video
                    center = tuple(map(int, det.center))
                    fontType = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.putText(frame, f"ID {tag_id}", center, fontType, 0.6, (255, 255, 0), 2)

                counter += 1

                cv2.imshow("OAK-D-Lite", frame)
                key = cv2.waitKey(1)

                if key == ord("q") or counter == 500:
                    cv2.imwrite(f'Results_APrTag_{time.time()}.jpg', frame)
                    cv2.destroyAllWindows()
                    break

        return tag_id, R, t


# main
def main():
    tag_size = 0.015
    camera = CameraAprilTagDet(width=1920, height=1080, fx=1498.367322, fy=1497.377563)
    camera.init_rgb()
    camera.apriltag_detect(tag_size)
    tag_id, R_tag_cam, t_tag_cam = camera.apriltag_detect(tag_size)
    print(f'AprilTag ID: {tag_id} detected')


if __name__ == "__main__":
    main()
