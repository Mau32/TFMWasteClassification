# Get datasets from universe.roboflow.com

import time
import cv2
import os
import depthai as dai


# Directories setup
def object_labels():
    dataset_config = ["test", "train", "valid"]
    dataset_name = input("Enter the name of the dataset: ")
    labels = []

    # Labels list creation by the user
    while True:
        element = input("Enter the name of a label or 'end' to finish: ")
        if element.lower() == 'end':
            break

        labels.append(element)
        labels.sort()

    for label in labels:
        path = os.path.join(dataset_name, "CollectedImages", label)
        if not os.path.exists(path):
            os.makedirs(path)  # Creates the database directories

    for dataset_cfg in dataset_config:
        path2 = os.path.join(dataset_name, dataset_cfg, "images")
        path3 = os.path.join(dataset_name, dataset_cfg, "labels")
        if not os.path.exists(path2):
            os.makedirs(path2)  # Creates the dataset directories
        if not os.path.exists(path3):
            os.makedirs(path3)  # Creates the dataset directories

    return dataset_name, labels


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


class CameraDataCol(CameraConfig):
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

    # Image collection
    def img_collection(self, data: str, labels: list[str]):
        for index, label in enumerate(labels):
            counter = 1

            if index == len(labels) - 1:
                print(f'Press "s" to start collecting images for {label} or "q" to end collecting')
            else:
                print(f'Press "s" to start collecting images for {label} or "q" to go to the next one...')

            with dai.Device(self.pipeline) as self.device:
                print('Camera working')

                q = self.device.getOutputQueue(name="out", maxSize=4, blocking=False)

                while self.device.isPipelineRunning():
                    inMessage = q.get()
                    inColor = inMessage["rgb"]
                    frame = inColor.getCvFrame()

                    cv2.imshow("OAK-D-Lite", frame)
                    key = cv2.waitKey(1)

                    if key == ord("s"):
                        cv2.imwrite(f'{data}/CollectedImages/{label}/Image_{label}_{time.time()}.jpg', frame)
                        print(f'Collecting image for {label}, image #{counter}')
                        counter += 1

                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        break

        cv2.destroyAllWindows()
        return

    # Image capturing
    def img_capture(self):

        # success, img = cap.read()
        with dai.Device(self.pipeline) as self.device:
            print('Camera working')

            q = self.device.getOutputQueue(name="out", maxSize=4, blocking=False)

            while self.device.isPipelineRunning():

                inMessage = q.get()
                inColor = inMessage["rgb"]
                frame = inColor.getCvFrame()

                cv2.imshow("OAK-D-Lite", frame)
                key = cv2.waitKey(1)

                if key == ord("s"):
                    img_name = f'Image_{time.time()}.jpg'
                    cv2.imwrite(img_name, frame)
                    print(f'Image {img_name} captured')

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    break

            cv2.destroyAllWindows()
            return


# main
def main():
    dataset_name, labels = object_labels()
    camera = CameraDataCol(width=1920, height=1080, fx=1498.367322, fy=1497.377563)
    camera.init_rgb()
    camera.img_collection(dataset_name, labels)
    print(f'The dataset for {labels} has been created')

    camera.img_capture()


if __name__ == "__main__":
    main()
