# Get datasets from universe.roboflow.com
# This script is using the GPU to run smoothly.
# For that is necessary CUDA 12.1 and CUDNN 8.9.7.29
# The necessary PyTorch library can be looked on https://pytorch.org/ or by running:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import depthai as dai


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


class CameraGenObjDet(CameraConfig):
    # Constructor
    def __init__(self, width: int, height: int, fx: float, fy: float) -> None:
        super().__init__(width, height, fx, fy)

        self.pipeline = None
        self.device = None
        self.config = None
        self.calculationAlgorithm = None
        self.topLeft = None
        self.bottomRight = None

    # Init rgb
    def init_rgb_depth(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()

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

        # Define sources and outputs
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)

        xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xoutSpatialData = self.pipeline.create(dai.node.XLinkOut)
        xinSpatialCalcConfig = self.pipeline.create(dai.node.XLinkIn)

        xoutDepth.setStreamName("depth")
        xoutSpatialData.setStreamName("spatialData")
        xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setCamera("left")
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setCamera("right")

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        # Config
        self.topLeft = dai.Point2f(0.4, 0.4)
        self.bottomRight = dai.Point2f(0.6, 0.6)

        self.config = dai.SpatialLocationCalculatorConfigData()
        self.config.depthThresholds.lowerThreshold = 100
        self.config.depthThresholds.upperThreshold = 10000
        self.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MIN
        self.config.roi = dai.Rect(self.topLeft, self.bottomRight)

        spatialLocationCalculator.inputConfig.setWaitForMessage(False)
        spatialLocationCalculator.initialConfig.addROI(self.config)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
        stereo.depth.link(spatialLocationCalculator.inputDepth)

        spatialLocationCalculator.out.link(xoutSpatialData.input)
        xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

        print('Depth Camera initialized')
        return

    # Object detector showing camera
    def obj_detect(self, model, labels: list[str], color: tuple[int, int, int]):
        with dai.Device(self.pipeline) as self.device:
            print('Camera working')

            q = self.device.getOutputQueue(name="out", maxSize=4, blocking=False)

            # Output queue will be used to get the depth frames from the outputs defined above
            depthQueue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            spatialCalcQueue = self.device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
            spatialCalcConfigInQueue = self.device.getInputQueue("spatialCalcConfig")

            while self.device.isPipelineRunning():
                inMessage = q.get()
                inColor = inMessage["rgb"]
                frame = inColor.getCvFrame()
                offset_text = 3
                axis_length = 50
                results = model(frame, stream=True)
                detected = {}

                # Camera coordinate system drawing
                fontType = cv2.FONT_HERSHEY_TRIPLEX
                cx, cy = int(self.c[0]), int(self.c[1])
                cv2.line(frame, (cx, cy), (cx + axis_length, cy), (0, 0, 255), 2)  # X-axis (red)
                cv2.putText(frame, "H", (cx + axis_length, cy), fontType, 0.5, color=(0, 0, 255))
                cv2.line(frame, (cx, cy), (cx, cy - axis_length), (0, 255, 0), 2)  # Y-axis (green)
                cv2.putText(frame, "V", (cx, cy - axis_length), fontType, 0.5, color=(0, 255, 0))

                for r in results:
                    boxes = r.boxes

                    for box in boxes:
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w = x2 - x1
                        h = y2 - y1
                        conf = math.ceil((box.conf[0] * 100)) / 100 # Confidence
                        cls = int(box.cls[0])   # Class Name
                        current_class = labels[cls]

                        inDepth = depthQueue.get()  # Blocking call, will wait until a new data has arrived
                        depthFrame = inDepth.getFrame()  # depthFrame values are in millimeters (for color map)
                        # spatialData = spatialCalcQueue.get().getSpatialLocations()

                        # Setting the location for the ROI
                        cut = 0.3  # Area reduction to a more centered position of the object detected
                        x1_ROI, y1_ROI = int(x1 + cut*w), int(y1 + cut*h)
                        x2_ROI, y2_ROI = int(x2 - cut*w), int(y2 - cut*h)
                        self.topLeft = dai.Point2f(x1_ROI / self.resolution[0], y1_ROI / self.resolution[1])
                        self.bottomRight = dai.Point2f(x2_ROI / self.resolution[0], y2_ROI / self.resolution[1])
                        self.config.roi = dai.Rect(self.topLeft, self.bottomRight)
                        self.config.calculationAlgorithm = self.calculationAlgorithm
                        cfg = dai.SpatialLocationCalculatorConfig()
                        cfg.addROI(self.config)
                        spatialCalcConfigInQueue.send(cfg)

                        spatialData = spatialCalcQueue.get().getSpatialLocations()

                        for depthData in spatialData:
                            # Listing the objects detected
                            if f'{current_class}' not in detected:
                                detected[f'{current_class}'] = []

                            # Values for x, y and z are in millimeters
                            x = int(depthData.spatialCoordinates.x)
                            y = int(depthData.spatialCoordinates.y)
                            z = int(depthData.spatialCoordinates.z)

                            print(f'Detecting: {current_class} in X: {x} mm; Y: {y} mm; Z: {z} mm')

                            # Box drawing and label text for every object detected
                            cvzone.putTextRect(frame, f'{labels[cls]} {conf}',
                                               (x1 + offset_text, y1 - offset_text), scale=1, thickness=1, colorB=color,
                                               colorT=(0, 0, 0), colorR=color, offset=5)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                            # Box drawing of the ROI for every object detected
                            cvzone.putTextRect(frame, "ROI", (x1_ROI + offset_text, y1_ROI - offset_text), scale=1,
                                               thickness=1, colorB=(255, 255, 255), colorT=(0, 0, 0),
                                               colorR=(255, 255, 255), offset=5)
                            cv2.putText(frame, f'X: {x} mm', (x1_ROI + 10, y1_ROI + 20), fontType, 0.5,
                                        color=(255, 255, 255))
                            cv2.putText(frame, f'Y: {y} mm', (x1_ROI + 10, y1_ROI + 35), fontType, 0.5,
                                        color=(255, 255, 255))
                            cv2.putText(frame, f'Z: {z} mm', (x1_ROI + 10, y1_ROI + 50), fontType, 0.5,
                                        color=(255, 255, 255))
                            cv2.rectangle(frame, (x1_ROI, y1_ROI), (x2_ROI, y2_ROI), (255, 255, 255), 3)

                            # Listing detections values in meters
                            detected[f'{current_class}'].append([x/1000, y/1000, z/1000, 1])

                cv2.imshow("OAK-D-Lite", frame)
                key = cv2.waitKey(1)

                if key == ord("q"):
                    cv2.imwrite(f'Results_{time.time()}.jpg', frame)
                    cv2.destroyAllWindows()
                    break

        return detected


# Object detector showing image
def obj_detect_img(img, model, labels: list[str], color: tuple[int, int, int]):
    while True:
        offset_text = 3
        results = model(img, show=False)
        img = cv2.imread(img)
        detected = {}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100 # Confidence
                cls = int(box.cls[0])   # Class Name
                current_class = labels[cls]

                # Listing the objects detected
                if f'{current_class}' not in detected:
                    detected[f'{current_class}'] = []

                detected[f'{current_class}'].append([(x1, y1), (x2, y2)])
                print(f'Detecting: {conf} {current_class} in (sup, inf) = (({x1}, {y1}), ({x2}, {y2}))')

                cvzone.putTextRect(img, f'{labels[cls]} {conf}',
                                   (x1 + offset_text, y1 - offset_text), scale=1, thickness=1, colorB=color,
                                   colorT=(0, 0, 0), colorR=color, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        cv2.imshow("Image", img)
        key = cv2.waitKey(0)


        if key == ord("q"):
            cv2.imwrite(f'Results_{time.time()}.jpg', img)
            cv2.destroyAllWindows()
            return detected


# main
def main():
    model = YOLO("Yolo-Weights/yolov8l_PROPUESTA_I.pt")  # YOLO model location
    objects = ['Carton', 'Envases', 'Vidrio']  # Labels for every object to detect in "PROPUESTA I"
    # objects = ['Carton', 'Latas', 'Plastico', 'Vidrio']   # Labels for every object to detect in "PROPUESTA II"
    color = (0, 255, 0)
    img = "Image1.jpg"  # For Image
    camera = CameraGenObjDet(width=1920, height=1080, fx=1498.367322, fy=1497.377563)
    camera.init_rgb_depth()
    camera.obj_detect(model, objects, color)
    # obj_detect_img(img, model, objects, color)


if __name__ == "__main__":
    main()
