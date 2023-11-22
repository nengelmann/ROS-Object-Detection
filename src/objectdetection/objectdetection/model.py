import time
from pathlib import Path

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

np.random.seed(42)


class YOLOv3:
    """
    Yolov3 model for object detection.
    This implementation is based on: https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
    """

    def __init__(
        self,
        img_size=416,
        conf=0.5,
        min_bbox_area=400,
        min_bbox_side=20,
        debug=True,
    ):
        package_name = "objectdetection"
        package_share_directory = get_package_share_directory(package_name)
        weights_path = Path(package_share_directory) / "yolov3.weights"
        config_path = Path(package_share_directory) / "yolov3.cfg"
        classes_path = Path(package_share_directory) / "coco.names"

        # for testing outside of ros2
        # weights_path = Path(__file__).parent.parent / 'model/yolov3.weights'
        # config_path = Path(__file__).parent.parent / 'model/yolov3.cfg'
        # classes_path = Path(__file__).parent.parent / 'model/coco.names'

        # raise error if paths are not valid
        if not weights_path.exists():
            raise FileNotFoundError(f"File not found: {weights_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"File not found: {config_path}")
        if not classes_path.exists():
            raise FileNotFoundError(f"File not found: {classes_path}")

        self.img_size = img_size
        self.conf = conf

        with open(str(classes_path), "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.colors = np.random.randint(
            0, 255, size=(len(self.classes), 3), dtype="uint8"
        )

        self.net = cv2.dnn.readNetFromDarknet(
            str(config_path), str(weights_path)
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Force CPU

        layer_names = self.net.getLayerNames()
        # for CPU version
        self.layer_names = [
            layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
        ]

        # Advanced filter for small bbox areas.
        # Takes model feature maps into account (just use the layers with bbox sizes above min_bbox)
        # 'yolo_106':   106th layer with stride 8, 3 anchor boxes and output size 52x52 is responsible to detect small objects
        #               -> smallest side of bbox detection will be ~ (input size / 52)
        # 'yolo_94': 94th layer with stride 16, 3 anchor boxes and output size 26x26  is responsible to detect medium objects
        #               -> smallest side of detection will be ~ (input size / 26)
        # 'yolo_82': 82nd layer with stride 32, 3 anchor boxes and output size 13x13  is responsible to detect large objects
        #               -> smallest side of detection will be ~ (input size / 13)
        yolo_106_min = int((self.img_size / 52) * 1.5)  # 1.5 safety factor
        yolo_94_min = int((self.img_size / 26) * 1.5)  # 1.5 safety factor
        # always keep yolo_82_min, definitely used for normal and large objects

        self.min_bbox_area = min_bbox_area
        self.min_bbox_side = min_bbox_side

        if min_bbox_side > yolo_106_min:
            # if yolo_106_min is smaller, layer 106 is not needed
            self.layer_names.remove("yolo_106")
        if min_bbox_side > yolo_94_min:
            # if yolo_94_min is smaller, layer 94 is not needed
            self.layer_names.remove("yolo_94")

        self.debug = debug
        if debug:
            cv2.namedWindow("window", cv2.WINDOW_NORMAL)
            cv2.createTrackbar("confidence", "window", 50, 100, self._trackbar)

    def detect(self, img):
        """Preprocess and detect objects in image."""
        blob = self.pre_process(img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)
        outputs = np.vstack(outputs)
        boxes, confidences, class_ids, class_names = self.post_process(
            img, outputs
        )
        return boxes, confidences, class_ids, class_names

    def pre_process(self, img):
        """Preprocess image for object detection."""
        blob = cv2.dnn.blobFromImage(
            img,
            1 / 255.0,
            (self.img_size, self.img_size),
            swapRB=True,
            crop=False,
        )
        return blob

    def post_process(self, img, outputs):
        """Postprocess detected objects."""
        H, W = img.shape[:2]

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            scores = output[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # threshold confidence
            if confidence > self.conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                # threshold bbox size
                if (
                    w * h > self.min_bbox_area
                    and min(w, h) > self.min_bbox_side
                ):
                    p0 = int(x - w // 2), int(y - h // 2)
                    boxes.append([*p0, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # remove boxes by non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf, self.conf - 0.1
        )
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = [boxes[i] for i in indices]
            confidences = [confidences[i] for i in indices]
            class_ids = [class_ids[i] for i in indices]

        class_names = [self.classes[class_id] for class_id in class_ids]

        return boxes, confidences, class_ids, class_names

    def overlay(self, img, boxes, confidences, class_ids):
        """Overlay detected objects on image."""
        if self.debug:
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                (x, y) = (box[0], box[1])
                (w, h) = (box[2], box[3])
                color = [int(c) for c in self.colors[class_id]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.classes[class_id], conf)
                cv2.putText(
                    img,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            return img
        else:
            raise Exception(
                "Debug mode is disabled. Print in of overlay with OpenCV is"
                " limited to debug mode. Enable debug mode with 'debug=True'"
                " argument."
            )

    def show(self, img, wait=1):
        """Show image in opencv window."""
        if self.debug:
            cv2.imshow("window", img)
            cv2.waitKey(wait)
        else:
            raise Exception(
                "Debug mode is disabled. Show an image with OpenCV is limited"
                " to debug mode. Enable debug mode with 'debug=True' argument."
            )

    def get_class(self, class_id):
        """Get class name from class id."""
        return self.classes[class_id]

    def _trackbar(self, val):
        self.conf = val / 100


if __name__ == "__main__":
    yolo = YOLOv3(min_bbox_area=400, min_bbox_side=25, debug=True)

    img_path1 = (
        Path(__file__).parent.parent.parent.parent / "assets/test_img2.jpg"
    )
    img_path2 = (
        Path(__file__).parent.parent.parent.parent / "assets/test_img1.png"
    )

    for img_path in [img_path1, img_path2]:
        img = cv2.imread(str(img_path))
        img_proc = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # convert to RGB due to model processing with RGB
        t0 = time.time()
        boxes, confidences, class_ids, _ = yolo.detect(img_proc)
        print(f"Detection and processing took {time.time()-t0:.2f}s")
        img = yolo.overlay(img, boxes, confidences, class_ids)
        yolo.show(img, wait=0)
