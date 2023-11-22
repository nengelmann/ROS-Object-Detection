#!/usr/bin/env python3

import rclpy
from cv_bridge import CvBridge
from foxglove_msgs.msg import (
    Color,
    ImageAnnotations,
    Point2,
    PointsAnnotation,
    TextAnnotation,
)
from rclpy.node import Node
from sensor_msgs.msg import Image

from .model import YOLOv3


class ObjectDetectionNode(Node):
    """
    Node for object detection using YOLOv3.
    """
    def __init__(self, min_bbox_area=400, min_bbox_side=25, debug=False):
        super().__init__("objectdetection_node")
        self.publisher_ = self.create_publisher(
            ImageAnnotations, "/d455_1_rgb_image/detected_objects", 10
        )
        self.subscription = self.create_subscription(
            Image, "/d455_1_rgb_image", self.listener_callback, 10
        )
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.debug = False

        self.model = YOLOv3(
            min_bbox_area=400, min_bbox_side=20, debug=self.debug
        )

        print("Object detection node started and active.")

    def listener_callback(self, msg):
        """Process image and publish detected objects."""
        # Convert ROS Image message to OpenCV image
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        timestamp = msg.header.stamp

        # Process image with YOLOv3
        boxes, confidences, class_ids, class_names = self.model.detect(img)
        if self.debug:
            img_show = self.model.overlay(img, boxes, confidences, class_ids)
            self.model.show(img_show)

        img_annotations = ImageAnnotations()

        if len(boxes) > 0:
            # Publish detected objects
            print(f"Detected objects: {class_names}")

            for box, conf, class_name, class_id in zip(
                boxes, confidences, class_names, class_ids
            ):
                (x, y) = (box[0], box[1])
                (w, h) = (box[2], box[3])
                r, g, b = self.model.colors[class_id]
                r, g, b = r / 255.0, g / 255.0, b / 255.0

                img_annotations.points.append(
                    PointsAnnotation(
                        timestamp=timestamp,
                        type=3,  # LINE_STRIP: 3
                        points=[
                            Point2(x=float(x), y=float(y)),
                            Point2(x=float(x + w), y=float(y)),
                            Point2(x=float(x + w), y=float(y + h)),
                            Point2(x=float(x), y=float(y + h)),
                            Point2(x=float(x), y=float(y)),
                        ],
                        outline_color=Color(a=1.0, b=b, g=g, r=r),
                        fill_color=Color(a=0.1, b=b, g=g, r=r),
                        thickness=4.0,
                    )
                )

                img_annotations.texts.append(
                    TextAnnotation(
                        timestamp=timestamp,
                        text=f"{class_name}: {conf:.2f}",
                        font_size=35.0,
                        position=Point2(x=float(x), y=float(y)),
                        text_color=Color(a=1.0, b=b, g=g, r=r),
                        background_color=Color(a=0.0, b=b, g=g, r=r),
                    )
                )

        else:
            # dummy annotation to keep image annotations in sync with image
            img_annotations.texts.append(
                TextAnnotation(
                    timestamp=timestamp,
                    text="dummy_msg",
                    font_size=1.0,
                    position=Point2(x=10.0, y=10.0),
                    text_color=Color(a=0.0, b=0.0, g=0.0, r=0.0),
                )
            )

        self.publisher_.publish(img_annotations)


def main():
    rclpy.init()
    node = ObjectDetectionNode(
        min_bbox_area=400, min_bbox_side=25, debug=False
    )
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
