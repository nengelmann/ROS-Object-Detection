from setuptools import setup

package_name = "objectdetection"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["./model/coco.names"]),
        ("share/" + package_name, ["./model/yolov3.cfg"]),
        ("share/" + package_name, ["./model/yolov3.weights"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nengelmann",
    maintainer_email="mail@nico-engelmann.de",
    description="ROS2 node for object detection using YOLOv3",
    license="MIT",
    entry_points={
        "console_scripts": [
            "node = objectdetection.node:main",
        ],
    },
)
