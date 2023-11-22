# ROS-Object-Detection
Extract and publish objects from a ROS video stream.

# Installation

## Prerequisites

- Linux (Ubuntu22.04)
- [Foxglove Studio](https://foxglove.dev/download)
- [ROS2 (Humble)](https://docs.ros.org/en/humble/Installation.html)
- [foxglove_bridge](https://index.ros.org/p/foxglove_bridge/) \
```sudo apt install ros-humble-foxglove-bridge```
- [foxglove_msgs](https://index.ros.org/p/foxglove_msgs/) \
```sudo apt install ros-humble-foxglove-msgs```

'sudo apt install ros-humble-vision-msgs' ?


## Download yolov3 weights

Download yolov3 weights, config and class names from [here](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html) and paste them into `./src/objectdetection/model` via command line.
```bash
curl -o ./src/objectdetection/model/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
curl -o ./src/objectdetection/model/yolov3.cfg https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg
curl -o ./src/objectdetection/model/coco.names https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names
```

## Build
```bash
rm -rf ./build ./install ./log
source /opt/ros/humble/setup.bash
colcon build --packages-select objectdetection
source ./install/setup.bash
```

# Usage

Run object detection node (separate terminal):
```bash
source ./install/setup.bash
ros2 run objectdetection node
```

Run the Foxglove bridge (separate terminal):
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

Run the ros bag (separate terminal):
```bash
ros2 bag play /path/to/data/lounge/
```