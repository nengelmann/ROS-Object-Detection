
# ROS installation
[Source](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt upgrade
# Due to hold back upgrades
sudo apt --with-new-pkgs upgrade gjs libgjs0g
sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools
```
```bash
sudo apt install ros-humble-foxglove-bridge
```

In the git repo create a ROS (python specific) package (single project node) directly in the repo:
```bash
mkdir temp_dir
cd temp_dir
```
`ros2 pkg create --build-type ament_python temp_dir`


I want to write a Python ROS2 node, which uses YOLOv3 to extract & publish objects from a ROS video stream. 