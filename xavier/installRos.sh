#!/bin/bash

echo " # ---------- package download ---------- # "
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

echo " # ---------- ros repository public key add ---------- # "
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

echo " # ---------- install ros-melodic version ---------- # "
sudo apt update
sudo apt install ros-melodic-desktop

sudo apt-get install python-pip

echo " # ---------- python venv create ---------- # "
sudo python -m vev .ros-venv

source .ros-venv/bin/activate

echo " # ---------- rosdep pacakge install ---------- # "
sudo pip install –U rosdep

rosdep update

source /opt/ros/melodic/setup.bash

echo " # ---------- ros env variable into bash session ---------- # "
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc 

echo " # ---------- pacakge install for ros package build ---------- # "
sudo apt-get install camke python-catkin-pkg python-empy python-nose python-setuptools libgtest-dev python-rosinstall python-rosinstall- generator python-wstool build-essential git 

echo " # ---------- intialization for ros environment ---------- # "
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make

source devel/setup.sh
echo $ROS_PACKAGE_PATH

echo 'source /opt/ros/melodic/setup.bash' >> ~/.bashrc
echo 'source ~/catkin_ws/devel/setup.bash' >> ~/.bashrc

echo " # ---------- roscore execution ---------- # "
roscore

echo " # ---------- ros installation done. ---------- # "
