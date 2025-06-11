#!/bin/bash

echo
echo "==================================="
echo " # package download"
echo "==================================="

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654


echo
echo "==================================="
echo " # install ros-melodic version"
echo "==================================="

sudo apt update
sudo apt install ros-melodic-desktop


echo
echo "==================================="
echo " # python venv create"
echo "==================================="

sudo python -m vev .ros-venv

source .ros-venv/bin/activate


echo
echo "==================================="
echo " # rosdep pacakge install"
echo "==================================="

sudo pip install –U rosdep

rosdep update

source /opt/ros/melodic/setup.bash


echo
echo "==================================="
echo " # ros env variable into bash session"
echo "==================================="

echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc 


echo
echo "==================================="
echo " # pacakge install for ros package build"
echo "==================================="

sudo apt-get install camke python-catkin-pkg python-empy python-nose python-setuptools libgtest-dev python-rosinstall python-rosinstall- generator python-wstool build-essential git 


echo
echo "==================================="
echo " # intialization for ros environment"
echo "==================================="

mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make

source devel/setup.sh
echo $ROS_PACKAGE_PATH

echo 'source /opt/ros/melodic/setup.bash' >> ~/.bashrc
echo 'source ~/catkin_ws/devel/setup.bash' >> ~/.bashrc


echo
echo "==================================="
echo " # ---------- roscore execution ----------"
echo "==================================="

roscore


echo
echo "==================================="
echo " # ---------- ros installation done. ----------"
echo "==================================="
