#!/bin/bash

echo
echo "==================================="
echo " #  reveal the CUDA location"
echo "==================================="

sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
sudo ldconfig

echo
echo "==================================="
echo " # third-party libraries"
echo "==================================="

sudo apt-get install -y build-essential cmake git unzip pkg-config zlib1g-dev
sudo apt-get install -y libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev
sudo apt-get install -y libpng-dev libtiff-dev libglew-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev libcanberra-gtk*
sudo apt-get install -y python-dev python-numpy python-pip
sudo apt-get install -y python3-dev python3-numpy python3-pip
sudo apt-get install -y libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt-get install -y libtbb2 libtbb-dev libdc1394-22-dev libxine2-dev
sudo apt-get install -y gstreamer1.0-tools libgstreamer-plugins-base1.0-dev
sudo apt-get install -y libgstreamer-plugins-good1.0-dev
sudo apt-get install -y libv4l-dev v4l-utils v4l2ucp qv4l2
sudo apt-get install -y libtesseract-dev libxine2-dev libpostproc-dev
sudo apt-get install -y libavresample-dev libvorbis-dev
sudo apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install -y liblapack-dev liblapacke-dev libeigen3-dev gfortran
sudo apt-get install -y libhdf5-dev libprotobuf-dev protobuf-compiler
sudo apt-get install -y libgoogle-glog-dev libgflags-dev

echo
echo "==================================="
echo " # Qt5"
echo "==================================="

sudo apt-get install -y qt5-default

echo
echo "==================================="
echo " # Download OpenCV"
echo "==================================="

cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.6.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.6.0.zip

echo
echo "==================================="
echo " # unpack"
echo "==================================="

unzip opencv.zip
unzip opencv_contrib.zip

echo
echo "==================================="
echo " # some administration to make live easier later on"
echo "==================================="

mv opencv-4.6.0 opencv
mv opencv_contrib-4.6.0 opencv_contrib

echo
echo "==================================="
echo " # clean up the zip files"
echo "==================================="

rm opencv.zip
rm opencv_contrib.zip

echo
echo "==================================="
echo " # Build Make"
echo "==================================="

cd ~/opencv
mkdir build
cd build

echo
echo "==================================="
echo " # cmake"
echo "==================================="

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D WITH_OPENCL=OFF \
-D WITH_CUDA=ON \
-D CUDA_ARCH_BIN=5.3 \
-D CUDA_ARCH_PTX="" \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_NEON=ON \
-D WITH_QT=OFF \
-D WITH_OPENMP=ON \
-D BUILD_TIFF=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D BUILD_TESTS=OFF \
-D WITH_EIGEN=ON \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D BUILD_opencv_python3=TRUE \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF \
-D PYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.8.so \
-D PYTHON_NUMPY_INCLUDE_DIR=/usr/lib/python3/dist-packages/numpy/core/include/numpy/ \
-D PYTHON_PACKAGES_PATH=/usr/lib/python3.8/dist-packages/ ..

echo
echo "==================================="
echo " # Make"
echo "==================================="

make -j4

sudo rm -r /usr/include/opencv4/opencv2
sudo make install
sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig

# cleaning (frees 300 MB)

make clean
sudo apt-get update

sudo rm -rf ~/opencv
sudo rm -rf ~/opencv_contrib

echo " # Installation OpenCV Done"

