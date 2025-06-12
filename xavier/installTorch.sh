
echo
echo "==================================="
echo " # install the dependencies (if not already onboard)"
echo "==================================="

sudo apt install nvidia-jetpack
sudo apt-get install -y python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
sudo pip3 install -U jetson-stats
sudo -H pip3 install future
sudo pip3 install -U --user wheel mock pillow
sudo -H pip3 install testresources


echo
echo "==================================="
echo " # above 58.3.0 you get version issues"
echo "==================================="

sudo -H pip3 install setuptools==58.3.0
sudo -H pip3 install Cython


echo
echo "==================================="
echo " # install gdown to download from Google drive"
echo "==================================="

sudo -H pip3 install gdown


echo
echo "==================================="
echo " # download the wheel"
echo "==================================="

export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl


echo
echo "==================================="
echo " # install PyTorch 2.0.0"
echo "==================================="

python3 -m pip install --upgrade pip

python3 -m pip install numpy==1.24.4

python3 -m pip install --no-cache $TORCH_INSTALL
