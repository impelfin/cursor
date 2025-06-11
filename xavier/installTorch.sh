# install the dependencies (if not already onboard)

sudo apt-get install -y python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
sudo -H pip3 install future
sudo pip3 install -U --user wheel mock pillow
sudo -H pip3 install testresources


# above 58.3.0 you get version issues

sudo -H pip3 install setuptools==58.3.0
sudo -H pip3 install Cython


# install gdown to download from Google drive

sudo -H pip3 install gdown


# download the wheel

export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl


# install PyTorch 2.0.0

python3 -m pip install --upgrade pip

python3 -m pip install numpy==1.26.1 

python3 -m pip install --no-cache $TORCH_INSTALL

