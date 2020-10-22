sudo apt-get install libatlas-base-dev -y
sudo apt install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools python3-wheel python3-pillow python3-numpy -y
sudo apt-get install tightvncserver -y

pip3 install opencv-python --no-cache-dir
pip3 install imutils --no-cache-dir
pip3 install matplotlib --no-cache-dir
pip3 install scikit-image --no-cache-dir
pip3 install numpy --upgrade
pip3 install "picamera[array]"
pip3 install scikit-learn --no-cache-dir
pip3 install torch torch-vision --no-cache-dir
pip3 install apscheduler --no-cache-dir

wget https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B/raw/master/torch-1.4.0a0%2Bf43194e-cp37-cp37m-linux_armv7l.whl
wget https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B/raw/master/torchvision-0.5.0a0%2B9cdc814-cp37-cp37m-linux_armv7l.whl
pip3 install torch-1.4.0a0%2Bf43194e-cp37-cp37m-linux_armv7l.whl torchvision-0.5.0a0%2B9cdc814-cp37-cp37m-linux_armv7l.whl

pip3 install pycairo --no-cache-dir
sudo apt-get install python3-gi-cairo -y

