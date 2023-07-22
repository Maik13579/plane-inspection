ARG ROS_DISTRO=noetic

FROM ros-base-image:$ROS_DISTRO

## Install deps
RUN apt update && apt install -y\
  python3-pip \
  ffmpeg libsm6 libxext6 \
  ros-$ROS_DISTRO-tf \
  ros-$ROS_DISTRO-tf2-sensor-msgs \
  ros-$ROS_DISTRO-tf2-geometry-msgs \
  ros-$ROS_DISTRO-visualization-msgs \
  ros-$ROS_DISTRO-vision-msgs \
&& pip3 install --upgrade open3d numpy compas scipy python-dateutil\
&& rm -rf /var/lib/apt/lists/*
RUN pip3 install opencv-python


WORKDIR /root/catkin_ws

# Get source code
COPY python /root/python
COPY ros src/ros

# Build python package
RUN cd /root/python && python3 setup.py bdist_wheel sdist && pip3 install -e .

# Build ros package
RUN source /opt/ros/$ROS_DISTRO/setup.bash && catkin_make
