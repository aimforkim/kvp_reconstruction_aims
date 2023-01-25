ARG ROSDISTRO=noetic

FROM ros:$ROSDISTRO-robot

ARG ROSDISTRO

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

ENV TERM xterm-256color

# Tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    zsh \
    ssh \
    git \
    curl \
    python3-pip \
    python3-flake8 \
    build-essential \
    cmake \
    iputils-ping 

RUN pip3 install -U --no-cache-dir\
    argcomplete \
    flake8-blind-except \
    flake8-builtins \
    flake8-class-newline \
    flake8-comprehensions \
    flake8-deprecated \
    flake8-return \
    flake8-length \
    flake8-todo \
    flake8-quotes \
    black \
    numpy \
    open3d \
    pyquaternion \
    python-dateutil

# ROS
RUN apt-get update && apt-get install --no-install-recommends -y \
    pkg-config \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall-generator \
    python3-vcstool \
    ros-$ROSDISTRO-rqt \
    ros-$ROSDISTRO-rqt-action \
    ros-$ROSDISTRO-rqt-console \
    ros-$ROSDISTRO-rqt-service-caller \
    ros-$ROSDISTRO-rqt-graph \
    ros-$ROSDISTRO-rqt-topic \
    ros-$ROSDISTRO-rqt-top \
    ros-$ROSDISTRO-rqt-reconfigure \
    ros-$ROSDISTRO-rqt-tf-tree \
    ros-$ROSDISTRO-rviz \
    # Realsense
    software-properties-common \
    ros-$ROS_DISTRO-rqt-image-view \
    ros-$ROS_DISTRO-ddynamic-reconfigure \
    ros-$ROS_DISTRO-rgbd-launch 

# Realsense
RUN apt-key adv \
    --keyserver keyserver.ubuntu.com \
    --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || \
    apt-key adv \
    --keyserver hkp://keyserver.ubuntu.com:80 \
    --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE \
    && add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u \
    && apt-get install -y --install-recommends \
    librealsense2-dkms librealsense2-utils \
    librealsense2-dev librealsense2-dbg

# Reduce size by removing apt cache
RUN ["/bin/bash", "-c", "rm -rf /var/lib/apt/lists/*"]

# Create workspace
RUN mkdir -p dev_ws/src
WORKDIR /dev_ws/src

# Clone pkgs
RUN ["/bin/bash", "-c", "git clone https://github.com/rjwb1/realsense-ros.git"]

# Copy local pkg's from host to container
COPY . /dev_ws/src/

# Build catkin workspace
WORKDIR /dev_ws
RUN ["/bin/bash", "-c", "source /opt/ros/$ROSDISTRO/setup.bash &&\
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release &&\
    catkin build \
    "]

# Set shell to zsh
SHELL ["zsh", "chsh -s ~/.zshrc" ]

# Copy and make setup script executable
COPY .docker/setup.zsh /dev_ws/setup.zsh
RUN ["/bin/bash", "-c", " chmod 777 /dev_ws/setup.zsh"]

# Copy and make entrypoint script executable
RUN ["/bin/bash", "-c", "cp /dev_ws/src/.docker/entrypoint_zsh.sh / && chmod 777 /entrypoint_zsh.sh"]
ENTRYPOINT ["zsh", "/entrypoint_zsh.sh" ]

# start ZSH shell
CMD ["zsh"]