#!/usr/bin/env bash

echo -e "Building moveit1_ur:lastest \nWARNING: this script must be run from the root of the repo not from within the .docker folder"

docker build --pull --rm -f ./.docker/Dockerfile_zsh  -t ros_kvp:latest .