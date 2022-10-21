#!/usr/bin/env bash

echo -e "Building kvp_recon_ros:lastest 

docker build --pull --rm -f ./.docker/Dockerfile  -t kvp_recon_ros:latest .