#!/bin/bash

# Go to the Desktop
cd ~/Desktop 
# If the software was already there remove it before downloading the last version
if test -d ./BabyRobot; then
    rm -r -f ./BabyRobot
fi
# Cloning the repository from github
git clone https://github.com/Eric-Canas/BabyRobot
cd ./BabyRobot