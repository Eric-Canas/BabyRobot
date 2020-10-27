sudo apt-get install unzip -y

cd /home/pi/Desktop 
wget --no-check-certificate --content-disposition https://github.com/Eric-Canas/BabyRobot/archive/main.zip
unzip BabyRobot-main.zip -d .
rm ./BabyRobot-main.zip
