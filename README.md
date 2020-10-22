# BabyRobot

For connecting: 
```bash
ssh -X pi@raspberrypi
password: password
```

(Streaming video only can be viewed from Putty. Enabling X11Forward)

For downloading the project:
```bash
sudo apt-get install unzip -y

cd Desktop 
wget --no-check-certificate --content-disposition https://github.com/Eric-Canas/BabyRobot/archive/main.zip
unzip BabyRobot-main.zip -d .
rm ./BabyRobot-main.zip
```

For executing it:
```bash
cd BabyRobot-main
python3 ./main.py
```
