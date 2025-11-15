conda create --name pytorch_lightning  python==3.10

conda env remove --name pytorch_lightning

pip install torch==2.2 torchvision==0.17.0 torchmetrics==0.7.0  lightning==2.2 numpy==1.24.3 pandas==2.3.3 pillow==12.0.0 openpyxl

pip install litserve
pip install -U openai-whisper

sudo chmod +x /Users/cuijinlong/Documents/dev_tools/ffmpeg
mv /Users/cuijinlong/Documents/dev_tools/ffmpeg /usr/local/bin/
ls -la /usr/local/bin/ffmpeg
sudo chmod +x /usr/local/bin/ffmpeg
echo $PATH
which ffmpeg