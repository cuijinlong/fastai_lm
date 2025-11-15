import whisper
import os
import subprocess

# 方法1：直接设置环境变量
os.environ["PATH"] = "/usr/local/bin:" + os.environ["PATH"]

# 方法2：在调用 whisper 时指定 ffmpeg 路径
os.environ["WHISPER_FFMPEG_PATH"] = "/usr/local/bin/ffmpeg"

# 测试 ffmpeg 是否可用
try:
    result = subprocess.run(["/usr/local/bin/ffmpeg", "-version"],
                          capture_output=True, text=True, check=True)
    print("ffmpeg is working:")
    print(result.stdout.split('\n')[0])
except Exception as e:
    print(f"ffmpeg error: {e}")


model = whisper.load_model("base")
result = model.transcribe("/Users/cuijinlong/Documents/workspace_py/fastai_lm/ffmpeg/基底细胞癌病例.m4a")
print(result["text"])

# response = requests.post("http://127.0.0.1:8000/predict",
#                          json={"audio_path": "/Users/cuijinlong/Documents/workspace_py/fastai_lm/ffmpeg/a.m4a"})
# print(f"Status: {response.status_code}\nResponse:\n {response.text}")
