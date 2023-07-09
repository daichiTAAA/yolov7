import requests
import json
import concurrent.futures

# POSTリクエストのボディを定義します。
data = {
    "weights": ["yolov7.pt"],
    "source": "inference/images/test",
    "img_size": 640,
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "device": "cpu",
    "view_img": False,
    "save_txt": True,
    "save_conf": False,
    "nosave": True,
    "classes": None,
    "agnostic_nms": False,
    "augment": False,
    "update": False,
    "project": "runs/detect",
    "name": "exp",
    "exist_ok": False,
    "no_trace": False,
}

# JSON形式に変換します。
json_data = json.dumps(data)

# POSTリクエストを送信する関数を定義します。
def send_post_request():
    response = requests.post("http://localhost:8000/detect/", data=json_data)
    print(response.json())

# 複数のスレッドでPOSTリクエストを同時に送信します。
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    for _ in range(5):
        executor.submit(send_post_request)
