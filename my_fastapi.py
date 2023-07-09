from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from multiprocessing import Process, Queue

from detect_api import detect_api, load_model

app = FastAPI()

class DetectionParams(BaseModel):
    # weights: List[str] = ['yolov7.pt']
    source: str = 'inference/images/test'
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    device: str = 'cpu'
    view_img: bool = False
    save_txt: bool = True
    save_conf: bool = False
    nosave: bool = True
    classes: Optional[List[int]] = None
    agnostic_nms: bool = False
    augment: bool = False
    update: bool = False
    project: str = 'runs/detect'
    name: str = 'exp'
    exist_ok: bool = False
    no_trace: bool = False

model = None
names = None
colors = None
old_img_w = None
old_img_h = None
old_img_b = None
half = None
classify = None
imgsz = None
trace = None
stride = None
modelc = None

@app.on_event("startup")
async def load_model_on_startup(
    weights=["yolov7.pt"],
    source="inference/images/test",
    img_size=640,
    conf_thres=0.25,
    iou_thres=0.45,
    device="cpu",
    view_img=False,
    save_txt=True,
    save_conf=False,
    nosave=True,
    classes=None,
    agnostic_nms=False,
    augment=False,
    update=False,
    project="runs/detect",
    name="exp",
    exist_ok=False,
    no_trace=False,
):
    global model
    global names
    global colors
    global old_img_w
    global old_img_h
    global old_img_b
    global half
    global classify
    global imgsz
    global trace
    global stride
    global modelc
    model, names, colors, old_img_w, old_img_h, old_img_b, half, classify, imgsz, trace, stride, imgsz, modelc = load_model(
        weights,
        source,
        img_size,
        conf_thres,
        iou_thres,
        device,
        view_img,
        save_txt,
        save_conf,
        nosave,
        classes,
        agnostic_nms,
        augment,
        update,
        project,
        name,
        exist_ok,
        no_trace,
    )  # スタートアップ時にモデルを読み込みます。

@app.post("/detect/")
async def detect(params: DetectionParams):
    # ここで提供されたコードを使用して検出を実行します。
    # `params`オブジェクトは、リクエストボディから自動的に生成され、
    # その属性はリクエストボディの対応するキーによって設定されます。
    # 例えば、`params.weights`、`params.source`などです。
    # この関数内で`detect`関数を呼び出し、必要なパラメータを`params`から取得します。
    # print(f"params: {params}\n"
    #       f"params.weights: {params.weights}\n")
    result = detect_api(
        model,
        names,
        colors,
        old_img_w,
        old_img_h,
        old_img_b,
        half,
        classify,
        imgsz,
        trace,
        stride,
        modelc,
        params.device,
        params.save_txt,
        params.view_img,
        params.conf_thres,
        params.iou_thres,
        params.save_conf,
        params.classes,
        params.agnostic_nms,
        params.augment,
        params.update,
        params.project,
        params.name,
        params.exist_ok,
        params.img_size,
        params.no_trace,
        params.nosave,
        params.source,
    )
    # 結果を返します。
    return {"result": result}