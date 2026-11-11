"""번호판 모델을 두 가지로 잰다.

1) 찾는 비율: 한국 번호판이 찍힌 공개 사진(Wikimedia Commons 'Automobiles with license plates of
   South Korea' 분류, 50장)에서 점수 기준 이상 상자가 하나라도 나오는지. 정답 상자가 없어서
   위치가 맞는지는 놓친 사진을 눈으로 확인했다(README '번호판' 절).
2) 잘못 가리는 비율: COCO val2017 500장 중 차량(car·motorcycle·bus·truck) 라벨이 없는 사진에서
   번호판 상자가 나온 사진 수.

app.js 와 같은 640 레터박스(회색 114, RGB 0~1)를 쓴다.

    python tools/eval_plates.py <한국 번호판 사진 폴더> <COCO 사진 폴더> <COCO 라벨 폴더(YOLO txt)> <사진 목록.txt>
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
S = 640
VEHICLES = {2, 3, 5, 7}
THR = [0.3, 0.4, 0.5, 0.6, 0.7]


def plates(sess, img):
    h, w = img.shape[:2]
    r = S / max(w, h)
    nw, nh = round(w * r), round(h * r)
    px, py = (S - nw) // 2, (S - nh) // 2
    lb = np.full((S, S, 3), 114, np.uint8)
    lb[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
    o = sess.run(None, {sess.get_inputs()[0].name: x})[0]   # [n, 7] = 배치, x1, y1, x2, y2, 종류, 점수
    return [float(row[6]) for row in o]


def main():
    kr, coco, labels, pick = map(Path, sys.argv[1:5])
    sess = ort.InferenceSession(str(ROOT / "model/yolo-v9-t-640-license-plates-end2end.onnx"))

    kr_scores = {}
    for p in sorted(kr.iterdir()):
        img = None if p.name.startswith(".") else cv2.imread(str(p))
        if img is not None:
            kr_scores[p.name] = [round(s, 3) for s in plates(sess, img)]
    found = {t: sum(any(s >= t for s in v) for v in kr_scores.values()) for t in THR}

    no_veh = 0
    wrong_imgs = {t: 0 for t in THR}
    for iid in pick.read_text().split():
        img = cv2.imread(str(coco / f"{iid}.jpg"))
        if img is None:
            continue
        lab = labels / f"{iid}.txt"
        cls = {int(l.split()[0]) for l in lab.read_text().splitlines() if l.strip()} if lab.exists() else set()
        if cls & VEHICLES:
            continue
        no_veh += 1
        sc = plates(sess, img)
        for t in THR:
            wrong_imgs[t] += any(s >= t for s in sc)

    result = {
        "korean_photos": len(kr_scores),
        "korean_photos_with_plate_box": {str(t): found[t] for t in THR},
        "coco_no_vehicle_photos": no_veh,
        "coco_no_vehicle_photos_with_plate_box": {str(t): wrong_imgs[t] for t in THR},
        "korean_scores": kr_scores,
    }
    out = ROOT / "eval" / "plates.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "korean_scores"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
