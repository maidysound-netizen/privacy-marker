"""사람을 얼마나 놓치는지 잰다 (COCO val2017 일부, 정답 박스 있음).

app.js 와 같은 모델·같은 전처리(640 레터박스)·같은 후처리 규칙을 파이썬으로 다시 짠 것이다.
브라우저(ONNX Runtime Web)와 파이썬(onnxruntime)은 같은 ONNX 파일을 쓰지만, 사진을 줄이는
방식(캔버스 vs OpenCV)이 달라 점수가 조금 다를 수 있다.

    python tools/eval_recall.py <이미지 폴더> <라벨 폴더(YOLO txt)> <사진 목록 txt>

YOLO26 은 COCO train2017 로 학습했으므로 val2017 은 학습에 쓰이지 않은 사진이다.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
S = 640
PERSON, VEHICLES = 0, {2, 3, 5, 7}
WANTED = [PERSON, 62, 63, 67, *sorted(VEHICLES)]
SENS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FACE_THR = [0.5, 0.6, 0.7, 0.8]


def letterbox(img):
    h, w = img.shape[:2]
    r = S / max(w, h)
    nw, nh = round(w * r), round(h * r)
    px, py = (S - nw) // 2, (S - nh) // 2
    lb = np.full((S, S, 3), 114, np.uint8)
    lb[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return lb, r, px, py


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def nms(cand, thr, same_class):
    cand.sort(key=lambda c: -c["score"])
    keep = []
    for c in cand:
        if all((same_class and k["cls"] != c["cls"]) or iou(k["box"], c["box"]) < thr for k in keep):
            keep.append(c)
    return keep


def unletter(box, r, px, py, w, h):
    return [min(max((v - (py if j % 2 else px)) / r, 0), h if j % 2 else w) for j, v in enumerate(box)]


def detect(sess, lb, r, px, py, w, h):
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
    o = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]  # [84, 8400]
    scores = o[4:]
    sub = scores[WANTED]
    best_i = sub.argmax(0)
    best = np.array(WANTED)[best_i]
    bs = sub.max(0)
    top = scores.max(0)
    cand = []
    for i in np.where(bs >= 0.10)[0]:
        if top[i] > bs[i] + 1e-6 and best[i] != PERSON:
            continue
        cx, cy, bw, bh = o[:4, i]
        cand.append({"cls": int(best[i]), "score": float(bs[i]),
                     "box": unletter([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], r, px, py, w, h)})
    return nms(cand, 0.7, True)


def faces(sess, lb, r, px, py, w, h):
    o = dict(zip([a.name for a in sess.get_outputs()], sess.run(None, {"input": lb.transpose(2, 0, 1)[None].astype(np.float32)})))
    cand = []
    for s in (8, 16, 32):
        cols = S // s
        sc = np.sqrt(np.clip(o[f"cls_{s}"][0, :, 0], 0, 1) * np.clip(o[f"obj_{s}"][0, :, 0], 0, 1))
        bb = o[f"bbox_{s}"][0]
        for i in np.where(sc >= min(FACE_THR))[0]:
            rr, cc = divmod(int(i), cols)
            cx, cy = (cc + bb[i, 0]) * s, (rr + bb[i, 1]) * s
            fw, fh = np.exp(bb[i, 2]) * s, np.exp(bb[i, 3]) * s
            cand.append({"cls": -1, "score": float(sc[i]),
                         "box": unletter([cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2], r, px, py, w, h)})
    return nms(cand, 0.3, False)


def load_gt(path, w, h):
    gt = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, bw, bh = line.split()[:5]
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            gt.append({"cls": int(c), "box": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]})
    return gt


def covered_fraction(box, preds):
    """정답 상자 중 예측 막대(몸 전체)로 덮인 넓이 비율. 격자로 근사."""
    x1, y1, x2, y2 = box
    xs, ys = np.linspace(x1, x2, 20), np.linspace(y1, y2, 20)
    gx, gy = np.meshgrid(xs, ys)
    hit = np.zeros_like(gx, bool)
    for p in preds:
        hit |= (gx >= p[0]) & (gx <= p[2]) & (gy >= p[1]) & (gy <= p[3])
    return hit.mean()


def size_of(box):
    a = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if a < 32 ** 2 else "medium" if a < 96 ** 2 else "large"   # COCO 기준


def main():
    img_dir, lab_dir, pick = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    det = ort.InferenceSession(str(ROOT / "model/yolo26n.onnx"))
    fac = ort.InferenceSession(str(ROOT / "model/face_detection_yunet_2023mar.onnx"))
    ids = pick.read_text().split()

    per_sens = {s: {"tp": 0, "tp_face": 0, "cov": 0, "fp": 0, "n": 0,
                    "by_size": {k: [0, 0] for k in ("small", "medium", "large")},
                    "veh_tp": 0, "veh_n": 0} for s in SENS}
    face_fp_noperson = {t: 0 for t in FACE_THR}
    n_person_imgs = n_noperson_imgs = 0
    missed_examples = []

    for k, iid in enumerate(ids):
        img = cv2.imread(str(img_dir / f"{iid}.jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(lab_dir / f"{iid}.txt", w, h)
        gp = [g for g in gt if g["cls"] == PERSON]
        gv = [g for g in gt if g["cls"] in VEHICLES]
        lb, r, px, py = letterbox(img)
        preds = detect(det, lb, r, px, py, w, h)
        fs = faces(fac, lb, r, px, py, w, h)

        if not gp:
            n_noperson_imgs += 1
            for t in FACE_THR:
                face_fp_noperson[t] += sum(f["score"] >= t for f in fs)
        else:
            n_person_imgs += 1
        f06 = [f for f in fs if f["score"] >= 0.6]

        for s in SENS:
            st = per_sens[s]
            pp = [p for p in preds if p["cls"] == PERSON and p["score"] >= s]
            used = set()
            for g in gp:
                st["n"] += 1
                sz = size_of(g["box"])
                st["by_size"][sz][1] += 1
                best, bj = 0, -1
                for j, p in enumerate(pp):
                    if j in used:
                        continue
                    v = iou(g["box"], p["box"])
                    if v > best:
                        best, bj = v, j
                hit = best >= 0.5
                if hit:
                    used.add(bj)
                    st["tp"] += 1
                    st["by_size"][sz][0] += 1
                # 사람 상자로는 놓쳤어도, 정답 상자 위쪽 절반에 얼굴(0.6 이상)이 있으면 얼굴 막대로 가려진다
                gx1, gy1, gx2, gy2 = g["box"]
                by_face = any(gx1 <= (f["box"][0] + f["box"][2]) / 2 <= gx2 and
                              gy1 <= (f["box"][1] + f["box"][3]) / 2 <= gy1 + (gy2 - gy1) / 2 for f in f06)
                if hit or by_face:
                    st["tp_face"] += 1
                if covered_fraction(g["box"], [p["box"] for p in pp]) >= 0.5:
                    st["cov"] += 1
                if s == 0.20 and not hit and not by_face and len(missed_examples) < 400:
                    missed_examples.append({"image": iid, "size": sz, "box": [round(v) for v in g["box"]]})
            st["fp"] += len(pp) - len(used)
            vp = [p for p in preds if p["cls"] in VEHICLES and p["score"] >= s]
            for g in gv:
                st["veh_n"] += 1
                if any(iou(g["box"], p["box"]) >= 0.5 for p in vp):
                    st["veh_tp"] += 1
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(ids)}", file=sys.stderr)

    rows = []
    for s in SENS:
        st = per_sens[s]
        rows.append({
            "sens": s,
            "people": st["n"],
            "recall_iou50": round(st["tp"] / st["n"], 3),
            "recall_with_faces": round(st["tp_face"] / st["n"], 3),
            "covered50_body_mode": round(st["cov"] / st["n"], 3),
            "recall_by_size": {k: [v[0], v[1], round(v[0] / v[1], 3) if v[1] else None] for k, v in st["by_size"].items()},
            "unmatched_person_bars_per_image": round(st["fp"] / n_person_imgs, 2),
            "vehicle_recall_iou50": round(st["veh_tp"] / st["veh_n"], 3) if st["veh_n"] else None,
            "vehicles": st["veh_n"],
        })
    result = {
        "data": f"COCO val2017: 사람 있는 사진 {n_person_imgs}장 + 사람 없는 사진 {n_noperson_imgs}장",
        "images": ids,
        "by_sensitivity": rows,
        "face_bars_on_no_person_images": {str(t): face_fp_noperson[t] for t in FACE_THR},
        "missed_at_0.20": missed_examples,
    }
    out = ROOT / "eval" / "result.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for r_ in rows:
        print(json.dumps({k: v for k, v in r_.items()}, ensure_ascii=False))
    print("face bars on no-person images:", result["face_bars_on_no_person_images"], "/", n_noperson_imgs, "images")


if __name__ == "__main__":
    main()
