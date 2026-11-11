import sys, cv2, numpy as np, onnxruntime as ort
img = cv2.imread(sys.argv[2]); H, W = img.shape[:2]; S = 640
r = S / max(W, H); nw, nh = round(W*r), round(H*r); px, py = (S-nw)//2, (S-nh)//2
lb = np.full((S, S, 3), 114, np.uint8); lb[py:py+nh, px:px+nw] = cv2.resize(img, (nw, nh))
# OpenCV reference
ref = cv2.FaceDetectorYN.create(sys.argv[1], "", (S, S), 0.5, 0.3, 5000)
_, f = ref.detect(lb); f = [] if f is None else f
print("cv2:", [(round(x[14], 3), [round(v) for v in x[:4]]) for x in f])
# my decode (same as app.js)
sess = ort.InferenceSession(sys.argv[1])
x = lb.transpose(2, 0, 1)[None].astype(np.float32)  # BGR, 0..255
o = dict(zip([a.name for a in sess.get_outputs()], sess.run(None, {"input": x})))
out = []
for s in (8, 16, 32):
    cols = S // s; cls, obj, bb = o[f"cls_{s}"][0, :, 0], o[f"obj_{s}"][0, :, 0], o[f"bbox_{s}"][0]
    for i in range(len(cls)):
        sc = np.sqrt(np.clip(cls[i], 0, 1) * np.clip(obj[i], 0, 1))
        if sc < 0.5: continue
        r_, c_ = divmod(i, cols)
        cx, cy = (c_ + bb[i, 0]) * s, (r_ + bb[i, 1]) * s
        w, h = np.exp(bb[i, 2]) * s, np.exp(bb[i, 3]) * s
        out.append((float(sc), [cx - w/2, cy - h/2, w, h]))
out.sort(key=lambda t: -t[0]); keep = []
def iou(a, b):
    ax2, ay2, bx2, by2 = a[0]+a[2], a[1]+a[3], b[0]+b[2], b[1]+b[3]
    iw = max(0, min(ax2, bx2)-max(a[0], b[0])); ih = max(0, min(ay2, by2)-max(a[1], b[1]))
    return iw*ih / (a[2]*a[3] + b[2]*b[3] - iw*ih)
for c in out:
    if all(iou(c[1], k[1]) < 0.3 for k in keep): keep.append(c)
print("mine:", [(round(s, 3), [round(v) for v in b]) for s, b in keep])
