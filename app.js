// 까만 막대 — 브라우저 안에서 사람·얼굴·화면·차량을 찾아 까만 막대로 가린다.
// 사진은 서버로 보내지 않는다. 저장은 새 캔버스로 다시 그려서 EXIF가 남지 않는다.
"use strict";

const ORT_VER = "1.20.1"; // vendor/ort 에 들어 있는 버전
const MODEL_URL = "model/yolo26n.onnx";
const FACE_URL = "model/face_detection_yunet_2023mar.onnx"; // OpenCV Zoo YuNet (MIT)
const SIZE = 640;              // 두 모델 모두 입력이 640×640
const MIN_SCORE = 0.10;        // 한 번 추론해 두고, 슬라이더는 이 위에서만 거른다
const NMS_IOU = 0.7;           // Ultralytics 기본값
const FACE_MIN = 0.6;          // 얼굴 점수 기준. 근거는 README '측정' 절 (사람 없는 사진에서의 잘못 가림)
const FACE_NMS = 0.3;          // OpenCV FaceDetectorYN 기본값
const MAX_SIDE = 4096;         // 아주 큰 사진은 메모리 때문에 이 크기로 줄여 저장
const MAX_PHOTOS = 50;         // 한 번에 다루는 사진 수 (휴대폰 메모리)
const PERSON = 0;
const SCREENS = { 62: "모니터·TV", 63: "노트북", 67: "휴대폰" };
const VEHICLES = { 2: "자동차", 3: "오토바이", 5: "버스", 7: "트럭" };

const $ = (id) => document.getElementById(id);
const view = $("view"), ctx = view.getContext("2d");

// 설정은 모든 사진에 같이 적용, 막대 끄기·직접 그린 막대는 사진마다 따로
const state = {
  session: null, face: null,
  photos: [],        // {blob, name, W, H, dets, faces, off:Set, manual:[], status, thumb, seen}
  cur: -1,
  cache: null,       // {photo, canvas} 지금 보고 있는 사진의 원본 크기 캔버스
  cover: "face", screens: true, vehicles: false, sens: 0.20,
};
const cur = () => state.photos[state.cur];

// ---------- 모델 ----------
async function loadModels() {
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", location.href).href;   // onnxruntime-web ${ORT_VER}, 직접 포함
  if (!self.crossOriginIsolated) ort.env.wasm.numThreads = 1;
  try {
    state.session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  } catch (e) {
    console.error(e);
    setStatus("모델을 불러오지 못했어요. 인터넷 연결을 확인하고 새로고침하세요.");
    return;
  }
  try {
    state.face = await ort.InferenceSession.create(FACE_URL, { executionProviders: ["wasm"] });
  } catch (e) {
    // 얼굴 모델이 없어도 사람 상자 위쪽 어림값으로 가릴 수 있다
    console.error(e);
  }
  setStatus(state.face ? "준비됐어요. 사진을 고르세요(여러 장 가능)." : "준비됐어요. (얼굴 모델 없이 머리 위치를 어림합니다)", "ok");
  pump();
}

// 긴 쪽을 640에 맞추고 가운데에 놓은 640×640 픽셀 (회색 114로 채움)
function letterboxPixels(img, W, H) {
  const r = SIZE / Math.max(W, H);
  const nw = Math.round(W * r), nh = Math.round(H * r);
  const px = Math.floor((SIZE - nw) / 2), py = Math.floor((SIZE - nh) / 2);
  const c = document.createElement("canvas"); c.width = c.height = SIZE;
  const g = c.getContext("2d");
  g.fillStyle = "rgb(114,114,114)"; g.fillRect(0, 0, SIZE, SIZE);
  g.imageSmoothingQuality = "high";
  g.drawImage(img, px, py, nw, nh);
  return { d: g.getImageData(0, 0, SIZE, SIZE).data, r, px, py };
}

// YOLO: RGB, 0~1 / YuNet: BGR, 0~255 (OpenCV 방식)
function toTensor(d, bgr) {
  const n = SIZE * SIZE, x = new Float32Array(3 * n);
  const k = bgr ? 1 : 1 / 255, [a, b, c] = bgr ? [2, 1, 0] : [0, 1, 2];
  for (let i = 0; i < n; i++) {
    x[i] = d[i * 4 + a] * k; x[n + i] = d[i * 4 + b] * k; x[2 * n + i] = d[i * 4 + c] * k;
  }
  return new ort.Tensor("float32", x, [1, 3, SIZE, SIZE]);
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = ix * iy;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i);
}

function nms(cand, thr, sameClassOnly) {
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cand) if (keep.every((k) => (sameClassOnly && k.cls !== c.cls) || iou(k.box, c.box) < thr)) keep.push(c);
  return keep;
}

function unletter(box, lb, W, H) {
  return box.map((v, j) => Math.min(Math.max((v - (j % 2 ? lb.py : lb.px)) / lb.r, 0), j % 2 ? H : W));
}

async function detectObjects(lb, W, H) {
  const out = await state.session.run({ [state.session.inputNames[0]]: toTensor(lb.d, false) });
  const o = out[state.session.outputNames[0]];
  const [, C, N] = o.dims;              // [1, 84, 8400]: cx,cy,w,h + 80 클래스 점수
  const d = o.data;
  const wanted = [PERSON, ...Object.keys(SCREENS).map(Number), ...Object.keys(VEHICLES).map(Number)];
  const cand = [];
  for (let i = 0; i < N; i++) {
    let best = -1, bs = 0;
    for (const k of wanted) { const s = d[(4 + k) * N + i]; if (s > bs) { bs = s; best = k; } }
    if (bs < MIN_SCORE) continue;
    // 다른 클래스 점수가 더 높으면 그 물건으로 본다 (예: 사람보다 의자일 확률이 높은 칸).
    // 사람은 놓치는 쪽이 손해라 예외로 남긴다.
    let top = 0; for (let k = 0; k < C - 4; k++) top = Math.max(top, d[(4 + k) * N + i]);
    if (top > bs + 1e-6 && best !== PERSON) continue;
    const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
    cand.push({ cls: best, score: bs, box: unletter([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], lb, W, H) });
  }
  return nms(cand, NMS_IOU, true).map((k, i) => ({ id: "d" + i, ...k }));
}

// YuNet 출력 풀기: OpenCV FaceDetectorYN 과 같은 계산 (tools/check_yunet.py 로 대조함)
async function detectFaces(lb, W, H) {
  if (!state.face) return [];
  const o = await state.face.run({ input: toTensor(lb.d, true) });
  const cand = [];
  for (const s of [8, 16, 32]) {
    const cols = SIZE / s, cls = o[`cls_${s}`].data, obj = o[`obj_${s}`].data, bb = o[`bbox_${s}`].data;
    for (let i = 0; i < cls.length; i++) {
      const score = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
      if (score < FACE_MIN) continue;
      const r = Math.floor(i / cols), c = i % cols;
      const cx = (c + bb[i * 4]) * s, cy = (r + bb[i * 4 + 1]) * s;
      const w = Math.exp(bb[i * 4 + 2]) * s, h = Math.exp(bb[i * 4 + 3]) * s;
      cand.push({ score, box: unletter([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], lb, W, H) });
    }
  }
  return nms(cand, FACE_NMS, false).map((k, i) => ({ id: "f" + i, ...k }));
}

// ---------- 막대 계산 ----------
function headBox([x1, y1, x2, y2], W) {
  const w = x2 - x1, h = y2 - y1;
  const hh = Math.min(h, Math.max(0.28 * h, 0.5 * w));   // 서 있는 사람은 위 28%, 상반신만 찍혔으면 더 크게
  const padX = 0.06 * w;
  return [Math.max(0, x1 - padX), Math.max(0, y1 - 0.03 * h), Math.min(W, x2 + padX), y1 + hh];
}

// 얼굴 상자는 눈썹~턱이라 머리카락·귀까지 덮게 넓힌다
function faceBar([x1, y1, x2, y2], W, H) {
  const w = x2 - x1, h = y2 - y1;
  return [Math.max(0, x1 - 0.18 * w), Math.max(0, y1 - 0.35 * h), Math.min(W, x2 + 0.18 * w), Math.min(H, y2 + 0.12 * h)];
}

function autoBars(p) {
  if (!p || !p.dets) return [];
  const bars = [];
  const add = (id, label, score, box) => bars.push({ id, label, score, box, on: !p.off.has(id) });
  const people = p.dets.filter((d) => d.cls === PERSON && d.score >= state.sens);
  const faces = p.faces || [];
  const usedFace = new Set();

  for (const d of people) {
    if (state.cover === "body") { add(d.id, "사람", d.score, d.box); continue; }
    // 얼굴 모드: 이 사람의 머리 쪽에 얼굴이 있으면 얼굴만, 없으면(뒷모습·작음) 머리 쪽 어림값
    const hb = headBox(d.box, p.W);
    const mine = faces.filter((f) => {
      const cx = (f.box[0] + f.box[2]) / 2, cy = (f.box[1] + f.box[3]) / 2;
      return cx >= hb[0] && cx <= hb[2] && cy >= hb[1] && cy <= hb[3];
    });
    if (mine.length) mine.forEach((f) => usedFace.add(f.id));
    else add(d.id, "사람(얼굴 못 찾음)", d.score, hb);
  }
  // 얼굴은 사람 상자와 상관없이 모두 가린다 (사람을 놓쳐도 얼굴로 한 번 더 잡는다)
  for (const f of faces) {
    if (state.cover === "body" && people.some((d) => {
      const cx = (f.box[0] + f.box[2]) / 2, cy = (f.box[1] + f.box[3]) / 2;
      return cx >= d.box[0] && cx <= d.box[2] && cy >= d.box[1] && cy <= d.box[3];
    })) continue;   // 몸 전체로 이미 덮였다
    add(f.id, "얼굴", f.score, faceBar(f.box, p.W, p.H));
  }
  for (const d of p.dets) {
    if (d.score < state.sens) continue;
    if (SCREENS[d.cls] && state.screens) add(d.id, SCREENS[d.cls], d.score, d.box);
    if (VEHICLES[d.cls] && state.vehicles) add(d.id, VEHICLES[d.cls], d.score, d.box);
  }
  return bars;
}

function allOnBoxes(p) {
  return [...autoBars(p).filter((b) => b.on).map((b) => b.box), ...p.manual];
}

// ---------- 사진 읽기 ----------
async function decode(blob) {
  const url = URL.createObjectURL(blob);
  const el = new Image();
  // decode() 대신 onload: 백그라운드 탭에서도 멈추지 않는다
  const ok = await new Promise((res) => { el.onload = () => res(true); el.onerror = () => res(false); el.src = url; });
  URL.revokeObjectURL(url);
  if (!ok) return null;
  // 브라우저가 EXIF 방향을 반영해 그려 준다. 너무 큰 사진은 줄인다.
  let W = el.naturalWidth, H = el.naturalHeight;
  const k = Math.min(1, MAX_SIDE / Math.max(W, H));
  W = Math.round(W * k); H = Math.round(H * k);
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  c.getContext("2d").drawImage(el, 0, 0, W, H);
  return c;
}

async function canvasOf(p) {
  if (state.cache && state.cache.photo === p) return state.cache.canvas;
  const c = await decode(p.blob);
  if (c && p === cur()) state.cache = { photo: p, canvas: c };
  return c;
}

function thumbOf(c) {
  const t = document.createElement("canvas"), k = 120 / Math.max(c.width, c.height);
  t.width = Math.round(c.width * k); t.height = Math.round(c.height * k);
  t.getContext("2d").drawImage(c, 0, 0, t.width, t.height);
  return t.toDataURL("image/jpeg", 0.7);
}

// 사진을 하나씩 차례로 처리 (동시에 여러 장 돌리면 휴대폰이 버거워한다)
let pumping = false;
async function pump() {
  if (pumping || !state.session) return;
  pumping = true;
  try {
    for (let p; (p = state.photos.find((x) => x.status === "wait"));) {
      p.status = "run"; renderStrip(); progress();
      const c = await decode(p.blob);
      if (!c) { p.status = "error"; renderStrip(); if (p === cur()) render(); continue; }
      p.W = c.width; p.H = c.height; p.thumb = thumbOf(c);
      const t = performance.now();
      const lb = letterboxPixels(c, p.W, p.H);
      p.dets = await detectObjects(lb, p.W, p.H);
      p.faces = await detectFaces(lb, p.W, p.H);
      p.ms = Math.round(performance.now() - t);
      p.status = "done";
      if (p === cur()) state.cache = { photo: p, canvas: c };
      renderStrip();
      if (state.cur < 0 || p === cur()) select(state.photos.indexOf(p));
    }
  } finally { pumping = false; }
  progress();
}

function addFiles(files) {
  const imgs = [...files].filter((f) => !f.type || f.type.startsWith("image/"));
  if (!imgs.length) return;
  const room = MAX_PHOTOS - state.photos.length;
  if (room <= 0) { setStatus(`한 번에 ${MAX_PHOTOS}장까지예요. 저장한 뒤 '모두 지우기'를 누르세요.`); return; }
  const first = state.photos.length;
  for (const f of imgs.slice(0, room)) {
    state.photos.push({ blob: f, name: (f.name || "photo").replace(/\.[^.]+$/, ""), status: "wait",
      off: new Set(), manual: [], seen: false });
  }
  if (imgs.length > room) setStatus(`${room}장만 넣었어요. 한 번에 ${MAX_PHOTOS}장까지예요.`);
  $("stage").style.display = "block";
  if (state.cur < 0) state.cur = first;
  renderStrip(); progress(); render();
  pump();
}

// ---------- 그리기 ----------
async function select(i) {
  state.cur = i;
  const p = cur();
  renderStrip();
  if (!p || p.status !== "done") { render(); return; }
  p.seen = true;
  await canvasOf(p);
  if (p !== cur()) return;   // 그 사이 다른 사진을 눌렀다
  const vw = Math.min(p.W, 1600); view.width = vw; view.height = Math.round(p.H * vw / p.W);
  render();
  renderStrip(); progress();
}

function render(drag) {
  const p = cur();
  $("nav-cur").textContent = state.photos.length ? `${state.cur + 1} / ${state.photos.length}` : "";
  if (!p || p.status !== "done" || !state.cache || state.cache.photo !== p) {
    view.width = 800; view.height = 450;
    ctx.fillStyle = "#111"; ctx.fillRect(0, 0, view.width, view.height);
    ctx.fillStyle = "#8c8c8c"; ctx.font = "24px ui-monospace, monospace"; ctx.textAlign = "center";
    ctx.fillText(p && p.status === "error" ? "이 사진은 열 수 없어요" : "찾는 중…", view.width / 2, view.height / 2);
    ctx.textAlign = "start";
    renderList([]);
    return;
  }
  const s = view.width / p.W;
  ctx.drawImage(state.cache.canvas, 0, 0, view.width, view.height);
  const bars = autoBars(p);
  ctx.lineWidth = Math.max(2, view.width / 400);
  for (const b of bars) {
    const [x1, y1, x2, y2] = b.box.map((v) => v * s);
    if (b.on) { ctx.fillStyle = "#000"; ctx.fillRect(x1, y1, x2 - x1, y2 - y1); }
    else {
      ctx.setLineDash([8, 6]); ctx.strokeStyle = "#ffd400"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.setLineDash([]);
    }
  }
  ctx.fillStyle = "#000";
  for (const m of p.manual) { const [x1, y1, x2, y2] = m.map((v) => v * s); ctx.fillRect(x1, y1, x2 - x1, y2 - y1); }
  if (drag) {
    ctx.fillStyle = "rgba(0,0,0,.6)";
    const [x1, y1, x2, y2] = drag.map((v) => v * s); ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  }
  renderList(bars);
}

function renderList(bars) {
  const ul = $("list"); ul.innerHTML = "";
  const p = cur();
  if (!p || p.status !== "done") return;
  const count = (f) => bars.filter(f).length;
  const faces = count((b) => b.label === "얼굴"), guess = count((b) => b.label.startsWith("사람"));
  const screens = count((b) => Object.values(SCREENS).includes(b.label));
  const cars = count((b) => Object.values(VEHICLES).includes(b.label));
  const parts = [`얼굴 ${faces}개`, state.cover === "body" ? `사람 ${guess}명` : `얼굴 못 찾은 사람 ${guess}명`];
  if (state.screens) parts.push(`화면 ${screens}개`);
  if (state.vehicles) parts.push(`차량 ${cars}대`);
  const li = (t) => { const e = document.createElement("li"); e.textContent = t; ul.appendChild(e); };
  li(`자동으로 찾은 것: ${parts.join(", ")}` + (p.manual.length ? ` · 직접 그린 막대 ${p.manual.length}개` : "") +
    (bars.some((b) => !b.on) ? " · 노란 점선은 끈 막대" : ""));
  if (guess && state.cover === "face") li("얼굴을 못 찾은 사람은 머리 쪽을 어림해서 가렸어요. 빗나갔는지 확인하세요.");
  if (!bars.length) li("자동으로 찾은 게 없어요. 가릴 곳이 있다면 사진 위를 끌어서 막대를 그리세요.");
  li(`처리 시간 ${p.ms}ms (이 기기 안에서)`);
}

function renderStrip() {
  const strip = $("strip"), many = state.photos.length > 1;
  $("nav").hidden = !many;
  strip.hidden = !many;
  if (!many) return;
  strip.innerHTML = "";
  state.photos.forEach((p, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "thumb";
    b.setAttribute("aria-label", `${i + 1}번 사진 ${p.name}` + (p.seen ? " (확인함)" : ""));
    if (i === state.cur) b.setAttribute("aria-current", "true");
    if (p.thumb) { const im = document.createElement("img"); im.src = p.thumb; im.alt = ""; b.appendChild(im); }
    else b.textContent = p.status === "error" ? "!" : "…";
    if (p.seen) { const m = document.createElement("span"); m.className = "seen"; m.textContent = "✓"; b.appendChild(m); }
    b.addEventListener("click", () => select(i));
    strip.appendChild(b);
  });
  strip.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function progress() {
  const n = state.photos.length;
  if (!n) return;
  const done = state.photos.filter((p) => p.status === "done").length;
  const unseen = state.photos.filter((p) => p.status === "done" && !p.seen).length;
  if (done < n) setStatus(`찾는 중… ${done} / ${n}장`);
  else if (n > 1 && unseen) setStatus(`${n}장 모두 찾았어요. 아직 안 본 사진 ${unseen}장 — 한 장씩 넘겨 확인하세요.`, "ok");
  else setStatus(n > 1 ? `${n}장 모두 확인했어요. 저장하세요.` : "찾기 완료. 막대를 확인하고 저장하세요.", "ok");
  $("saveall").hidden = n < 2;
  $("saveall").textContent = `모두 저장 (${done}장, zip)`;
  $("share").textContent = n > 1 ? `모두 공유 (${done}장)` : "공유";
}

// ---------- 막대 편집 (누르기 = 켜고 끄기, 끌기 = 새 막대) ----------
function toImg(ev) {
  const p = cur(), r = view.getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width * p.W, (ev.clientY - r.top) / r.height * p.H];
}
const ready = () => { const p = cur(); return p && p.status === "done" && state.cache?.photo === p; };
let down = null;
view.addEventListener("pointerdown", (ev) => { if (!ready()) return; view.setPointerCapture(ev.pointerId); down = toImg(ev); });
view.addEventListener("pointermove", (ev) => {
  if (!down) return; const p = toImg(ev);
  render([Math.min(down[0], p[0]), Math.min(down[1], p[1]), Math.max(down[0], p[0]), Math.max(down[1], p[1])]);
});
view.addEventListener("pointerup", (ev) => {
  if (!down) return; const ph = cur(), p = toImg(ev), a = down; down = null;
  const moved = Math.hypot(p[0] - a[0], p[1] - a[1]) / ph.W * view.getBoundingClientRect().width;
  if (moved < 8) {
    // 작게 누름: 눌린 곳의 직접 그린 막대를 지우거나, 자동 막대를 켜고 끈다 (위에 그린 것부터)
    const inside = (b) => p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];
    const mi = ph.manual.findLastIndex(inside);
    if (mi >= 0) ph.manual.splice(mi, 1);
    else {
      const hit = autoBars(ph).reverse().find((b) => inside(b.box));
      if (hit) ph.off.has(hit.id) ? ph.off.delete(hit.id) : ph.off.add(hit.id);
    }
  } else {
    ph.manual.push([Math.min(a[0], p[0]), Math.min(a[1], p[1]), Math.max(a[0], p[0]), Math.max(a[1], p[1])]);
  }
  render();
});

// ---------- 저장 ----------
async function outputBlob(p) {
  const src = await canvasOf(p);
  const c = document.createElement("canvas"); c.width = p.W; c.height = p.H;
  const g = c.getContext("2d");
  g.drawImage(src, 0, 0);
  g.fillStyle = "#000";
  for (const [x1, y1, x2, y2] of allOnBoxes(p)) g.fillRect(Math.floor(x1), Math.floor(y1), Math.ceil(x2 - x1) + 1, Math.ceil(y2 - y1) + 1);
  return new Promise((res) => c.toBlob(res, "image/jpeg", 0.92));
}
function outNames() {
  const used = new Map();
  return state.photos.map((p) => {
    const base = `${p.name}_가림`, n = (used.get(base) || 0) + 1; used.set(base, n);
    return n > 1 ? `${base}_${n}.jpg` : `${base}.jpg`;
  });
}
function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
const donePhotos = () => state.photos.map((p, i) => [p, outNames()[i]]).filter(([p]) => p.status === "done");

$("save").addEventListener("click", async () => {
  const p = cur(); if (!ready()) return;
  download(await outputBlob(p), outNames()[state.cur]);
  setStatus("저장했어요. 위치정보 없이 막대가 칠해진 사진입니다.", "ok");
});
$("saveall").addEventListener("click", async () => {
  const list = donePhotos(); if (!list.length) return;
  const files = [];
  for (const [i, [p, name]] of list.entries()) {
    setStatus(`zip 만드는 중… ${i + 1} / ${list.length}`);
    files.push({ name, data: new Uint8Array(await (await outputBlob(p)).arrayBuffer()) });
  }
  download(makeZip(files), `까만막대_${files.length}장.zip`);
  const unseen = list.filter(([p]) => !p.seen).length;
  setStatus(`${files.length}장을 zip으로 저장했어요.` + (unseen ? ` (안 본 사진 ${unseen}장 포함)` : ""), "ok");
});
if (navigator.canShare) {
  const probe = new File([new Blob()], "x.jpg", { type: "image/jpeg" });
  if (navigator.canShare({ files: [probe] })) {
    $("share").hidden = false;
    $("share").addEventListener("click", async () => {
      const list = state.photos.length > 1 ? donePhotos() : donePhotos().filter(([p]) => p === cur());
      const files = [];
      for (const [p, name] of list) files.push(new File([await outputBlob(p)], name, { type: "image/jpeg" }));
      if (!files.length) return;
      if (!navigator.canShare({ files })) { setStatus("이 기기는 여러 장 공유를 지원하지 않아요. '모두 저장'을 쓰세요."); return; }
      try { await navigator.share({ files }); } catch { /* 사용자가 취소 */ }
    });
  }
}

// 압축 없이 묶기만 하는 zip (JPEG는 이미 압축돼 있어 다시 압축해도 거의 안 줄어든다)
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(b) { let c = 0xFFFFFFFF; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
function makeZip(files) {
  const enc = new TextEncoder(), parts = [], central = [];
  let off = 0;
  for (const f of files) {
    const name = enc.encode(f.name), crc = crc32(f.data), n = f.data.length;
    const h = new DataView(new ArrayBuffer(30));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); // UTF-8 이름
    h.setUint16(8, 0, true); h.setUint16(10, 0, true); h.setUint16(12, 0x21, true);           // 저장 방식 0, 1980-01-01
    h.setUint32(14, crc, true); h.setUint32(18, n, true); h.setUint32(22, n, true);
    h.setUint16(26, name.length, true); h.setUint16(28, 0, true);
    parts.push(h, name, f.data);
    const c = new DataView(new ArrayBuffer(46));
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true);
    c.setUint16(10, 0, true); c.setUint16(12, 0, true); c.setUint16(14, 0x21, true);
    c.setUint32(16, crc, true); c.setUint32(20, n, true); c.setUint32(24, n, true);
    c.setUint16(28, name.length, true); c.setUint32(42, off, true);
    central.push(c, name);
    off += 30 + name.length + n;
  }
  const cdSize = central.reduce((s, x) => s + x.byteLength, 0);
  const e = new DataView(new ArrayBuffer(22));
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
  e.setUint32(12, cdSize, true); e.setUint32(16, off, true);
  return new Blob([...parts, ...central, e], { type: "application/zip" });
}

// ---------- 나머지 UI ----------
function setStatus(t, kind) { const s = $("status"); s.textContent = t; s.className = "status" + (kind ? " " + kind : ""); }

$("file").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", async () => {
  const u = b.dataset.sample, blob = await (await fetch(u)).blob();
  addFiles([new File([blob], u.split("/").pop(), { type: blob.type })]);
}));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
document.addEventListener("paste", (e) => {
  const fs = [...e.clipboardData.files].filter((x) => x.type.startsWith("image/"));
  addFiles(fs.map((f) => new File([f], f.name && f.name !== "image.png" ? f.name : "붙여넣은사진.png", { type: f.type })));
});

document.querySelectorAll("[data-cover]").forEach((b) => b.addEventListener("click", () => {
  state.cover = b.dataset.cover;
  document.querySelectorAll("[data-cover]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  render();
}));
$("screens").addEventListener("change", (e) => { state.screens = e.target.checked; render(); });
$("vehicles").addEventListener("change", (e) => { state.vehicles = e.target.checked; render(); });
const sens = $("sens");
const showSens = () => { $("sensv").textContent = Number(sens.value).toFixed(2); };
sens.addEventListener("input", () => { state.sens = Number(sens.value); showSens(); render(); });
showSens();
$("undo").addEventListener("click", () => { const p = cur(); if (p) { p.manual.pop(); render(); } });
$("prev").addEventListener("click", () => { if (state.cur > 0) select(state.cur - 1); });
$("next").addEventListener("click", () => { if (state.cur < state.photos.length - 1) select(state.cur + 1); });
document.addEventListener("keydown", (e) => {
  if (!state.photos.length || e.target.closest("input")) return;
  if (e.key === "ArrowLeft") $("prev").click();
  if (e.key === "ArrowRight") $("next").click();
});
$("reset").addEventListener("click", () => {
  Object.assign(state, { photos: [], cur: -1, cache: null });
  $("stage").style.display = "none"; window.scrollTo({ top: 0, behavior: "smooth" });
  setStatus("준비됐어요. 사진을 고르세요(여러 장 가능).", "ok");
});

if ("serviceWorker" in navigator && location.protocol !== "file:") navigator.serviceWorker.register("sw.js");
loadModels();
