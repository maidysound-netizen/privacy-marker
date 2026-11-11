// 까만 막대 — 브라우저 안에서 사람·화면을 찾아 까만 막대로 가린다.
// 사진은 서버로 보내지 않는다. 저장은 새 캔버스로 다시 그려서 EXIF가 남지 않는다.
"use strict";

const ORT_VER = "1.20.1"; // vendor/ort 에 들어 있는 버전
const MODEL_URL = "model/yolo26n.onnx";
const SIZE = 640;              // 모델 입력 크기 (export imgsz)
const MIN_SCORE = 0.10;        // 한 번 추론해 두고, 슬라이더는 이 위에서만 거른다
const NMS_IOU = 0.7;           // Ultralytics 기본값
const MAX_SIDE = 4096;         // 아주 큰 사진은 메모리 때문에 이 크기로 줄여 저장
const PERSON = 0;
const SCREENS = { 62: "모니터·TV", 63: "노트북", 67: "휴대폰" };

const $ = (id) => document.getElementById(id);
const view = $("view"), ctx = view.getContext("2d");

const state = {
  session: null,
  img: null, W: 0, H: 0, name: "photo",
  dets: [],          // {id, cls, score, box:[x1,y1,x2,y2]} 원본 좌표
  off: new Set(),    // 사용자가 끈 자동 막대 id
  manual: [],        // 사용자가 그린 막대 [x1,y1,x2,y2]
  cover: "head", screens: true, sens: 0.20,
};

// ---------- 모델 ----------
async function loadModel() {
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", location.href).href;   // onnxruntime-web ${ORT_VER}, 직접 포함
  if (!self.crossOriginIsolated) ort.env.wasm.numThreads = 1;
  try {
    state.session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    setStatus("준비됐어요. 사진을 고르세요.", "ok");
  } catch (e) {
    console.error(e);
    setStatus("모델을 불러오지 못했어요. 인터넷 연결을 확인하고 새로고침하세요.");
  }
}

function letterbox(img, W, H) {
  const r = SIZE / Math.max(W, H);
  const nw = Math.round(W * r), nh = Math.round(H * r);
  const px = Math.floor((SIZE - nw) / 2), py = Math.floor((SIZE - nh) / 2);
  const c = document.createElement("canvas"); c.width = c.height = SIZE;
  const g = c.getContext("2d");
  g.fillStyle = "rgb(114,114,114)"; g.fillRect(0, 0, SIZE, SIZE);
  g.imageSmoothingQuality = "high";
  g.drawImage(img, px, py, nw, nh);
  const d = g.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    x[i] = d[i * 4] / 255; x[n + i] = d[i * 4 + 1] / 255; x[2 * n + i] = d[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", x, [1, 3, SIZE, SIZE]), r, px, py };
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = ix * iy;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i);
}

async function detect(img, W, H) {
  const { tensor, r, px, py } = letterbox(img, W, H);
  const out = await state.session.run({ [state.session.inputNames[0]]: tensor });
  const o = out[state.session.outputNames[0]];
  const [, C, N] = o.dims;              // [1, 84, 8400]: cx,cy,w,h + 80 클래스 점수
  const d = o.data, wanted = [PERSON, ...Object.keys(SCREENS).map(Number)];
  const cand = [];
  for (let i = 0; i < N; i++) {
    let best = -1, bs = 0;
    for (const k of wanted) { const s = d[(4 + k) * N + i]; if (s > bs) { bs = s; best = k; } }
    if (bs < MIN_SCORE) continue;
    // 다른 클래스 점수가 더 높으면 그 물건으로 본다 (예: 사람보다 의자일 확률이 높은 칸)
    let top = 0; for (let k = 0; k < C - 4; k++) top = Math.max(top, d[(4 + k) * N + i]);
    if (top > bs + 1e-6 && best !== PERSON) continue;
    const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
    const box = [(cx - w / 2 - px) / r, (cy - h / 2 - py) / r, (cx + w / 2 - px) / r, (cy + h / 2 - py) / r]
      .map((v, j) => Math.min(Math.max(v, 0), j % 2 ? H : W));
    cand.push({ cls: best, score: bs, box });
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cand) if (keep.every((k) => k.cls !== c.cls || iou(k.box, c.box) < NMS_IOU)) keep.push(c);
  return keep.map((k, id) => ({ id, ...k }));
}

// ---------- 막대 계산 ----------
function headBox([x1, y1, x2, y2]) {
  const w = x2 - x1, h = y2 - y1;
  const hh = Math.min(h, Math.max(0.28 * h, 0.5 * w));   // 서 있는 사람은 위 28%, 상반신만 찍혔으면 더 크게
  const padX = 0.06 * w;
  return [Math.max(0, x1 - padX), Math.max(0, y1 - 0.03 * h), Math.min(state.W, x2 + padX), y1 + hh];
}

function autoBars() {
  return state.dets
    .filter((d) => d.score >= state.sens && (d.cls === PERSON || state.screens))
    .map((d) => ({
      id: d.id, on: !state.off.has(d.id), score: d.score,
      label: d.cls === PERSON ? "사람" : SCREENS[d.cls],
      box: d.cls === PERSON && state.cover === "head" ? headBox(d.box) : d.box,
    }));
}

function allOnBoxes() {
  return [...autoBars().filter((b) => b.on).map((b) => b.box), ...state.manual];
}

// ---------- 그리기 ----------
function render(drag) {
  if (!state.img) return;
  const s = view.width / state.W;
  ctx.drawImage(state.img, 0, 0, view.width, view.height);
  const bars = autoBars();
  ctx.lineWidth = Math.max(2, view.width / 400);
  for (const b of bars) {
    const [x1, y1, x2, y2] = b.box.map((v) => v * s);
    if (b.on) { ctx.fillStyle = "#000"; ctx.fillRect(x1, y1, x2 - x1, y2 - y1); }
    else {
      ctx.setLineDash([8, 6]); ctx.strokeStyle = "#ffd400"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.setLineDash([]);
    }
  }
  ctx.fillStyle = "#000";
  for (const m of state.manual) { const [x1, y1, x2, y2] = m.map((v) => v * s); ctx.fillRect(x1, y1, x2 - x1, y2 - y1); }
  if (drag) {
    ctx.fillStyle = "rgba(0,0,0,.6)";
    const [x1, y1, x2, y2] = drag.map((v) => v * s); ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  }
  renderList(bars);
}

function renderList(bars) {
  const ul = $("list"); ul.innerHTML = "";
  const people = bars.filter((b) => b.label === "사람");
  const summary = document.createElement("li");
  summary.textContent = `자동으로 찾은 것: 사람 ${people.length}명, 화면 ${bars.length - people.length}개` +
    (state.manual.length ? ` · 직접 그린 막대 ${state.manual.length}개` : "") +
    (bars.some((b) => !b.on) ? " · 노란 점선은 끈 막대" : "");
  ul.appendChild(summary);
  if (!bars.length) {
    const li = document.createElement("li");
    li.textContent = "자동으로 찾은 게 없어요. 가릴 곳이 있다면 사진 위를 끌어서 막대를 그리세요.";
    ul.appendChild(li);
  }
}

// ---------- 사진 열기 ----------
async function openBlob(blob, name) {
  if (!state.session) { setStatus("모델이 아직 준비 중이에요. 잠시 후 다시 시도하세요."); return; }
  setStatus("사진을 읽는 중…");
  const url = URL.createObjectURL(blob);
  const el = new Image();
  // decode() 대신 onload: 백그라운드 탭에서도 멈추지 않는다
  const ok = await new Promise((res) => { el.onload = () => res(true); el.onerror = () => res(false); el.src = url; });
  if (!ok) { setStatus("이 사진은 열 수 없어요. JPG·PNG·HEIC(아이폰은 '호환성 우선' 설정) 사진을 써 보세요."); URL.revokeObjectURL(url); return; }
  // 브라우저가 EXIF 방향을 반영해 그려 준다. 너무 큰 사진은 줄인다.
  let W = el.naturalWidth, H = el.naturalHeight;
  const k = Math.min(1, MAX_SIDE / Math.max(W, H));
  W = Math.round(W * k); H = Math.round(H * k);
  const base = document.createElement("canvas"); base.width = W; base.height = H;
  base.getContext("2d").drawImage(el, 0, 0, W, H);
  URL.revokeObjectURL(url);

  Object.assign(state, { img: base, W, H, name: (name || "photo").replace(/\.[^.]+$/, ""), off: new Set(), manual: [] });
  const vw = Math.min(W, 1600); view.width = vw; view.height = Math.round(H * vw / W);
  $("stage").style.display = "block";
  setStatus("사람과 화면을 찾는 중…");
  const t = performance.now();
  state.dets = await detect(base, W, H);
  setStatus(`찾기 완료 (${Math.round(performance.now() - t)}ms). 막대를 확인하고 저장하세요.`, "ok");
  render();
  $("stage").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- 막대 편집 (누르기 = 켜고 끄기, 끌기 = 새 막대) ----------
function toImg(ev) {
  const r = view.getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width * state.W, (ev.clientY - r.top) / r.height * state.H];
}
let down = null;
view.addEventListener("pointerdown", (ev) => { if (!state.img) return; view.setPointerCapture(ev.pointerId); down = toImg(ev); });
view.addEventListener("pointermove", (ev) => {
  if (!down) return; const p = toImg(ev);
  render([Math.min(down[0], p[0]), Math.min(down[1], p[1]), Math.max(down[0], p[0]), Math.max(down[1], p[1])]);
});
view.addEventListener("pointerup", (ev) => {
  if (!down) return; const p = toImg(ev), a = down; down = null;
  const moved = Math.hypot(p[0] - a[0], p[1] - a[1]) / state.W * view.getBoundingClientRect().width;
  if (moved < 8) {
    // 작게 누름: 눌린 곳의 직접 그린 막대를 지우거나, 자동 막대를 켜고 끈다 (위에 그린 것부터)
    const inside = (b) => p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];
    const mi = state.manual.findLastIndex(inside);
    if (mi >= 0) state.manual.splice(mi, 1);
    else {
      const hit = autoBars().reverse().find((b) => inside(b.box));
      if (hit) state.off.has(hit.id) ? state.off.delete(hit.id) : state.off.add(hit.id);
    }
  } else {
    state.manual.push([Math.min(a[0], p[0]), Math.min(a[1], p[1]), Math.max(a[0], p[0]), Math.max(a[1], p[1])]);
  }
  render();
});

// ---------- 저장 ----------
function outputBlob() {
  const c = document.createElement("canvas"); c.width = state.W; c.height = state.H;
  const g = c.getContext("2d");
  g.drawImage(state.img, 0, 0);
  g.fillStyle = "#000";
  for (const [x1, y1, x2, y2] of allOnBoxes()) g.fillRect(Math.floor(x1), Math.floor(y1), Math.ceil(x2 - x1) + 1, Math.ceil(y2 - y1) + 1);
  return new Promise((res) => c.toBlob(res, "image/jpeg", 0.92));
}
const outName = () => `${state.name}_가림.jpg`;

$("save").addEventListener("click", async () => {
  const blob = await outputBlob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = outName(); a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  setStatus("저장했어요. 위치정보 없이 막대가 칠해진 사진입니다.", "ok");
});
if (navigator.canShare) {
  const probe = new File([new Blob()], "x.jpg", { type: "image/jpeg" });
  if (navigator.canShare({ files: [probe] })) {
    $("share").hidden = false;
    $("share").addEventListener("click", async () => {
      const file = new File([await outputBlob()], outName(), { type: "image/jpeg" });
      try { await navigator.share({ files: [file] }); } catch { /* 사용자가 취소 */ }
    });
  }
}

// ---------- 나머지 UI ----------
function setStatus(t, kind) { const s = $("status"); s.textContent = t; s.className = "status" + (kind ? " " + kind : ""); }

$("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) openBlob(f, f.name); e.target.value = ""; });
document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", async () => {
  const u = b.dataset.sample; openBlob(await (await fetch(u)).blob(), u.split("/").pop());
}));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) openBlob(f, f.name); });
document.addEventListener("paste", (e) => {
  const f = [...e.clipboardData.files].find((x) => x.type.startsWith("image/")); if (f) openBlob(f, "붙여넣은사진");
});

document.querySelectorAll("[data-cover]").forEach((b) => b.addEventListener("click", () => {
  state.cover = b.dataset.cover;
  document.querySelectorAll("[data-cover]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  render();
}));
$("screens").addEventListener("change", (e) => { state.screens = e.target.checked; render(); });
const sens = $("sens");
const showSens = () => { $("sensv").textContent = Number(sens.value).toFixed(2); };
sens.addEventListener("input", () => { state.sens = Number(sens.value); showSens(); render(); });
showSens();
$("undo").addEventListener("click", () => { state.manual.pop(); render(); });
$("reset").addEventListener("click", () => {
  Object.assign(state, { img: null, dets: [], manual: [], off: new Set() });
  $("stage").style.display = "none"; window.scrollTo({ top: 0, behavior: "smooth" });
  setStatus("준비됐어요. 사진을 고르세요.", "ok");
});

if ("serviceWorker" in navigator && location.protocol !== "file:") navigator.serviceWorker.register("sw.js");
loadModel();
