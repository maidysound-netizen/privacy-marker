// 처음 한 번 받은 뒤에는 인터넷 없이 동작하도록 앱 파일·모델·런타임을 저장해 둔다.
const CACHE = "kkaman-v8";
const CORE = ["./", "index.html", "app.js", "manifest.webmanifest", "icons/icon.svg", "icons/icon-192.png",
  "model/yolo26n.onnx", "model/face_detection_yunet_2023mar.onnx", "model/yolo-v9-t-640-license-plates-end2end.onnx", "vendor/ort/ort.min.js", "vendor/ort/ort-wasm-simd-threaded.mjs", "vendor/ort/ort-wasm-simd-threaded.wasm"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(CORE)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  if (u.origin !== location.origin) return;
  // 페이지·코드는 새 버전을 먼저 시도, 모델·런타임(vendor/)은 저장본을 먼저 쓴다
  const networkFirst = !u.pathname.includes("/vendor/") && /\.(html|js|webmanifest)$|\/$/.test(u.pathname);
  e.respondWith(networkFirst
    ? fetch(e.request).then((r) => { const cp = r.clone(); caches.open(CACHE).then((c) => c.put(e.request, cp)); return r; }).catch(() => caches.match(e.request))
    : caches.match(e.request).then((hit) => hit || fetch(e.request).then((r) => { if (r.ok) { const cp = r.clone(); caches.open(CACHE).then((c) => c.put(e.request, cp)); } return r; })));
});
