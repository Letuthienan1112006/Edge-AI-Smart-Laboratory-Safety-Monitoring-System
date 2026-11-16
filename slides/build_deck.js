// Slide deck for the 14/11/2026 final (5 min talk + 5 min demo): 5 slides, Vietnamese.
// Build:  NODE_PATH=<dir with pptxgenjs> node slides/build_deck.js   ->  slides/chung_khao_edge_ai_lab_safety.pptx
// Every number comes from docs/competition-claims.md / report/bai_bao_ngan.tex; images from poster/img.
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = process.env.PPTX_SKILL_DIR;
const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));

const ROOT = path.resolve(__dirname, "..");
const IMG = (f) => path.join(ROOT, "poster", "img", f);
const SIMG = (f) => path.join(__dirname, "img", f);
const OUT = path.join(__dirname, "chung_khao_edge_ai_lab_safety.pptx");

const THEME = {
  name: "EdgeAI-Lab",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "16324F", lt2: "EDF3F6",
    accent1: "0E7C86", accent2: "2E9E4F", accent3: "E0A100", accent4: "C62828", accent5: "6B7280", accent6: "3B6EA8",
    hlink: "0E7C86", folHlink: "6B7280",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Edge AI Smart Laboratory Safety Monitoring System";
pres.author = "Lê Tự Thiện An; Trịnh Anh Văn; Võ Nhật Nam; Phạm Trọng Thành";
const C = pres.SchemeColor;

// ---- layouts ----
pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text2 },
  objects: [],
  slideNumber: { x: 9.2, y: 5.25, w: 0.5, h: 0.3, fontSize: 10, color: C.background2 },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [],
  slideNumber: { x: 9.2, y: 5.25, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5 },
});
// title placeholders are filled by position on each slide (same x/y/size everywhere)
const TITLE = { x: 0.5, y: 0.3, w: 9, h: 0.7, fontFace: "Cambria", fontSize: 30, bold: true, valign: "middle", margin: 0 };

// ---- helpers ----
const card = (s, name, x, y, w, h, fill) =>
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name });

function chartBase(extra) {
  return Object.assign({
    barDir: "col", showTitle: true, titleFontSize: 12, titleColor: "16324F", titleFontFace: "+mn-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: "1B2430", dataLabelFontFace: "+mn-lt",
    showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelColor: "1B2430", catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt", barGapWidthPct: 60,
  }, extra);
}

// helpers for flow diagrams (boxes + arrows), defined once
function flowBox(s, name, x, y, w, h, fill, lines, color, size) {
  card(s, name, x, y, w, h, fill);
  s.addText(lines, { x, y, w, h, fontSize: size || 10, color: color || "1B2430", align: "center", valign: "middle", margin: 2, isTextBox: true, objectName: name + "Text" });
}
function flowArrow(s, name, x1, y1, x2, y2, color) {
  s.addShape(pres.ShapeType.line, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001, flipH: x2 < x1, flipV: y2 < y1,
    line: { color: color || "16324F", width: 1.5, endArrowType: "triangle" }, objectName: name });
}
function rowLabel(s, name, text, y, h, color) {
  s.addText(text, { x: 0.5, y, w: 1.15, h, fontSize: 12, bold: true, color, valign: "middle", margin: 0, isTextBox: true, objectName: name });
}

// =====================================================================
// F0. AI compression / optimisation pipeline (main deck, slide 2) — plain technical drawing: white boxes, thin dark lines, one accent
// =====================================================================
function compressSlide() {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Luồng nén và tối ưu mô hình AI", Object.assign({ color: C.text2, isTextBox: true, objectName: "Title" }, TITLE));
  const INK = "1F2937", LINE = "4B5563", MUTED = "6B7280", ACC = "B4441F";
  const box = (name, x, y, w, h, head, sub, num) => {
    s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: "FFFFFF" }, line: { color: LINE, width: 1 }, objectName: name });
    s.addText([
      { text: head, options: { bold: true, color: INK, fontSize: 11.5, breakLine: true } },
      { text: sub, options: { color: MUTED, fontSize: 9.5, breakLine: !!num } },
      ...(num ? [{ text: num, options: { bold: true, color: ACC, fontSize: 11 } }] : []),
    ], { x: x + 0.06, y, w: w - 0.12, h, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name + "Text" });
  };
  const arrow = (name, x1, y1, x2, y2, dash) => s.addShape(pres.ShapeType.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001, flipH: x2 < x1, flipV: y2 < y1,
    line: { color: LINE, width: 1.25, endArrowType: "triangle", dashType: dash ? "dash" : "solid" }, objectName: name });
  const label = (name, text, x, y, w) => s.addText(text, { x, y, w, h: 0.28, fontSize: 11, bold: true, color: INK, margin: 0, isTextBox: true, objectName: name });

  // ---- row 1: offline, on the laptop ----
  label("LabelOffline", "A · Chuẩn bị mô hình (một lần, trên máy tính)", 0.5, 1.02, 6);
  const W = 1.66, G = 0.17, Y1 = 1.35, H1 = 1.12;
  const xs = [0, 1, 2, 3, 4].map((i) => 0.5 + i * (W + G));
  box("S1", xs[0], Y1, W, H1, "1. Huấn luyện", "YOLO26n FP32\nảnh lab + 238 COCO", "mAP50-95 0,661");
  box("S2", xs[1], Y1, W, H1, "2. Tách đầu ra", "6 tensor: khung và\nđiểm lớp mỗi tầng", "");
  box("S3", xs[2], Y1, W, H1, "3. Nén INT8", "RKNN-Toolkit2\n300 ảnh hiệu chuẩn", "6 thang riêng");
  box("S4", xs[3], Y1, W, H1, "4. Kiểm tra", "so NPU với CPU\ntrên cùng bộ ảnh", "bắt 4 lỗi");
  box("S5", xs[4], Y1, W, H1, "5. Lên NPU", "INT8 trên\nOrange Pi 5", "mAP50-95 0,657");
  for (let i = 0; i < 4; i++) arrow("A" + (i + 1), xs[i] + W, Y1 + H1 / 2, xs[i + 1], Y1 + H1 / 2);

  // why split: one line comparison under row 1
  s.addShape(pres.ShapeType.line, { x: 0.5, y: 2.68, w: 9.0, h: 0, line: { color: "D1D5DB", width: 0.75 }, objectName: "Rule1" });
  s.addText([
    { text: "Vì sao tách đầu ra: ", options: { bold: true, color: INK } },
    { text: "xuất mặc định gộp tọa độ (0–640 px) và điểm lớp (0–1) vào một thang 8 bit, mỗi bước ≈ 2,5 px nên mép khung bị lệch: mAP50-95 " },
    { text: "0,481 (mất 18 điểm)", options: { bold: true, color: ACC } },
    { text: ". Tách 6 tensor, mỗi tensor một thang: " },
    { text: "0,657 (mất 0,4 điểm)", options: { bold: true, color: ACC } },
    { text: "." },
  ], { x: 0.5, y: 2.74, w: 9.0, h: 0.55, fontSize: 11, color: "374151", margin: 0, valign: "top", isTextBox: true, objectName: "WhySplit" });

  // ---- row 2: on the board, every frame ----
  label("LabelOnline", "B · Chạy trên bo mạch (mỗi khung hình)", 0.5, 3.38, 6);
  const W2 = 1.66, Y2 = 3.7, H2 = 0.82;
  const xs2 = [0, 1, 2, 3, 4].map((i) => 0.5 + i * (W2 + G));
  box("R1", xs2[0], Y2, W2, H2, "Camera", "lấy ảnh mới nhất", "");
  box("R2", xs2[1], Y2, W2, H2, "NPU lõi 0 · INT8", "người + áo blouse", "≈25 ms/khung");
  box("R3", xs2[2], Y2, W2, H2, "CPU", "giải mã, theo vết,\nđiểm chân trong vùng", "");
  box("R4", xs2[3], Y2, W2, H2, "Xác nhận 3 s", "chỉ khi thấy rõ\ntrang phục", "");
  box("R5", xs2[4], Y2, W2, H2, "Cảnh báo", "web · LED · OLED · Zalo", "");
  for (let i = 0; i < 4; i++) arrow("B" + (i + 1), xs2[i] + W2, Y2 + H2 / 2, xs2[i + 1], Y2 + H2 / 2);

  // results line
  s.addText([
    { text: "Kết quả: ", options: { bold: true, color: INK } },
    { text: "CPU FP32 10,5 FPS, 86,8 °C  →  NPU INT8 " },
    { text: "≈30 FPS", options: { bold: true, color: ACC } },
    { text: " (bằng camera), p99 41,4 ms, " },
    { text: "2,2 W", options: { bold: true, color: ACC } },
    { text: ". Mô hình phụ chỉ chạy khi cần trên lõi NPU riêng: 3,34 → 1,98 W." },
  ], { x: 0.5, y: 4.7, w: 9.0, h: 0.5, fontSize: 11, color: "374151", margin: 0, valign: "top", isTextBox: true, objectName: "Result" });

  s.addNotes(
    "[0:40–1:50] Đây là luồng nén và tối ưu mô hình AI của nhóm, gồm hai phần.\n" +
    "Phần A làm một lần trên máy tính: huấn luyện YOLO26n trên ảnh phòng thí nghiệm, thêm 238 ảnh COCO người mặc áo trắng thường để mô hình không nhầm áo thun trắng; xuất mô hình với đầu ra tách thành 6 tensor; lượng tử hóa INT8 bằng RKNN-Toolkit2 với 300 ảnh hiệu chuẩn; rồi so kết quả NPU với CPU trên cùng bộ ảnh trước khi dùng. Bước kiểm tra này bắt được 4 lỗi chuyển đổi mà nhìn tốc độ thì không thấy.\n" +
    "Điểm mấu chốt là bước tách đầu ra. Nếu xuất theo mặc định, tọa độ khung và điểm lớp dùng chung một thang 8 bit, mỗi bước khoảng 2,5 điểm ảnh, nên mép khung bị lệch và mAP50-95 tụt từ 0,661 xuống 0,481. Tách ra thì còn 0,657, gần như bằng bản gốc.\n" +
    "Phần B chạy trên bo mạch mỗi khung: lấy ảnh mới nhất, NPU suy luận khoảng 25 ms, CPU giải mã, theo vết và xét điểm chân, xác nhận 3 giây rồi cảnh báo. Kết quả: từ 10,5 khung/giây trên CPU lên khoảng 30 khung/giây, bằng tốc độ camera, chỉ 2,2 W."
  );
}

// =====================================================================
// F1. Processing flow: usual vs ours (main deck, slide 2)
// =====================================================================
function flowSlide() {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Giải pháp và luồng xử lý", Object.assign({ color: C.text2, isTextBox: true, objectName: "Title" }, TITLE));
  const W = 1.2, G = 0.14, X0 = 1.65;
  const xs = [0, 1, 2, 3, 4, 5].map((i) => X0 + i * (W + G));
  // usual
  rowLabel(s, "LabelUsual", "Thông thường", 1.15, 0.9, "C62828");
  const usual = ["Camera\n(đọc cả ảnh cũ)", "YOLO trên CPU\nFP32", "Chạy mọi\nmô hình mỗi khung", "Không thấy áo\n= vi phạm", "Báo ngay\nmỗi khung", "Gửi video\nlên máy chủ"];
  usual.forEach((t, i) => flowBox(s, "U" + (i + 1), xs[i], 1.15, W, 0.9, "F6E3E3", t, "5B1A1A", 9.5));
  for (let i = 0; i < 5; i++) flowArrow(s, "UA" + (i + 1), xs[i] + W, 1.6, xs[i + 1], 1.6, "C62828");
  s.addText("10,5 FPS · 86,8 °C · báo nhầm khi người bị che · lộ hình người", { x: X0, y: 2.08, w: 7.85, h: 0.3, fontSize: 11, color: "C62828", bold: true, margin: 0, isTextBox: true, objectName: "UsualResult" });
  // ours
  rowLabel(s, "LabelOurs", "Của nhóm", 2.65, 0.9, "2E9E4F");
  const ours = ["Camera\nảnh mới nhất", "NPU INT8\nđầu ra tách", "Theo vết +\ngán áo đúng người", "Điểm chân trong vùng\n+ thấy rõ trang phục", "Xác nhận 3 s,\nnghỉ 60 s", "Web, LED, OLED,\nZalo (chỉ chữ)"];
  ours.forEach((t, i) => flowBox(s, "O" + (i + 1), xs[i], 2.65, W, 0.9, i === 1 || i === 3 ? "0E7C86" : "E3F2EA", t, i === 1 || i === 3 ? "FFFFFF" : "17402A", 9.5));
  for (let i = 0; i < 5; i++) flowArrow(s, "OA" + (i + 1), xs[i] + W, 3.1, xs[i + 1], 3.1, "2E9E4F");
  // async side stages
  flowBox(s, "Async", xs[2], 3.85, W * 2 + G, 0.55, "EDF3F6", "Tầng B, C: lõi NPU riêng, chỉ khi cần (thích nghi)", "16324F", 10);
  flowArrow(s, "AsyncA", xs[1] + W / 2, 3.55, xs[2], 4.12, "6B7280");
  s.addText("≈30 FPS (bằng camera) · p99 41,4 ms · 2,2 W · 31 → 6 cảnh báo, cả 6 đúng · hình người ở lại trên bo mạch", { x: X0, y: 4.55, w: 7.8, h: 0.5, fontSize: 11, color: "2E9E4F", bold: true, margin: 0, valign: "top", isTextBox: true, objectName: "OursResult" });
  s.addNotes(
    "[0:40–1:50] Hàng trên là cách làm thường gặp: đọc camera không chờ ảnh mới nên xử lý lại ảnh cũ, chạy YOLO trên CPU, chạy mọi mô hình ở mọi khung, cứ không thấy áo là báo vi phạm và báo ngay từng khung, hoặc gửi video lên máy chủ. Trên bo mạch này cách đó chỉ đạt 10,5 khung/giây và CPU nóng 86,8 độ.\n" +
    "Hàng dưới là luồng của nhóm. Hai ô màu đậm là hai điểm chính: mô hình INT8 với đầu ra tách chạy trên NPU, và luật chỉ kết luận khi chân ở trong vùng và nhìn thấy rõ trang phục. Thêm vào đó: lấy ảnh mới nhất, theo vết và gán áo đúng người, xác nhận 3 giây, và các mô hình phụ chỉ chạy khi cần trên nhân NPU riêng.\n" +
    "Kết quả: khoảng 30 khung/giây, bằng tốc độ camera; độ trễ p99 41,4 ms; 2,2 W; trên video đông người số cảnh báo giảm từ 31 xuống 6 và cả 6 đều đúng."
  );
}

// =====================================================================
// F2. INT8 compression: usual export vs split outputs (appendix)
// =====================================================================
function quantSlide() {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Phụ lục 4 · Nén INT8 tách đầu ra", Object.assign({ color: C.text2, isTextBox: true, objectName: "Title" }, TITLE));
  const W = 1.65, G = 0.22, X0 = 1.75;
  const xs = [0, 1, 2, 3].map((i) => X0 + i * (W + G));
  rowLabel(s, "QLabelUsual", "Xuất mặc định", 1.15, 1.0, "C62828");
  const u = [
    "Mô hình FP32\nmAP50-95 0,661",
    "Gộp 1 tensor:\ntọa độ 0–640 px\n+ điểm lớp 0–1",
    "1 thang 8 bit chung\nbước ≈ 640/255\n≈ 2,5 px",
    "Mép khung lệch 2,7 px\nmAP50-95 0,481\n(mất 18 điểm)",
  ];
  u.forEach((t, i) => flowBox(s, "QU" + (i + 1), xs[i], 1.15, W, 1.0, i === 3 ? "C62828" : "F6E3E3", t, i === 3 ? "FFFFFF" : "5B1A1A", 10));
  for (let i = 0; i < 3; i++) flowArrow(s, "QUA" + (i + 1), xs[i] + W, 1.65, xs[i + 1], 1.65, "C62828");
  rowLabel(s, "QLabelOurs", "Nhóm em", 2.6, 1.0, "2E9E4F");
  const o = [
    "Mô hình FP32\n(cùng trọng số)",
    "Tách 6 tensor:\nkhung và điểm lớp\ncủa 3 tầng đặc trưng",
    "Mỗi tensor 1 thang\nriêng, NPU chạy INT8;\nCPU giải mã + NMS",
    "mAP50-95 0,657\n(mất 0,4 điểm)\nIoU với FP32 0,97",
  ];
  o.forEach((t, i) => flowBox(s, "QO" + (i + 1), xs[i], 2.6, W, 1.0, i === 3 ? "2E9E4F" : "E3F2EA", t, i === 3 ? "FFFFFF" : "17402A", 10));
  for (let i = 0; i < 3; i++) flowArrow(s, "QOA" + (i + 1), xs[i] + W, 3.1, xs[i + 1], 3.1, "2E9E4F");
  card(s, "GateCard", X0, 3.85, 7.3, 0.85, "EDF3F6");
  s.addText([
    { text: "Cổng kiểm tra trước khi dùng: ", options: { bold: true, color: "16324F" } },
    { text: "chạy cùng ảnh qua bản CPU (ONNX) và bản NPU, so khung (recall, precision, IoU). Bước này bắt được 4 lỗi chuyển đổi mà tốc độ vẫn bình thường, trong đó có lỗi tọa độ INT8 bị chia cho kích thước ảnh (49 phát hiện, 0 khớp)." },
  ], { x: X0 + 0.15, y: 3.88, w: 7.0, h: 0.8, fontSize: 11, color: "1B2430", margin: 0, valign: "middle", isTextBox: true, objectName: "GateText" });
  s.addText("Đã thử và không giúp: hiệu chuẩn KL, lượng tử hóa hỗn hợp, đổi bộ ảnh hiệu chuẩn. Số đo trên NPU thật, tập kiểm tra góc 1, 3 lần huấn luyện.", { x: X0, y: 4.8, w: 7.3, h: 0.4, fontSize: 10, color: C.accent5, margin: 0, isTextBox: true, objectName: "QNote" });
  s.addNotes("Phụ lục: giải thích nén INT8. Đầu ra mặc định gộp tọa độ khung (0 đến 640 điểm ảnh) và điểm lớp (0 đến 1) vào một tensor dùng chung một thang 8 bit, nên mỗi bước lượng tử khoảng 2,5 điểm ảnh, mép khung bị làm tròn, mAP50-95 tụt 18 điểm. Tách thành 6 tensor, mỗi tensor một thang, thì gần như giữ nguyên chất lượng. Trước khi dùng luôn chạy cổng kiểm tra so NPU với CPU.");
}

// =====================================================================
// 1. Title (dark)
// =====================================================================
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addText("EDGE AI DESIGN CHALLENGE 2026 · CHUNG KẾT · ĐỀ TÀI", { x: 0.5, y: 0.35, w: 9, h: 0.3, fontSize: 12, color: C.background2, charSpacing: 2, margin: 0, isTextBox: true, objectName: "Kicker" });
  s.addText("Edge AI Smart Laboratory Safety Monitoring System", { x: 0.5, y: 0.85, w: 5.3, h: 1.5, fontFace: "Cambria", fontSize: 30, bold: true, color: C.background1, valign: "top", margin: 0, isTextBox: true, objectName: "Title" });
  s.addText("Hệ thống AI biên giám sát an toàn phòng thí nghiệm trên Orange Pi 5", { x: 0.5, y: 2.4, w: 5.3, h: 0.6, fontSize: 16, color: C.background2, margin: 0, isTextBox: true, objectName: "Subtitle" });
  card(s, "MessageCard", 0.5, 3.1, 5.3, 0.9, C.accent1);
  s.addText([
    { text: "Báo động khi người ", options: {} },
    { text: "không mặc áo blouse", options: { bold: true } },
    { text: " bước vào vùng nguy hiểm: xử lý tại chỗ trên NPU, không gửi hình ra ngoài.", options: {} },
  ], { x: 0.65, y: 3.15, w: 5.0, h: 0.8, fontSize: 14, color: C.background1, valign: "middle", margin: 0, isTextBox: true, objectName: "Message" });
  s.addText([
    { text: "Lê Tự Thiện An (nhóm trưởng)", options: { breakLine: true, bold: true } },
    { text: "Trịnh Anh Văn · Võ Nhật Nam", options: { breakLine: true } },
    { text: "Phạm Trọng Thành", options: {} },
  ], { x: 0.5, y: 4.2, w: 5.4, h: 0.95, fontSize: 12, color: C.background2, margin: 0, isTextBox: true, valign: "top", objectName: "Team" });
  s.addImage({ path: IMG("hardware_full.jpg"), x: 6.1, y: 1.1, w: 3.5, h: 3.5 * 854 / 1300, altText: "Thiết bị: Orange Pi 5 trong vỏ in 3D với camera, đèn LED và màn OLED" });
  s.addText("Thiết bị demo: Orange Pi 5, camera C270, 3 đèn LED, màn OLED", { x: 6.1, y: 3.5, w: 3.5, h: 0.5, fontSize: 11, color: C.background2, margin: 0, isTextBox: true, objectName: "PhotoCaption" });
  s.addNotes(
    "[0:00–0:40] Xin chào hội đồng. Nhóm em làm một thiết bị giám sát an toàn phòng thí nghiệm trên bo mạch Orange Pi 5: nó nhìn camera và báo động khi có người KHÔNG mặc áo blouse bước vào vùng nguy hiểm.\nBa điểm chính: toàn bộ nhận diện chạy tại chỗ trên NPU nên hình người không gửi ra ngoài; chỉ tin nhắn Zalo cần Internet; và mọi số liệu em trình bày sau đây đều đo trên thiết bị thật."
  );
}

compressSlide();

// =====================================================================
// M3. Product achieved
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Sản phẩm đạt được", Object.assign({ color: C.text2, isTextBox: true, objectName: "Title" }, TITLE));
  s.addImage({ path: IMG("hardware_full.jpg"), x: 0.5, y: 1.1, w: 2.9, h: 2.9 * 854 / 1300, altText: "Thiết bị Orange Pi 5 trong vỏ in 3D với camera, LED, OLED" });
  s.addText("Thiết bị: Orange Pi 5, camera C270, 3 LED, màn OLED, vỏ in 3D", { x: 0.5, y: 3.05, w: 2.9, h: 0.4, fontSize: 10, color: C.accent5, margin: 0, isTextBox: true, objectName: "HwCaption" });
  s.addImage({ path: IMG("dashboard.png"), x: 3.6, y: 1.1, w: 3.7, h: 3.7 * 952 / 1500, altText: "Trang web giám sát khi có báo động" });
  s.addText("Trang web giám sát (tiếng Việt) khi báo động", { x: 3.6, y: 3.48, w: 3.7, h: 0.3, fontSize: 10, color: C.accent5, margin: 0, isTextBox: true, objectName: "WebCaption" });
  s.addImage({ path: IMG("zalo_crop.png"), x: 7.5, y: 1.1, w: 2.0, h: 2.0 * 338 / 392, altText: "Tin nhắn Zalo cảnh báo" });
  s.addText("Tin nhắn Zalo tới người phụ trách", { x: 7.5, y: 2.88, w: 2.0, h: 0.4, fontSize: 10, color: C.accent5, margin: 0, isTextBox: true, objectName: "ZaloCaption" });
  const feats = [
    ["Phát hiện người và áo blouse", "YOLO26n INT8 trên NPU, mọi khung hình"],
    ["Vùng nguy hiểm tự vẽ", "vẽ ngay trên trang web khi đổi chỗ camera"],
    ["Cảnh báo nhiều kênh", "web kèm ảnh bằng chứng, đèn LED, màn OLED, Zalo"],
    ["4 trạng thái an toàn", "AN TOÀN · CHÚ Ý · BÁO ĐỘNG · MẤT TÍN HIỆU"],
    ["Tự chạy khi cấp điện", "tự phục hồi khi mất camera, không cần mạng"],
  ];
  feats.forEach((f, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.5 + col * 3.05, y = 3.9 + row * 0.68, w = 2.9;
    card(s, "Feat" + (i + 1), x, y, w, 0.6, C.background2);
    s.addText([{ text: f[0], options: { bold: true, color: "16324F", breakLine: true } }, { text: f[1], options: { fontSize: 9.5 } }],
      { x: x + 0.1, y, w: w - 0.2, h: 0.6, fontSize: 11, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: "FeatText" + (i + 1) });
  });
  s.addNotes(
    "[1:50–2:50] Đây là sản phẩm nhóm đã làm ra. Bên trái là thiết bị: bo mạch Orange Pi 5 trong vỏ in 3D, camera, ba đèn LED và màn OLED. Ở giữa là trang web giám sát tiếng Việt lúc có báo động: video có khung, số người, ảnh bằng chứng người vi phạm. Bên phải là tin nhắn Zalo gửi tới người phụ trách.\n" +
    "Các tính năng: phát hiện người và áo blouse trên NPU; vẽ vùng nguy hiểm ngay trên web khi đổi chỗ camera; cảnh báo qua web, đèn, OLED và Zalo; bốn trạng thái, trong đó mất hình thì không bao giờ báo an toàn; tự chạy khi cắm điện và tự phục hồi khi mất camera. Chỉ tin Zalo cần Internet."
  );
}

// =====================================================================
// M4. Performance
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Hiệu năng đo trên thiết bị thật", Object.assign({ color: C.text2, isTextBox: true, objectName: "Title" }, TITLE));
  const stats = [["≈30 FPS", "bằng tốc độ camera"], ["41,4 ms", "độ trễ p99"], ["2,2 W", "công suất khi chạy"], ["24 giờ", "chạy liên tục, 0 lỗi"]];
  stats.forEach((st, i) => {
    const x = 0.5 + i * 2.3;
    card(s, "Stat" + (i + 1), x, 1.1, 2.1, 0.95, C.background2);
    s.addText(st[0], { x, y: 1.13, w: 2.1, h: 0.52, fontFace: "Cambria", fontSize: 26, bold: true, color: C.accent1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "StatValue" + (i + 1) });
    s.addText(st[1], { x, y: 1.65, w: 2.1, h: 0.35, fontSize: 11, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "StatLabel" + (i + 1) });
  });
  s.addChart(pres.charts.BAR, [{ name: "mAP50-95", labels: ["FP32", "INT8 mặc định", "INT8 đầu ra tách"], values: [0.661, 0.481, 0.657] }],
    chartBase({ x: 0.5, y: 2.2, w: 3.0, h: 2.1, title: "Độ chính xác sau nén INT8 (mAP50-95)", chartColors: ["16324F", "C62828", "2E9E4F"], dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 0.8, catAxisLabelFontSize: 9 }));
  s.addChart(pres.charts.BAR, [{ name: "W", labels: ["Nghỉ", "Thích nghi", "Song song"], values: [0.86, 1.98, 3.34] }],
    chartBase({ x: 3.6, y: 2.2, w: 2.8, h: 2.1, title: "Công suất bo mạch (W)", chartColors: ["6B7280", "2E9E4F", "C62828"], dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 4.2, catAxisLabelFontSize: 9 }));
  s.addChart(pres.charts.BAR, [
    { name: "Phát hiện người không áo", labels: ["Góc 1", "Góc 2"], values: [81, 88] },
    { name: "Báo nhầm người có áo", labels: ["Góc 1", "Góc 2"], values: [8, 10] },
  ], chartBase({ x: 6.5, y: 2.2, w: 3.0, h: 2.1, title: "Luật vùng (%)", chartColors: ["2E9E4F", "C62828"], showLegend: true, legendPos: "b", legendFontSize: 9, legendColor: "1B2430", legendFontFace: "+mn-lt", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 100, catAxisLabelFontSize: 9 }));
  s.addText([
    { text: "Tối ưu chính: ", options: { bold: true, color: "16324F" } },
    { text: "chuyển từ CPU (10,5 FPS, 86,8 °C) sang NPU INT8; tách đầu ra giữ chất lượng (mất 0,4 thay vì 18 điểm); chế độ thích nghi giảm 41% điện so với chạy song song." },
  ], { x: 0.5, y: 4.4, w: 9.0, h: 0.5, fontSize: 11, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "PerfTakeaway" });
  s.addText("Tốc độ: camera tối đa 30 FPS (đo chuẩn có thêm tầng B, C: 29,02). Công suất: đồng hồ USB-C, ±10%, video phát lại. Luật vùng: INT8 trên NPU, theo từng người, TB 3 lần huấn luyện; còn bỏ sót.", { x: 0.5, y: 4.9, w: 8.6, h: 0.4, fontSize: 9, color: C.accent5, margin: 0, valign: "top", isTextBox: true, objectName: "PerfNote" });
  s.addNotes(
    "[2:50–4:00] Hiệu năng, tất cả đo trên bo mạch thật. Hệ thống chạy khoảng 30 khung hình mỗi giây, bằng tốc độ tối đa của camera; độ trễ p99 41,4 mili giây; công suất 2,2 W; chạy liên tục 24 giờ không lỗi.\n" +
    "Biểu đồ trái: nén INT8 theo cách mặc định làm mAP50-95 tụt từ 0,661 xuống 0,481; nhóm tách đầu ra thành 6 tensor nên còn 0,657. Biểu đồ giữa: chế độ thích nghi tiêu thụ 1,98 W so với 3,34 W khi chạy song song mọi mô hình. Biểu đồ phải: phát hiện 81 đến 88% người không mặc áo, báo nhầm 8 đến 10%.\n" +
    "Em nói rõ giới hạn: vẫn còn bỏ sót nên đây là công cụ hỗ trợ, mỗi cảnh báo kèm ảnh để người xác nhận; mô hình nhạy với góc camera nên phải đặt camera đúng góc và kiểm tra trước khi dùng."
  );
}

// =====================================================================
// M5. Conclusion + demo (dark)
// =====================================================================
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addText("Kết luận và demo", Object.assign({ color: C.background1, isTextBox: true, objectName: "Title" }, TITLE));
  s.addText("Kết luận", { x: 0.5, y: 1.15, w: 4.3, h: 0.35, fontSize: 15, bold: true, color: C.background1, margin: 0, isTextBox: true, objectName: "ConclHead" });
  const concl = [
    "Hệ thống giám sát áo blouse chạy hoàn toàn trên NPU của bo mạch giá rẻ, không gửi hình ra ngoài",
    "≈30 FPS, độ trễ p99 41,4 ms, 2,2 W, ổn định 24 giờ",
    "Đóng góp: nén INT8 đầu ra tách, chế độ thích nghi, luật cảnh báo chỉ kết luận khi thấy rõ trang phục",
    "Hướng tiếp: dữ liệu đúng vị trí lắp đặt, đánh giá té ngã và lửa/khói",
  ];
  s.addText(concl.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < concl.length - 1, paraSpaceAfter: 6 } })),
    { x: 0.5, y: 1.55, w: 4.3, h: 3.0, fontSize: 13, color: C.background1, margin: 0, valign: "top", isTextBox: true, objectName: "ConclList" });
  card(s, "DemoCard", 5.1, 1.15, 4.4, 3.45, "1F4466");
  s.addText("Kịch bản demo (5 phút)", { x: 5.3, y: 1.25, w: 4.0, h: 0.35, fontSize: 15, bold: true, color: C.background1, margin: 0, isTextBox: true, objectName: "DemoHead" });
  const steps = [
    "Không ai trong khung → AN TOÀN, đèn xanh",
    "Người mặc áo lab vào vùng → khung xanh, vẫn an toàn",
    "Người không áo vào vùng → BÁO ĐỘNG: đèn đỏ nháy, OLED, ảnh, Zalo",
    "Mặc áo vào tại chỗ → hết báo",
    "Không áo đứng ngoài vùng → chỉ CHÚ Ý",
  ];
  s.addText(steps.map((t, i) => ({ text: t, options: { bullet: { type: "number" }, breakLine: i < steps.length - 1, paraSpaceAfter: 5 } })),
    { x: 5.3, y: 1.7, w: 4.0, h: 2.8, fontSize: 12, color: C.background1, margin: 0, valign: "top", isTextBox: true, objectName: "DemoSteps" });
  s.addText("Cảm ơn hội đồng đã lắng nghe.", { x: 0.5, y: 4.75, w: 9.0, h: 0.4, fontSize: 14, bold: true, color: C.background2, align: "center", margin: 0, isTextBox: true, objectName: "Thanks" });
  s.addNotes(
    "[4:00–4:40] Tóm lại: nhóm đã làm ra một thiết bị giám sát áo blouse chạy hoàn toàn tại chỗ trên NPU, khoảng 30 khung hình mỗi giây, 2,2 W, ổn định 24 giờ. Ba đóng góp chính là nén INT8 đầu ra tách, chế độ thích nghi và luật chỉ kết luận khi thấy rõ trang phục.\n" +
    "Bây giờ em xin demo trực tiếp. Hệ thống chờ 3 giây để chắc chắn nên báo động hiện sau khoảng 3 đến 5 giây.\n" +
    "[Nhắc demo, không đọc] Một người vào vùng mỗi lần, không đứng sát người mặc áo; đóng vai không áo bằng áo tối màu. Nếu trang hiện MẤT TÍN HIỆU: đúng thiết kế, cắm lại camera. Có video dự phòng."
  );
}

// =====================================================================
// A. Appendix divider
// =====================================================================
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addText("Phụ lục", { x: 0.5, y: 1.8, w: 9, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, margin: 0, isTextBox: true, objectName: "Title" });
  s.addText("Dùng khi hội đồng hỏi: so sánh từng khâu, áo thun trắng, kiến trúc RK3588S, nén INT8, luồng xử lý thật, độ chính xác và giới hạn.", { x: 0.5, y: 2.8, w: 8, h: 0.8, fontSize: 16, color: C.background2, margin: 0, isTextBox: true, objectName: "Subtitle" });
  s.addNotes("Phần trình bày chính kết thúc ở slide 5. Các slide sau chỉ mở khi trả lời câu hỏi.");
}

// =====================================================================
// 3. Comparison
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Phụ lục 1 · So sánh từng khâu", Object.assign({ color: C.text2, isTextBox: true, objectName: "Title" }, TITLE));
  const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "16324F" }, fontSize: 12 } });
  const rows = [
    [hdr("Khâu"), hdr("Cách làm thông thường"), hdr("Nhóm em làm thêm"), hdr("Kết quả đo")],
    ["Nơi suy luận", "Gửi video lên máy chủ GPU/cloud, hoặc chạy CPU", "Chạy trên NPU của bo mạch, INT8", "CPU 10,5 FPS, 86,8 °C → NPU ≈30 FPS"],
    ["Lượng tử hóa INT8", "Xuất mặc định: tọa độ và điểm lớp chung 1 thang", "Tách đầu ra thành 6 tensor", "mAP50-95: 0,481 → 0,657 (FP32 0,661)"],
    ["Kiểm tra sau chuyển đổi", "Chỉ xem tốc độ và \"có phát hiện\"", "So kết quả NPU với CPU trên cùng ảnh", "Bắt 4 lỗi; IoU 0,84 → 0,97"],
    ["Chạy nhiều mô hình", "Mọi mô hình chạy mọi khung", "Chế độ thích nghi, mỗi tầng 1 lõi NPU", "Công suất 3,34 → 1,98 W; NPU 74 → 32 ms/khung"],
    ["Xác định vào vùng", "Dùng tâm khung người", "Dùng điểm chân", "97% cùng quyết định với nhãn tay"],
    ["Kết luận \"không áo\"", "Không thấy áo là báo vi phạm", "Chỉ kết luận khi thấy rõ trang phục", "Cảnh báo 31 → 6, cả 6 đúng"],
    ["Dữ liệu huấn luyện", "Chỉ ảnh tại chỗ", "Thêm 238 ảnh áo trắng thường (COCO)", "Nhầm áo thun trắng: 20/20 → 0–4/20"],
  ];
  const body = rows.map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { fontSize: 11, color: j === 3 ? "0E7C86" : "1B2430", bold: j === 0 || j === 3, fill: { color: i % 2 ? "FFFFFF" : "EDF3F6" } } })));
  s.addTable(body, { x: 0.5, y: 1.1, w: 9.0, colW: [1.55, 2.55, 2.4, 2.5], border: { type: "solid", pt: 0.5, color: "D5DEE6" }, valign: "middle", margin: [3, 5, 3, 5], rowH: 0.47, objectName: "CompareTable" });
  s.addNotes("Phụ lục: bảng so sánh từng khâu với cách làm thông thường, kèm số đo trước và sau. Các số là so sánh trong chính hệ thống của nhóm.");
}

// =====================================================================
// 4. Results
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Phụ lục 2 · Áo thun trắng", Object.assign({ color: C.text2, isTextBox: true, objectName: "Title" }, TITLE));
  const stats = [["≈30 FPS", "bằng tốc độ camera"], ["41,4 ms", "độ trễ p99"], ["2,2 W", "công suất"], ["24 giờ", "chạy liên tục, 0 lỗi"]];
  stats.forEach((st, i) => {
    const y = 1.1 + i * 0.95;
    card(s, "Stat" + (i + 1), 0.5, y, 2.3, 0.82, C.background2);
    s.addText(st[0], { x: 0.5, y: y + 0.02, w: 2.3, h: 0.48, fontFace: "Cambria", fontSize: 24, bold: true, color: C.accent1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "StatValue" + (i + 1) });
    s.addText(st[1], { x: 0.5, y: y + 0.48, w: 2.3, h: 0.3, fontSize: 11, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "StatLabel" + (i + 1) });
  });
  s.addText("Camera tối đa 30 FPS. Phép đo chuẩn có thêm tầng B, C: 29,02 FPS. Công suất: đồng hồ USB-C, ±10%.", { x: 0.5, y: 4.9, w: 2.3, h: 0.5, fontSize: 9, color: C.accent5, margin: 0, valign: "top", isTextBox: true, objectName: "StatNote" });
  s.addImage({ path: SIMG("lab_white_tshirt.jpg"), x: 3.05, y: 1.1, w: 4.0, h: 3.0, altText: "Ảnh lab thật: người mặc áo thun trắng được nhận đúng là không mặc áo lab, cạnh các bạn mặc áo lab" });
  s.addText("Ảnh thật tại phòng thí nghiệm, mô hình đang dùng", { x: 3.05, y: 4.12, w: 4.0, h: 0.28, fontSize: 10, color: C.accent5, margin: 0, isTextBox: true, objectName: "PhotoCaption" });
  s.addText([
    { text: "Áo thun trắng không bị nhầm là áo lab", options: { bold: true, color: "16324F", breakLine: true } },
    { text: "Người mặc áo thun trắng (khung cam) được nhận đúng là không mặc áo lab, trong khi các bạn mặc áo lab bên cạnh vẫn được nhận (khung xanh)." },
  ], { x: 7.25, y: 1.1, w: 2.25, h: 1.6, fontSize: 11, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 4, isTextBox: true, objectName: "TshirtCaption" });
  card(s, "TshirtStat", 7.25, 2.8, 2.25, 1.3, C.background2);
  s.addText([
    { text: "20/20 → 0–4/20", options: { bold: true, fontSize: 15, color: "0E7C86", breakLine: true } },
    { text: "ảnh áo thun trắng bị nhầm là áo lab, trước và sau khi thêm 238 ảnh áo trắng thường (COCO)", options: { fontSize: 10 } },
  ], { x: 7.35, y: 2.85, w: 2.05, h: 1.2, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: "TshirtStatText" });
  card(s, "AccCard", 3.05, 4.45, 6.45, 0.75, C.background2);
  s.addText([
    { text: "Luật vùng (INT8 trên NPU, theo từng người): ", options: { bold: true, color: "16324F" } },
    { text: "phát hiện 75–86% người không áo, báo nhầm 7–9% (góc 1). Còn bỏ sót nên mỗi cảnh báo kèm ảnh để người xác nhận." },
  ], { x: 3.2, y: 4.48, w: 6.2, h: 0.7, fontSize: 11, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: "AccText" });
  s.addNotes("Phụ lục: ảnh thật ở lab, người mặc áo thun trắng được nhận đúng là không mặc áo lab. Trước đây 20/20 ảnh áo thun trắng bị nhầm; sau khi thêm 238 ảnh COCO áo trắng thường còn 0 đến 4/20.");
}

// =====================================================================
// 2. Architecture
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Phụ lục 3 · Kiến trúc RK3588S", Object.assign({ color: C.text2, isTextBox: true, objectName: "Title" }, TITLE));
  const box = (name, x, y, w, h, fill, text, color, size) => {
    card(s, name, x, y, w, h, fill);
    s.addText(text, { x, y, w, h, fontSize: size || 11, color: color || C.text1, align: "center", valign: "middle", margin: 3, isTextBox: true, objectName: name + "Text" });
  };
  const arrow = (name, x1, y1, x2, y2) => s.addShape(pres.ShapeType.line, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001, flipH: x2 < x1, flipV: y2 < y1, line: { color: "16324F", width: 1.5, endArrowType: "triangle" }, objectName: name });
  // lane labels
  const lanes = [["NPU (3 nhân)", 1.15, C.accent1], ["CPU Cortex-A76/A55", 2.55, C.accent6], ["Đầu ra", 3.95, C.accent3]];
  lanes.forEach((l, i) => s.addText(l[0], { x: 0.5, y: l[1], w: 1.25, h: i === 2 ? 0.6 : (i === 1 ? 1.0 : 1.1), fontSize: 11, bold: true, color: l[2], valign: "middle", margin: 0, isTextBox: true, objectName: "Lane" + (i + 1) }));
  // NPU lane
  box("NpuA", 1.85, 1.15, 2.7, 1.1, C.accent1, [{ text: "Tầng A · lõi 0", options: { bold: true, breakLine: true } }, { text: "YOLO26n INT8 (đầu ra tách)\nngười + áo blouse, mọi khung" }], C.background1, 11);
  box("NpuB", 4.75, 1.15, 2.2, 1.1, C.background2, [{ text: "Tầng B · lõi 1", options: { bold: true, breakLine: true } }, { text: "kiểm tra theo người\nchỉ khi cần (bất đồng bộ)" }], C.text2, 10);
  box("NpuC", 7.15, 1.15, 2.35, 1.1, C.background2, [{ text: "Tầng C · lõi 2", options: { bold: true, breakLine: true } }, { text: "lửa/khói 1 Hz\n(có mã, đang tắt)" }], C.text2, 10);
  // CPU lane
  const cpu = [["Camera\nảnh mới nhất", 1.85], ["Theo vết\n+ gán áo", 3.4], ["Điểm chân\ntrong vùng", 4.95], ["Xác nhận 3 s\nnghỉ 60 s", 6.5], ["Trạng thái\n4 mức", 8.05]];
  cpu.forEach((c, i) => box("Cpu" + (i + 1), c[1], 2.55, 1.45, 1.0, C.background2, c[0], C.text2, 11));
  for (let i = 0; i < 4; i++) arrow("CpuArrow" + (i + 1), cpu[i][1] + 1.45, 3.05, cpu[i + 1][1], 3.05);
  arrow("CamToA", 2.4, 2.55, 2.4, 2.25); arrow("AToTrack", 3.4, 2.25, 3.9, 2.55);
  // outputs
  const outs = ["Web + ảnh bằng chứng", "Đèn LED", "Màn OLED", "Tin Zalo (cần mạng)"];
  outs.forEach((o, i) => box("Out" + (i + 1), 1.85 + i * 1.95, 3.95, 1.8, 0.6, C.background2, o, C.text2, 11));
  arrow("StateToOut", 8.75, 3.55, 8.75, 3.95);
  s.addText("GPU Mali-G610 không dùng: NPU đủ nhanh, CPU còn rảnh. Tầng B, C chỉ để đo chế độ thích nghi; sản phẩm chạy tầng A.", { x: 0.5, y: 4.75, w: 9.0, h: 0.4, fontSize: 11, color: C.accent5, margin: 0, valign: "middle", isTextBox: true, objectName: "ArchNote" });
  s.addNotes("Phụ lục: phân công các khối tính toán của RK3588S.");
}

quantSlide();

// =====================================================================
// 6. How it works
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Phụ lục 5 · Luồng xử lý thật", Object.assign({ color: C.text2, isTextBox: true, objectName: "Title" }, TITLE));
  const caps = [
    ["1", "Camera USB, 30 khung/giây"],
    ["2", "Tầng A trên NPU: người + áo (YOLO26n INT8)"],
    ["3", "Theo vết, gán áo, điểm chân"],
    ["4", "Xác nhận 3 s → báo động"],
  ];
  const files = ["pipe_a.jpg", "pipe_b.jpg", "pipe_c.jpg", "pipe_d.jpg"];
  const w = 2.1, h = 1.575, gap = 0.2;
  files.forEach((f, i) => {
    const x = 0.5 + i * (w + gap);
    s.addImage({ path: IMG(f), x, y: 1.15, w, h, altText: "Khung hình thật, bước " + caps[i][0] });
    s.addText([{ text: caps[i][0] + "  ", options: { bold: true, color: C.accent1 } }, { text: caps[i][1], options: {} }],
      { x, y: 2.78, w, h: 0.5, fontSize: 11, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "Caption" + (i + 1) });
  });
  // rule: three conditions = alarm
  const chips = [["Chân trong vùng nguy hiểm", C.accent6], ["Không mặc áo blouse", C.accent3], ["Thấy rõ trang phục", C.accent2]];
  let x = 0.5;
  const cw = 2.2, ch = 0.62, cy = 3.5;
  chips.forEach((c, i) => {
    card(s, "Chip" + (i + 1), x, cy, cw, ch, c[1]);
    s.addText(c[0], { x, y: cy, w: cw, h: ch, fontSize: 13, bold: true, color: i === 1 ? C.text1 : C.background1, align: "center", valign: "middle", margin: 4, isTextBox: true, objectName: "ChipText" + (i + 1) });
    s.addText(i < 2 ? "+" : "=", { x: x + cw, y: cy, w: 0.3, h: ch, fontSize: 18, bold: true, color: C.text2, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "Op" + (i + 1) });
    x += cw + 0.3;
  });
  card(s, "AlarmChip", x, cy, 9.5 - x, ch, C.accent4);
  s.addText("BÁO ĐỘNG", { x, y: cy, w: 9.5 - x, h: ch, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "AlarmText" });
  // outputs
  const outs = ["Trang web + ảnh bằng chứng", "Đèn LED xanh / vàng / đỏ", "Màn OLED", "Tin Zalo (cần Internet)"];
  const ow = 2.1;
  outs.forEach((o, i) => {
    const ox = 0.5 + i * (ow + 0.2);
    card(s, "Out" + (i + 1), ox, 4.35, ow, 0.55, C.background2);
    s.addText(o, { x: ox, y: 4.35, w: ow, h: 0.55, fontSize: 12, color: C.text2, bold: true, align: "center", valign: "middle", margin: 4, isTextBox: true, objectName: "OutText" + (i + 1) });
  });
  s.addNotes("Phụ lục: chỉ mở khi hội đồng hỏi. Số liệu chi tiết trong báo cáo.");
}

// =====================================================================
// 8. Accuracy + honest limits
// =====================================================================
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Phụ lục 6 · Độ chính xác, giới hạn", Object.assign({ color: C.text2, isTextBox: true, objectName: "Title" }, TITLE));
  s.addChart(pres.charts.BAR, [
    { name: "Phát hiện người không áo", labels: ["Góc camera 1", "Góc camera 2"], values: [81, 88] },
    { name: "Báo nhầm người có áo", labels: ["Góc camera 1", "Góc camera 2"], values: [8, 10] },
  ], chartBase({ x: 0.5, y: 1.15, w: 4.3, h: 3.2, title: "Luật vùng, INT8 trên NPU (%, TB 3 lần huấn luyện)", chartColors: ["2E9E4F", "C62828"], showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: "1B2430", legendFontFace: "+mn-lt", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 100 }));
  s.addText("Tính theo từng người trên ảnh kiểm tra độc lập; 97% cùng quyết định trong/ngoài vùng với khung gán tay.", { x: 0.5, y: 4.4, w: 4.3, h: 0.65, fontSize: 11, color: C.accent5, margin: 0, valign: "top", isTextBox: true, objectName: "ChartNote" });
  const lim = [
    ["Còn bỏ sót 14–25% người không áo", "Người che nhau hoặc áo bị che. Vì vậy hệ thống hỗ trợ giám sát: mỗi cảnh báo kèm ảnh để người xác nhận."],
    ["Góc camera ảnh hưởng rất lớn", "Ở góc quá khác mô hình không nhận áo (0/122 khung); đưa về góc cũ nhận 98%. Cách xử lý: cố định camera, kiểm tra trước khi dùng."],
    ["Chưa đánh giá thực tế", "Té ngã, lửa và khói đã có mã nhưng đang tắt. Công suất đo trên video phát lại, chưa gồm camera."],
  ];
  lim.forEach((l, i) => {
    const y = 1.15 + i * 1.3;
    card(s, "Limit" + (i + 1), 5.1, y, 4.4, 1.2, C.background2);
    s.addText(l[0], { x: 5.25, y: y + 0.06, w: 4.1, h: 0.35, fontSize: 13, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: "LimitHead" + (i + 1) });
    s.addText(l[1], { x: 5.25, y: y + 0.42, w: 4.1, h: 0.74, fontSize: 11, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "LimitBody" + (i + 1) });
  });
  s.addNotes("Phụ lục: chỉ mở khi hội đồng hỏi. Số liệu chi tiết trong báo cáo.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
