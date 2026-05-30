from ultralytics import YOLO
import cv2
import os
import numpy as np
import time
import json
from pathlib import Path
from IPython.display import Video, display, HTML
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
MODEL_PROFE  = "best.pt"
MODEL_NUEVO  = "runs_ball/Ball_Padel_LeMar_yolov8s/weights/best.pt"

# ── Dataset (para métricas) ──────────────────────────────────
DATASET     = "data/datasets/Ball_Padel_LeMar_split/val"
IMAGES_DIR  = f"{DATASET}/images"
LABELS_DIR  = f"{DATASET}/labels"

# ── Videos del partido ──────────────────────────────────────
VIDEOS_ROOT = "data/videos"
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"}

# ── Anotaciones de cancha ────────────────────────────────────
ANNOTATION_FILE = "annotations/cancha.json"   # ← ruta a tu JSON
COURT_FILTER    = True   # False = desactiva el filtro sin borrar el código

# ── Salida ───────────────────────────────────────────────────
OUTPUT_DIR  = Path("comparacion_modelos_ball")
VIDEO_DIR   = OUTPUT_DIR / "videos"
PLOT_DIR    = OUTPUT_DIR / "plots"
for d in [VIDEO_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO  = VIDEO_DIR / "SIDE_BY_SIDE_PROFE_VS_NUEVO.mp4"
OUTPUT_HTML   = OUTPUT_DIR / "reporte_analitico.html"

# ── Parámetros ───────────────────────────────────────────────
IMG_SIZE    = 960
CONF        = 0.05
IOU_THRESH  = 0.5
MAX_FRAMES  = None
SKIP_FRAMES = 1
device      = 0

# ─────────────────────────────────────────────────────────────
# COLORES
# ─────────────────────────────────────────────────────────────
COLOR_PROFE  = (0, 255, 255)    # cyan  (BGR)
COLOR_NUEVO  = (0, 255, 0)      # verde (BGR)
COLOR_GT     = (255, 100, 0)    # naranja (BGR)
COLOR_COURT  = (0, 200, 255)    # amarillo-naranja para el polígono de cancha
HEX_PROFE    = "#00FFFF"
HEX_NUEVO    = "#00FF88"
HEX_BG       = "#0d1117"

# =========================
# CARGA DE ANOTACIONES
# =========================
court_polygon_norm = []   # lista de (x_norm, y_norm)

if COURT_FILTER and Path(ANNOTATION_FILE).exists():
    with open(ANNOTATION_FILE, "r") as f:
        ann_data = json.load(f)
    court_polygon_norm = [
        (float(x), float(y))
        for x, y in ann_data.get("court_corners", [])
    ]
    print(f"✅ Anotaciones cargadas: {len(court_polygon_norm)} puntos en court_corners")
else:
    if COURT_FILTER:
        print(f"⚠️  No se encontró {ANNOTATION_FILE} — filtro de cancha DESACTIVADO")
    COURT_FILTER = False

# =========================
# MODELOS
# =========================
print("Cargando modelos...")
model_profe = YOLO(MODEL_PROFE)
model_nuevo = YOLO(MODEL_NUEVO)
print("  Clases profe:", model_profe.names)
print("  Clases nuevo:", model_nuevo.names)

# =========================
# HELPERS
# =========================

def get_best_box(result):
    """Devuelve SOLO la detección con mayor confianza."""
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0
    confs  = result.boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))
    b      = result.boxes[best_i]
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    conf   = float(b.conf[0])
    return [x1, y1, x2, y2], conf


def is_inside_court(box, court_norm, img_w, img_h):
    """
    True si el CENTRO de la bbox cae dentro del polígono de cancha.
    Si no hay polígono cargado, siempre retorna True (sin filtro).
    """
    if not court_norm or box is None:
        return True   # sin polígono → no filtramos

    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2

    pts = np.array(
        [[int(x * img_w), int(y * img_h)] for x, y in court_norm],
        dtype=np.int32
    )
    result = cv2.pointPolygonTest(pts, (float(cx), float(cy)), measureDist=False)
    return result >= 0   # >= 0 → dentro o sobre el borde


def draw_court_overlay(frame, court_norm, img_w, img_h,
                        color=COLOR_COURT, alpha=0.15):
    """
    Dibuja el polígono de cancha con:
      - relleno semitransparente
      - borde sólido
    """
    if not court_norm:
        return

    pts = np.array(
        [[int(x * img_w), int(y * img_h)] for x, y in court_norm],
        dtype=np.int32
    )

    # Relleno semitransparente
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Borde sólido
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    # Etiqueta pequeña en el primer vértice
    if len(pts) > 0:
        cv2.putText(
            frame, "CANCHA",
            (pts[0][0] + 4, pts[0][1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )


def load_gt_box(label_path, img_w, img_h):
    if not os.path.exists(str(label_path)):
        return None
    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return None
    parts = lines[0].split()
    if len(parts) < 5:
        return None
    _, cx, cy, bw, bh = map(float, parts[:5])
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return [x1, y1, x2, y2]


def iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def draw_box(img, box, conf, color, name):
    if box is None:
        return
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    cv2.circle(img, (cx, cy), 6, color, -1)
    lbl = f"{name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    by = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, by - th - 4), (x1 + tw + 4, by + 2), color, -1)
    cv2.putText(img, lbl, (x1+2, by), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)


def draw_gt(img, box):
    if box is None:
        return
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), COLOR_GT, 2)
    cv2.putText(img, "GT", (box[0], max(box[1]-8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GT, 1)


def overlay_header(img, text, color):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 38), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, text, (10, 27), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)


def safe_roc_auc(y_true, y_score):
    y_true = np.array(y_true); y_score = np.array(y_score)
    unique = np.unique(y_true)
    if len(unique) < 2:
        return float("nan"), f"(solo clase {unique[0]} presente)"
    try:
        return roc_auc_score(y_true, y_score), ""
    except Exception as e:
        return float("nan"), str(e)


def safe_avg_precision(y_true, y_score):
    y_true = np.array(y_true); y_score = np.array(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return float("nan")

# =========================
# ACUMULADORES DE MÉTRICAS
# =========================
stats = {
    "profe": {"y_true": [], "y_pred": [], "y_score": [],
              "tp": 0, "fp": 0, "fn": 0, "tn": 0,
              "iou_list": [], "conf_list": [], "inf_ms": []},
    "nuevo": {"y_true": [], "y_pred": [], "y_score": [],
              "tp": 0, "fp": 0, "fn": 0, "tn": 0,
              "iou_list": [], "conf_list": [], "inf_ms": []},
}

# =========================
# PASO 1 – NEGATIVOS SINTÉTICOS
# =========================
NEG_PER_VIDEO   = 30
NEG_CONF_MAX    = 0.15
NEGS_DIR        = OUTPUT_DIR / "negatives"
NEGS_DIR.mkdir(parents=True, exist_ok=True)

print("\n🔴 Generando negativos sintéticos desde videos...")
video_paths_neg = sorted([
    str(p) for p in Path(VIDEOS_ROOT).rglob("*")
    if p.suffix in VIDEO_EXTS
])

neg_image_paths = []

for vid_path in video_paths_neg:
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        continue
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_name = Path(vid_path).stem
    found_negs = 0
    sample_indices = np.linspace(0, n_frames - 1, min(n_frames, NEG_PER_VIDEO * 10), dtype=int)

    for fi in sample_indices:
        if found_negs >= NEG_PER_VIDEO:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        res_p = model_profe.predict(frame, imgsz=IMG_SIZE,conf=NEG_CONF_MAX, device=device, verbose=False)[0]
        res_n = model_nuevo.predict(frame, imgsz=IMG_SIZE,
                                    conf=NEG_CONF_MAX, device=device, verbose=False)[0]

        det_p = res_p.boxes is not None and len(res_p.boxes) > 0
        det_n = res_n.boxes is not None and len(res_n.boxes) > 0

        if not det_p and not det_n:
            neg_path = NEGS_DIR / f"{vid_name}_f{fi:06d}.jpg"
            cv2.imwrite(str(neg_path), frame)
            neg_image_paths.append(str(neg_path))
            found_negs += 1

    cap.release()
    print(f"  {Path(vid_path).name}: {found_negs} negativos extraídos")

print(f"  ✅ Total negativos sintéticos: {len(neg_image_paths)}")

# =========================
# PASO 2 – MÉTRICAS DESDE VAL + NEGATIVOS
# =========================
print("\n📊 Evaluando dataset de validación para métricas...")
valid_ext = {".jpg", ".jpeg", ".png"}
pos_image_paths = sorted([
    str(p) for p in Path(IMAGES_DIR).rglob("*")
    if p.suffix.lower() in valid_ext
])

image_paths = pos_image_paths + neg_image_paths
print(f"  Positivos (val): {len(pos_image_paths)}")
print(f"  Negativos (sint): {len(neg_image_paths)}")
print(f"  Total: {len(image_paths)}")

print(f"\n  Evaluando {len(image_paths)} imágenes...")
t_start = time.time()

for idx, img_path in enumerate(image_paths):
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    h, w = frame.shape[:2]

    stem       = Path(img_path).stem
    label_path = Path(LABELS_DIR) / (stem + ".txt")
    gt_box     = load_gt_box(str(label_path), w, h)
    has_gt     = gt_box is not None

    for key, model in [("profe", model_profe), ("nuevo", model_nuevo)]:
        t0     = time.perf_counter()
        res    = model.predict(frame, imgsz=IMG_SIZE, conf=CONF,
                               device=device, verbose=False)[0]
        inf_ms = (time.perf_counter() - t0) * 1000
        stats[key]["inf_ms"].append(inf_ms)

        box, conf = get_best_box(res)

        # ── Filtro de cancha también en métricas ─────────
        if COURT_FILTER and not is_inside_court(box, court_polygon_norm, w, h):
            box, conf = None, 0.0

        detected = box is not None
        y_score  = conf if detected else 0.0
        stats[key]["y_score"].append(y_score)
        stats[key]["y_true"].append(1 if has_gt else 0)
        stats[key]["conf_list"].append(conf if detected else 0.0)

        if has_gt and detected:
            iou_val = iou(gt_box, box)
            stats[key]["iou_list"].append(iou_val)
            if iou_val >= IOU_THRESH:
                stats[key]["tp"] += 1
                stats[key]["y_pred"].append(1)
            else:
                stats[key]["fp"] += 1
                stats[key]["fn"] += 1
                stats[key]["y_pred"].append(0)
        elif has_gt and not detected:
            stats[key]["fn"] += 1
            stats[key]["y_pred"].append(0)
        elif not has_gt and detected:
            stats[key]["fp"] += 1
            stats[key]["y_pred"].append(1)
        else:
            stats[key]["tn"] += 1
            stats[key]["y_pred"].append(0)

    if idx % 100 == 0:
        n_pos = sum(stats["profe"]["y_true"])
        n_neg = len(stats["profe"]["y_true"]) - n_pos
        print(f"  [{idx}/{len(image_paths)}] pos={n_pos} neg={n_neg}  {time.time()-t_start:.1f}s")

print(f"  ✅ Métricas completadas en {time.time()-t_start:.1f}s")

# =========================
# PASO 3 – VIDEO SIDE-BY-SIDE
# =========================
print("\n🎬 Buscando videos del partido...")
video_paths = sorted([
    str(p) for p in Path(VIDEOS_ROOT).rglob("*")
    if p.suffix in VIDEO_EXTS
])
print(f"  Videos encontrados: {len(video_paths)}")
for vp in video_paths:
    print(f"    {vp}")

if not video_paths:
    print("  ⚠️  No se encontraron videos en", VIDEOS_ROOT)
else:
    probe   = cv2.VideoCapture(video_paths[0])
    src_fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
    src_w   = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    OUT_W_HALF = 640
    OUT_H      = min(720, int(OUT_W_HALF * src_h / src_w))
    OUT_W      = OUT_W_HALF * 2
    OUT_FPS    = src_fps

    print(f"  Resolución output: {OUT_W}x{OUT_H} @ {OUT_FPS:.1f} fps")
    if COURT_FILTER:
        print(f"  🏟️  Filtro de cancha ACTIVO ({len(court_polygon_norm)} puntos)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, OUT_FPS, (OUT_W, OUT_H))

    total_frames_written = 0
    t_video_start = time.time()

    for vid_idx, vid_path in enumerate(video_paths):
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  ⚠️  No se pudo abrir: {vid_path}")
            continue

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_name = Path(vid_path).name
        print(f"\n  Procesando [{vid_idx+1}/{len(video_paths)}]: {vid_name} ({n_frames} frames)")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if MAX_FRAMES and total_frames_written >= MAX_FRAMES:
                break
            if frame_idx % SKIP_FRAMES != 0:
                frame_idx += 1
                continue

            fh, fw = frame.shape[:2]

            # ── Inferencia ───────────────────────────────────
            res_p = model_profe.predict(frame, imgsz=IMG_SIZE, conf=CONF,
                                        device=device, verbose=False)[0]
            res_n = model_nuevo.predict(frame, imgsz=IMG_SIZE, conf=CONF,
                                        device=device, verbose=False)[0]

            box_p, conf_p = get_best_box(res_p)
            box_n, conf_n = get_best_box(res_n)

            # ── Filtro de cancha ─────────────────────────────
            if COURT_FILTER:
                if not is_inside_court(box_p, court_polygon_norm, fw, fh):
                    box_p, conf_p = None, 0.0
                if not is_inside_court(box_n, court_polygon_norm, fw, fh):
                    box_n, conf_n = None, 0.0

            # ── Copias independientes ────────────────────────
            frame_p = frame.copy()
            frame_n = frame.copy()

            # 1) Polígono de cancha (primero, debajo de las cajas)
            if COURT_FILTER:
                draw_court_overlay(frame_p, court_polygon_norm, fw, fh)
                draw_court_overlay(frame_n, court_polygon_norm, fw, fh)

            # 2) Bbox de la pelota
            draw_box(frame_p, box_p, conf_p, COLOR_PROFE, "PROFE")
            draw_box(frame_n, box_n, conf_n, COLOR_NUEVO, "NUEVO")

            # 3) Header semitransparente
            filtered_p = " [OUT]" if (box_p is None and conf_p == 0.0) else ""
            filtered_n = " [OUT]" if (box_n is None and conf_n == 0.0) else ""
            label_p = f"PROFE  conf={conf_p:.2f}{filtered_p}" if res_p.boxes and len(res_p.boxes) else "PROFE  —"
            label_n = f"NUEVO  conf={conf_n:.2f}{filtered_n}" if res_n.boxes and len(res_n.boxes) else "NUEVO  —"
            overlay_header(frame_p, label_p, COLOR_PROFE)
            overlay_header(frame_n, label_n, COLOR_NUEVO)

            # 4) Footer
            info = f"{vid_name}  frame {frame_idx}"
            cv2.putText(frame_p, info, (6, frame_p.shape[0]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
            cv2.putText(frame_n, info, (6, frame_n.shape[0]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

            # ── Resize + combinar ────────────────────────────
            fp = cv2.resize(frame_p, (OUT_W_HALF, OUT_H))
            fn = cv2.resize(frame_n, (OUT_W_HALF, OUT_H))
            combined = np.hstack((fp, fn))
            combined[:, OUT_W_HALF-1:OUT_W_HALF+1] = (220, 220, 220)

            writer.write(combined)
            total_frames_written += 1
            frame_idx += 1

            if frame_idx % 200 == 0:
                elapsed  = time.time() - t_video_start
                fps_proc = total_frames_written / max(elapsed, 1)
                print(f"    frame {frame_idx}/{n_frames}  "
                      f"total={total_frames_written}  {fps_proc:.1f} fps proc.")

        cap.release()
        if MAX_FRAMES and total_frames_written >= MAX_FRAMES:
            print("  ⚠️  Límite MAX_FRAMES alcanzado.")
            break

    writer.release()
    elapsed_total = time.time() - t_video_start
    print(f"\n  ✅ Video generado: {OUTPUT_VIDEO}")
    print(f"     Frames totales: {total_frames_written}  |  Tiempo: {elapsed_total:.1f}s")

# ================================================================
# CÁLCULO DE MÉTRICAS  (sin cambios)
# ================================================================
def compute_metrics(s):
    y_true  = np.array(s["y_true"])
    y_pred  = np.array(s["y_pred"])
    y_score = np.array(s["y_score"])
    tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    acc   = (tp+tn) / (tp+tn+fp+fn) if (tp+tn+fp+fn) > 0 else 0.0
    roc_auc, roc_warn = safe_roc_auc(y_true, y_score)
    avg_prec  = safe_avg_precision(y_true, y_score)
    mean_iou  = float(np.mean(s["iou_list"])) if s["iou_list"] else 0.0
    pos_confs = [c for c in s["conf_list"] if c > 0]
    mean_conf = float(np.mean(pos_confs)) if pos_confs else 0.0
    mean_inf  = float(np.mean(s["inf_ms"])) if s["inf_ms"] else 0.0
    return dict(
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        roc_auc=roc_auc, roc_warn=roc_warn,
        avg_precision=avg_prec,
        mean_iou=mean_iou, mean_conf=mean_conf, mean_inf_ms=mean_inf,
        tp=tp, fp=fp, fn=fn, tn=tn,
        y_true=y_true, y_score=y_score, y_pred=y_pred,
        iou_list=s["iou_list"],
    )

m_p = compute_metrics(stats["profe"])
m_n = compute_metrics(stats["nuevo"])

print("\n══════════════════════════════════════════")
print("   MÉTRICAS FINALES")
print("══════════════════════════════════════════")
print(f"{'Métrica':<22} {'PROFE':>10} {'NUEVO':>10}")
print("─" * 46)
for k in ["accuracy","precision","recall","f1","roc_auc","avg_precision","mean_iou","mean_conf","mean_inf_ms"]:
    vp = m_p[k]; vn = m_n[k]
    vp_s = f"{vp:>10.4f}" if not np.isnan(vp) else "       N/A"
    vn_s = f"{vn:>10.4f}" if not np.isnan(vn) else "       N/A"
    warn = ""
    if k == "roc_auc":
        if m_p["roc_warn"]: warn += f" ⚠ PROFE:{m_p['roc_warn']}"
        if m_n["roc_warn"]: warn += f" ⚠ NUEVO:{m_n['roc_warn']}"
    print(f"  {k:<20} {vp_s} {vn_s}{warn}")
for k in ["tp","fp","fn","tn"]:
    print(f"  {k.upper():<20} {m_p[k]:>10} {m_n[k]:>10}")

# ================================================================
# PLOTS  (sin cambios)
# ================================================================
DARK = "#0d1117"; GREY = "#161b22"; TEXT = "#e6edf3"; ACCENT = "#58a6ff"
plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": GREY,
    "axes.edgecolor": "#30363d", "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": "#21262d",
    "font.family": "monospace",
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK)
metrics_keys = ["accuracy","precision","recall","f1","roc_auc","avg_precision","mean_iou"]
labels_k     = ["Accuracy","Precision","Recall","F1","ROC-AUC","mAP","Mean IoU"]
vp = [m_p[k] if not np.isnan(m_p[k]) else 0 for k in metrics_keys]
vn = [m_n[k] if not np.isnan(m_n[k]) else 0 for k in metrics_keys]
x  = np.arange(len(labels_k)); w = 0.35
ax = axes[0]
ax.bar(x - w/2, vp, w, label="PROFE", color=HEX_PROFE, alpha=0.9)
ax.bar(x + w/2, vn, w, label="NUEVO", color=HEX_NUEVO, alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(labels_k, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0, 1.15); ax.set_title("Comparación de Métricas", fontsize=11, color=TEXT)
ax.legend(facecolor=GREY, edgecolor="#30363d"); ax.grid(axis="y", alpha=0.4)
for i, k in enumerate(metrics_keys):
    if k == "roc_auc":
        if np.isnan(m_p["roc_auc"]): ax.text(i-w/2, 0.05, "N/A", ha="center", fontsize=7, color=HEX_PROFE)
        if np.isnan(m_n["roc_auc"]): ax.text(i+w/2, 0.05, "N/A", ha="center", fontsize=7, color=HEX_NUEVO)

ax2 = axes[1]; ax2.axis("off")
table_data = [["","Pred 0","Pred 1"],["Act 0",f"TN={m_p['tn']}",f"FP={m_p['fp']}"],["Act 1",f"FN={m_p['fn']}",f"TP={m_p['tp']}"]]
table2     = [["","Pred 0","Pred 1"],["Act 0",f"TN={m_n['tn']}",f"FP={m_n['fp']}"],["Act 1",f"FN={m_n['fn']}",f"TP={m_n['tp']}"]]
y_base = 0.85
for title, tdata, color in [("PROFE – Confusion Matrix", table_data, HEX_PROFE),("NUEVO – Confusion Matrix", table2, HEX_NUEVO)]:
    ax2.text(0.02 if "PROFE" in title else 0.52, y_base+0.08, title, transform=ax2.transAxes, color=color, fontsize=9, fontweight="bold")
    for r, row in enumerate(tdata):
        for c, val in enumerate(row):
            xp = (0.02 if "PROFE" in title else 0.52) + c * 0.14
            yp = y_base - r * 0.22
            bg = "#1f2937" if r==0 or c==0 else ("#134e2a" if (r==2 and c==2) else "#7f1d1d" if (r==1 and c==2 or r==2 and c==1) else "#1f2937")
            ax2.text(xp, yp, val, transform=ax2.transAxes, ha="center", va="center", fontsize=8, color=TEXT,
                     bbox=dict(facecolor=bg, edgecolor="#30363d", boxstyle="round,pad=0.3"))

plt.tight_layout()
bar_path = PLOT_DIR / "metricas_barras.png"
plt.savefig(bar_path, dpi=130, bbox_inches="tight", facecolor=DARK); plt.close()

for fig_fn, title, xlabel, ylabel, plot_data in [
    ("roc_curve.png",        "ROC Curve",              "False Positive Rate", "True Positive Rate", "roc"),
    ("pr_curve.png",         "Precision-Recall Curve", "Recall",              "Precision",          "pr"),
    ("iou_distribution.png", "Distribución de IoU",    "IoU",                 "Frecuencia",         "iou"),
    ("inference_time.png",   "Tiempo de Inferencia",   "ms",                  "Frecuencia",         "inf"),
]:
    fig, ax = plt.subplots(figsize=(8 if plot_data in ["iou","inf"] else 7,
                                    4 if plot_data in ["iou","inf"] else 5))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(GREY)

    if plot_data == "roc":
        for label, m, color in [("PROFE", m_p, HEX_PROFE), ("NUEVO", m_n, HEX_NUEVO)]:
            if np.isnan(m["roc_auc"]):
                ax.text(0.5, 0.5 if label=="PROFE" else 0.4,
                        f"{label}: ROC-AUC N/A", ha="center", color=color, fontsize=9, transform=ax.transAxes)
                continue
            fpr, tpr, _ = roc_curve(m["y_true"], m["y_score"])
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label}  AUC={m['roc_auc']:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.plot([0,1],[0,1],"--",color="#30363d")

    elif plot_data == "pr":
        for label, m, color in [("PROFE", m_p, HEX_PROFE), ("NUEVO", m_n, HEX_NUEVO)]:
            if np.isnan(m["avg_precision"]): continue
            p, r, _ = precision_recall_curve(m["y_true"], m["y_score"])
            ax.plot(r, p, color=color, lw=2, label=f"{label}  AP={m['avg_precision']:.3f}")
            ax.fill_between(r, p, alpha=0.08, color=color)

    elif plot_data == "iou":
        if m_p["iou_list"]: ax.hist(m_p["iou_list"], bins=30, color=HEX_PROFE, alpha=0.7, label=f"PROFE  μ={m_p['mean_iou']:.3f}")
        if m_n["iou_list"]: ax.hist(m_n["iou_list"], bins=30, color=HEX_NUEVO, alpha=0.7, label=f"NUEVO  μ={m_n['mean_iou']:.3f}")
        ax.axvline(IOU_THRESH, color="white", linestyle="--", alpha=0.6, label=f"IoU≥{IOU_THRESH}")

    elif plot_data == "inf":
        ax.hist(stats["profe"]["inf_ms"], bins=40, color=HEX_PROFE, alpha=0.7, label=f"PROFE  μ={m_p['mean_inf_ms']:.1f}ms")
        ax.hist(stats["nuevo"]["inf_ms"], bins=40, color=HEX_NUEVO, alpha=0.7, label=f"NUEVO  μ={m_n['mean_inf_ms']:.1f}ms")

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, color=TEXT)
    handles = ax.get_legend_handles_labels()
    if handles[0]: ax.legend(facecolor=GREY, edgecolor="#30363d")
    ax.grid(alpha=0.3)
    out_path = PLOT_DIR / fig_fn
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=DARK); plt.close()
    print(f"  Saved: {out_path}")

# ================================================================
# REPORTE HTML  (sin cambios relevantes, añade nota de filtro)
# ================================================================
def fmt_val(name, v):
    if np.isnan(v): return "N/A"
    if "ms" in name: return f"{v:.1f} ms"
    if "conf" in name.lower(): return f"{v:.3f}"
    return f"{v*100:.2f}%"

winner = lambda vp, vn: ("🏆","—") if vp>vn else ("—","🏆") if vn>vp else ("—","—")

rows = [
    ("Accuracy",            m_p["accuracy"],      m_n["accuracy"],      True),
    ("Precision",           m_p["precision"],     m_n["precision"],     True),
    ("Recall",              m_p["recall"],        m_n["recall"],        True),
    ("F1-Score",            m_p["f1"],            m_n["f1"],            True),
    ("ROC-AUC",             m_p["roc_auc"],       m_n["roc_auc"],       True),
    ("Avg Precision (mAP)", m_p["avg_precision"], m_n["avg_precision"], True),
    ("Mean IoU",            m_p["mean_iou"],      m_n["mean_iou"],      True),
    ("Mean Confidence",     m_p["mean_conf"],     m_n["mean_conf"],     True),
    ("Inf. Time (ms)",      m_p["mean_inf_ms"],   m_n["mean_inf_ms"],   False),
]

table_rows_html = ""
for name, vp, vn, higher_better in rows:
    wp, wn = ("—","—")
    if not (np.isnan(vp) or np.isnan(vn)):
        wp, wn = winner(vp, vn) if higher_better else winner(-vp, -vn)
    table_rows_html += f"""
        <tr>
          <td>{name}</td>
          <td class="val cyan">{fmt_val(name,vp)} {wp}</td>
          <td class="val green">{fmt_val(name,vn)} {wn}</td>
        </tr>"""

cm_rows_html = ""
for label, m, color in [("PROFE", m_p, "cyan"), ("NUEVO", m_n, "green")]:
    cm_rows_html += f"""
    <div class="cm-block">
      <h3 class="{color}">{label} – Confusion Matrix</h3>
      <table class="cm-table">
        <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
        <tr><td>Act 0</td><td class="tn">TN<br>{m['tn']}</td><td class="fp">FP<br>{m['fp']}</td></tr>
        <tr><td>Act 1</td><td class="fn">FN<br>{m['fn']}</td><td class="tp">TP<br>{m['tp']}</td></tr>
      </table>
    </div>"""

roc_note_p = f"<br><small>⚠ {m_p['roc_warn']}</small>" if m_p.get("roc_warn") else ""
roc_note_n = f"<br><small>⚠ {m_n['roc_warn']}</small>" if m_n.get("roc_warn") else ""
court_badge = f'<span style="color:#00c8ff">🏟️ Filtro de cancha ACTIVO ({len(court_polygon_norm)} pts)</span>' if COURT_FILTER else '<span style="color:#888">Filtro de cancha INACTIVO</span>'

html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Reporte Analítico – Comparación de Modelos</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:#0d1117;--surface:#161b22;--border:#30363d;
    --text:#e6edf3;--muted:#8b949e;
    --cyan:#00ffff;--green:#00ff88;--blue:#58a6ff;
    --red:#ff6b6b;--yellow:#ffd700;
  }}
  body {{ background:var(--bg);color:var(--text);font-family:'Space Grotesk',sans-serif;padding:2rem; }}
  h1 {{ font-family:'JetBrains Mono',monospace;font-size:1.6rem;color:var(--cyan);margin-bottom:.3rem; }}
  .subtitle {{ color:var(--muted);font-size:.85rem;margin-bottom:2rem; }}
  .section {{ margin-bottom:2.5rem; }}
  h2 {{ font-family:'JetBrains Mono',monospace;font-size:1rem;color:var(--blue);border-bottom:1px solid var(--border);padding-bottom:.5rem;margin-bottom:1rem; }}
  h3 {{ font-size:.9rem;margin-bottom:.5rem; }}
  .cyan {{ color:var(--cyan); }} .green {{ color:var(--green); }}
  .metrics-table {{ width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:.82rem; }}
  .metrics-table th {{ background:var(--surface);color:var(--muted);padding:.5rem 1rem;text-align:left;border:1px solid var(--border); }}
  .metrics-table td {{ padding:.5rem 1rem;border:1px solid var(--border); }}
  .metrics-table tr:nth-child(even) td {{ background:#0f1419; }}
  .val {{ text-align:right;font-weight:600;font-size:.9rem; }}
  .metrics-table td.cyan {{ color:var(--cyan); }} .metrics-table td.green {{ color:var(--green); }}
  .cm-wrap {{ display:flex;gap:2rem;flex-wrap:wrap; }}
  .cm-block {{ background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.5rem; }}
  .cm-table {{ border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:.85rem;margin-top:.5rem; }}
  .cm-table th,.cm-table td {{ border:1px solid var(--border);padding:.5rem 1rem;text-align:center; }}
  .cm-table th {{ background:#21262d;color:var(--muted); }}
  .tp {{ background:#0d2818;color:#4ade80;font-weight:700; }}
  .tn {{ background:#1a2540;color:var(--blue);font-weight:700; }}
  .fp {{ background:#3b1d1d;color:var(--red);font-weight:700; }}
  .fn {{ background:#3b2600;color:var(--yellow);font-weight:700; }}
  .plots-grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem; }}
  .plot-card {{ background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden; }}
  .plot-card img {{ width:100%;display:block; }}
  .plot-card p {{ padding:.4rem .8rem;font-size:.75rem;color:var(--muted); }}
  .cards {{ display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem; }}
  .card {{ background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.5rem;min-width:160px; }}
  .card .label {{ font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em; }}
  .card .value {{ font-size:1.4rem;font-weight:700;font-family:'JetBrains Mono',monospace;margin-top:.2rem; }}
  .note {{ font-size:.75rem;color:var(--muted);font-style:italic; }}
  footer {{ margin-top:3rem;font-size:.75rem;color:var(--muted);border-top:1px solid var(--border);padding-top:1rem; }}
</style>
</head>
<body>
<h1>🎾 Model Analytics Report</h1>
<p class="subtitle">PROFE vs NUEVO · Ball Padel Detection · IoU threshold = {IOU_THRESH} · {len(image_paths)} imágenes · {court_badge}</p>

<div class="section">
  <h2>01 / Summary Scorecards</h2>
  <div class="cards">
    <div class="card"><div class="label">PROFE F1</div><div class="value cyan">{m_p['f1']*100:.1f}%</div></div>
    <div class="card"><div class="label">NUEVO F1</div><div class="value green">{m_n['f1']*100:.1f}%</div></div>
    <div class="card"><div class="label">PROFE ROC-AUC</div><div class="value cyan">{"N/A" if np.isnan(m_p['roc_auc']) else f"{m_p['roc_auc']:.3f}"}{roc_note_p}</div></div>
    <div class="card"><div class="label">NUEVO ROC-AUC</div><div class="value green">{"N/A" if np.isnan(m_n['roc_auc']) else f"{m_n['roc_auc']:.3f}"}{roc_note_n}</div></div>
    <div class="card"><div class="label">PROFE Mean IoU</div><div class="value cyan">{m_p['mean_iou']:.3f}</div></div>
    <div class="card"><div class="label">NUEVO Mean IoU</div><div class="value green">{m_n['mean_iou']:.3f}</div></div>
    <div class="card"><div class="label">PROFE Inf.</div><div class="value cyan">{m_p['mean_inf_ms']:.1f}ms</div></div>
    <div class="card"><div class="label">NUEVO Inf.</div><div class="value green">{m_n['mean_inf_ms']:.1f}ms</div></div>
  </div>
  <p class="note">⚠ ROC-AUC = N/A cuando el dataset de validación tiene solo una clase. | Filtro de cancha aplicado también a métricas.</p>
</div>

<div class="section">
  <h2>02 / Metrics Table</h2>
  <table class="metrics-table">
    <thead><tr><th>Metric</th><th>PROFE (cyan)</th><th>NUEVO (green)</th></tr></thead>
    <tbody>{table_rows_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>03 / Confusion Matrices</h2>
  <div class="cm-wrap">{cm_rows_html}</div>
</div>

<div class="section">
  <h2>04 / Plots</h2>
  <div class="plots-grid">
    <div class="plot-card"><img src="plots/metricas_barras.png"><p>Comparación general de métricas y matrices de confusión</p></div>
    <div class="plot-card"><img src="plots/roc_curve.png"><p>ROC Curve – area bajo la curva</p></div>
    <div class="plot-card"><img src="plots/pr_curve.png"><p>Precision-Recall Curve – Average Precision</p></div>
    <div class="plot-card"><img src="plots/iou_distribution.png"><p>Distribución de IoU sobre detecciones válidas</p></div>
    <div class="plot-card"><img src="plots/inference_time.png"><p>Distribución del tiempo de inferencia por imagen</p></div>
  </div>
</div>

<footer>
  Generado con Ultralytics YOLO · scikit-learn · matplotlib · Videos: {len(video_paths)} archivos en {VIDEOS_ROOT}
</footer>
</body>
</html>"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"\n✅ Reporte HTML: {OUTPUT_HTML}")

try:
    display(Video(str(OUTPUT_VIDEO), embed=True))
    display(HTML(f'<a href="{OUTPUT_HTML}" target="_blank">📊 Abrir reporte analítico completo</a>'))
except Exception:
    pass
print("\n✅ Todo listo. Revisa comparacion_modelos_ball/")