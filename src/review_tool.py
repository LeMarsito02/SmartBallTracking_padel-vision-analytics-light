# ══════════════════════════════════════════════════════
# BALL REVIEW TOOL - OpenCV
# ══════════════════════════════════════════════════════
#
# ✔ Corre como script separado (python review_tool.py)
# ✔ Click para seleccionar pelota
# ✔ SPACE → comparar original
# ✔ → (flecha derecha) → skip
# ✔ 0-9 → selección rápida
# ✔ ESC → cerrar
# ✔ Guarda label automáticamente
# ✔ Elimina review pendiente
#
# ══════════════════════════════════════════════════════
# INSTALAR
# ══════════════════════════════════════════════════════
#
# pip install opencv-python pyyaml
#
# ══════════════════════════════════════════════════════
# CORRER
# ══════════════════════════════════════════════════════
#
# python review_tool.py
#
# Desde Jupyter:
# import subprocess
# subprocess.Popen(["python", "review_tool.py"])
#
# ══════════════════════════════════════════════════════

from pathlib import Path

import cv2
import yaml

import numpy as np


def to_python(obj):
    """
    Convierte tipos NumPy a tipos Python serializables.
    """
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, list):
        return [to_python(x) for x in obj]

    if isinstance(obj, tuple):
        return [to_python(x) for x in obj]

    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}

    return obj
# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════

REVIEW_DIR    = Path("dataset/review")

BOX_COLOR     = (0, 255, 0)
BOX_THICKNESS = 3

WINDOW_TITLE  = "Ball Review Tool"

WINDOW_W      = 1280
WINDOW_H      = 720


# ══════════════════════════════════════════════════════
# REVIEW FILES
# ══════════════════════════════════════════════════════

review_files = sorted(REVIEW_DIR.glob("*.yaml"))

if not review_files:
    print("✓ No hay reviews pendientes.")
    raise SystemExit


# ══════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════

state = {
    "index":         0,
    "show_original": False,
    "data":          None,
    "yaml_path":     None,
    "review_img":    None,
    "original_img":  None,
    "detections":    [],
    "display_img":   None,   # imagen que se muestra en pantalla (escalada)
    "scale":         1.0,
    "offset_x":      0,
    "offset_y":      0,
}


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def fit_image(img, max_w, max_h):
    """Escala la imagen para que quepa en max_w x max_h."""
    h, w  = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def draw_detections(img, detections, scale):
    """Dibuja bounding boxes y labels sobre la imagen escalada."""
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        x1 = int(x1 * scale)
        y1 = int(y1 * scale)
        x2 = int(x2 * scale)
        y2 = int(y2 * scale)

        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        label = f"[{idx}] {det['score']:.2f}"
        cv2.putText(
            img, label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, BOX_COLOR, 2,
        )
    return img


def draw_hud(img, yaml_path, detections, show_original):
    """Dibuja info y controles en la parte inferior."""
    h, w = img.shape[:2]

    # fondo semitransparente abajo
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    mode = "ORIGINAL" if show_original else "REVIEW"
    info = (
        f"{yaml_path.stem}  |  {len(detections)} detecciones  |  [{mode}]  |  "
        f"CLICK=seleccionar  SPACE=original  ->= skip  0-9=rapido  ESC=salir"
    )
    cv2.putText(
        img, info,
        (10, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (200, 200, 200), 1,
    )
    return img


# ══════════════════════════════════════════════════════
# CLICK CALLBACK
# ══════════════════════════════════════════════════════

def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state["show_original"]:
        return

    scale      = state["scale"]
    detections = state["detections"]

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        x1s = x1 * scale
        y1s = y1 * scale
        x2s = x2 * scale
        y2s = y2 * scale

        if x1s <= x <= x2s and y1s <= y <= y2s:
            save_selection(idx)
            return

    print("⚠  Click fuera de todas las detecciones.")


# ══════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════

def render():
    """Construye y muestra el frame actual."""
    if state["show_original"]:
        base = state["original_img"].copy()
        img, scale = fit_image(base, WINDOW_W, WINDOW_H)
    else:
        base = state["review_img"].copy()
        img, scale = fit_image(base, WINDOW_W, WINDOW_H)
        img = draw_detections(img, state["detections"], scale)

    state["scale"] = scale

    img = draw_hud(img, state["yaml_path"], state["detections"], state["show_original"])

    cv2.imshow(WINDOW_TITLE, img)


# ══════════════════════════════════════════════════════
# LOAD CURRENT
# ══════════════════════════════════════════════════════

def load_current():
    """Carga el review actual."""
    state["show_original"] = False

    if state["index"] >= len(review_files):
        print("\n✅  REVIEW FINALIZADO 🚀\n")
        cv2.destroyAllWindows()
        return False

    yaml_path = review_files[state["index"]]
    state["yaml_path"] = yaml_path

    with open(yaml_path, "r") as f:
        data = yaml.unsafe_load(f)

    state["data"]       = data
    state["detections"] = data["detections"]

    review_image_path    = yaml_path.with_suffix(".jpg")
    state["review_img"]  = cv2.imread(str(review_image_path))
    state["original_img"]= cv2.imread(data["image"])

    print(f"\n→ [{state['index'] + 1}/{len(review_files)}] {yaml_path.stem}  |  {len(state['detections'])} detecciones")

    render()
    return True


# ══════════════════════════════════════════════════════
# SAVE SELECTION
# ══════════════════════════════════════════════════════

def save_selection(idx):
    data       = state["data"]
    yaml_path  = state["yaml_path"]
    detections = state["detections"]
    selected   = detections[idx]

    review_image_path = yaml_path.with_suffix(".jpg")

    label_path = Path(data["label_file"])
    label_path.write_text(selected["yolo"] + "\n")

    print(f"✓ Detección {idx} guardada → {label_path.name}")

    yaml_path.unlink(missing_ok=True)
    review_image_path.unlink(missing_ok=True)

    state["index"] += 1
    load_current()


# ══════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════

def main():
    print("""
═══════════════════════════════════════
  BALL REVIEW TOOL
═══════════════════════════════════════

  CLICK     → seleccionar pelota
  SPACE     → comparar con original
  → (83)    → skip
  0-9       → selección rápida
  ESC       → cerrar

═══════════════════════════════════════
""")

    cv2.namedWindow(WINDOW_TITLE)
    cv2.setMouseCallback(WINDOW_TITLE, on_click)

    if not load_current():
        return

    while True:
        key = cv2.waitKey(20) & 0xFF

        # ESC → salir
        if key == 27:
            print("✓ Cerrado.")
            break

        # SPACE → toggle original
        elif key == ord(" "):
            state["show_original"] = not state["show_original"]
            render()

        # flecha derecha → skip
        elif key == 83:
            print("→ Skip")
            state["index"] += 1
            if not load_current():
                break

        # 0-9 → selección rápida
        elif ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            if idx < len(state["detections"]):
                save_selection(idx)
                if state["index"] >= len(review_files):
                    break
            else:
                print(f"⚠  No existe detección {idx}")

        # ventana cerrada manualmente
        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()