# ══════════════════════════════════════════════════════
# ANNOTATION VALIDATOR - OpenCV
# ══════════════════════════════════════════════════════
#
# ✔ Corre como script separado (python validate_annotations.py)
# ✔ Muestra imagen con anotaciones dibujadas
# ✔ Y → válida (guarda en validated/)
# ✔ N → inválida (guarda en rejected/)
# ✔ SPACE → comparar con original (sin anotaciones)
# ✔ ESC → cerrar
#
# ══════════════════════════════════════════════════════
# CORRER
# ══════════════════════════════════════════════════════
#
# python validate_annotations.py
#
# Desde Jupyter:
# import subprocess
# subprocess.run(["python", "validate_annotations.py"])
#
# ══════════════════════════════════════════════════════

from pathlib import Path
import json
import cv2
import numpy as np


# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════

ANNOTATIONS_DIR = Path("annotations")       # carpeta con los .json
IMAGES_DIR      = Path("frames")            # carpeta con las imágenes
VALIDATED_DIR   = Path("annotations/validated")
REJECTED_DIR    = Path("annotations/rejected")

WINDOW_TITLE    = "Annotation Validator"
WINDOW_W        = 1280
WINDOW_H        = 720

COLOR_MAP = {
    "court_corners":     (0,   255,   0),
    "court_limits":      (255, 255, 255),
    "court_perspective": (255, 165,   0),
    "net":               (0,     0, 255),
    "glass_limits":      (0,   255, 255),
    "mesh_limits":       (255,   0, 255),
}

CLOSE_LABELS = {"court_corners", "net", "mesh_limits", "glass_limits"}


# ══════════════════════════════════════════════════════
# SETUP DIRS
# ══════════════════════════════════════════════════════

VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════
# LOAD FILES
# ══════════════════════════════════════════════════════

annotation_files = sorted(ANNOTATIONS_DIR.glob("*.json"))

# excluir los que ya están en validated o rejected
done = (
    {f.stem for f in VALIDATED_DIR.glob("*.json")} |
    {f.stem for f in REJECTED_DIR.glob("*.json")}
)
annotation_files = [f for f in annotation_files if f.stem not in done]

if not annotation_files:
    print("✓ No hay anotaciones pendientes de validar.")
    raise SystemExit


# ══════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════

state = {
    "index":         0,
    "show_original": False,
    "json_path":     None,
    "annotations":   None,
    "original_img":  None,
    "annotated_img": None,
    "scale":         1.0,
}


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def fit_image(img, max_w, max_h):
    h, w  = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def denormalize(x_norm, y_norm, w, h):
    return int(x_norm * w), int(y_norm * h)


def draw_annotations(img, annotations):
    """Dibuja todos los puntos y líneas de las anotaciones."""
    h, w = img.shape[:2]

    for label, points in annotations.items():
        if not points:
            continue

        color     = COLOR_MAP.get(label, (255, 255, 255))
        points_px = [denormalize(x, y, w, h) for x, y in points]

        # puntos
        for px in points_px:
            cv2.circle(img, px, 5, color, -1)

        # líneas
        if len(points_px) > 1:
            for i in range(len(points_px) - 1):
                cv2.line(img, points_px[i], points_px[i + 1], color, 2)
            if label in CLOSE_LABELS:
                cv2.line(img, points_px[-1], points_px[0], color, 2)

        # nombre del label
        if points_px:
            cv2.putText(
                img, label,
                (points_px[0][0] + 6, points_px[0][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1,
            )

    return img


def draw_hud(img, json_path, index, total, show_original):
    h, w = img.shape[:2]

    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    mode = "ORIGINAL" if show_original else "ANOTADO"
    info = (
        f"[{index + 1}/{total}]  {json_path.stem}  |  [{mode}]  |  "
        f"Y=validar   N=rechazar   SPACE=original   ESC=salir"
    )
    cv2.putText(
        img, info,
        (10, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (200, 200, 200), 1,
    )
    return img


# ══════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════

def render():
    if state["show_original"]:
        base = state["original_img"].copy()
    else:
        base = state["annotated_img"].copy()

    img, scale = fit_image(base, WINDOW_W, WINDOW_H)
    state["scale"] = scale

    img = draw_hud(
        img,
        state["json_path"],
        state["index"],
        len(annotation_files),
        state["show_original"],
    )

    cv2.imshow(WINDOW_TITLE, img)


# ══════════════════════════════════════════════════════
# LOAD CURRENT
# ══════════════════════════════════════════════════════

def find_image(json_path):
    """Busca la imagen correspondiente al json (mismo nombre, varias extensiones)."""
    for ext in [".jpg", ".jpeg", ".png"]:
        # mismo directorio que el json
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
        # en IMAGES_DIR
        candidate = IMAGES_DIR / (json_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


def load_current():
    state["show_original"] = False

    if state["index"] >= len(annotation_files):
        print("\n✅  VALIDACIÓN FINALIZADA 🚀\n")
        cv2.destroyAllWindows()
        return False

    json_path = annotation_files[state["index"]]
    state["json_path"] = json_path

    with open(json_path, "r") as f:
        annotations = json.load(f)
    state["annotations"] = annotations

    img_path = find_image(json_path)
    if img_path is None:
        print(f"⚠  No se encontró imagen para {json_path.stem}, saltando...")
        state["index"] += 1
        return load_current()

    img = cv2.imread(str(img_path))
    state["original_img"]  = img.copy()
    state["annotated_img"] = draw_annotations(img.copy(), annotations)

    print(f"\n→ [{state['index'] + 1}/{len(annotation_files)}] {json_path.stem}")

    render()
    return True


# ══════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════

def save_result(validated: bool):
    json_path = state["json_path"]
    dest_dir  = VALIDATED_DIR if validated else REJECTED_DIR
    dest      = dest_dir / json_path.name

    dest.write_text(json_path.read_text())

    label = "✓ Validada" if validated else "✗ Rechazada"
    print(f"{label} → {dest}")

    state["index"] += 1
    load_current()


# ══════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════

def main():
    print("""
═══════════════════════════════════════
  ANNOTATION VALIDATOR
═══════════════════════════════════════

  Y         → validar  ✅
  N         → rechazar ❌
  SPACE     → comparar con original
  ESC       → cerrar

═══════════════════════════════════════
""")

    cv2.namedWindow(WINDOW_TITLE)

    if not load_current():
        return

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == 27:                  # ESC
            print("✓ Cerrado.")
            break

        elif key == ord(" "):          # SPACE → toggle original
            state["show_original"] = not state["show_original"]
            render()

        elif key == ord("y") or key == ord("Y"):
            save_result(validated=True)
            if state["index"] >= len(annotation_files):
                break

        elif key == ord("n") or key == ord("N"):
            save_result(validated=False)
            if state["index"] >= len(annotation_files):
                break

        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()