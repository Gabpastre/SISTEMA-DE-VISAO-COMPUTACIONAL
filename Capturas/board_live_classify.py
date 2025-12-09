# board_live_label_all.py
# Rotulagem ao vivo de TODAS as peças. Clique na câmera para salvar o crop.
# 1280x960 @30 (libcamerasrc). Marque 4 cantos (TL, TR, BL, BR) com 'm'.
# Teclas:
#   m = selecionar cantos
#   1 = alterna cor (brancas/pretas)
#   2 = peão (mantém diretórios wpawn/bpawn)
#   3 = torre
#   4 = cavalo
#   5 = bispo
#   6 = rainha
#   7 = rei
#   q = sair

import cv2 as cv
import numpy as np
import time
from pathlib import Path

# ---------- CAMERA (fixo 1280x960 @30) ----------
WIDTH, HEIGHT, FPS = 1280, 960, 30
def gst_pipeline(w, h, fps):
    return ("libcamerasrc ! "
            f"video/x-raw,width={w},height={h},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false")
def open_cap():
    cap = cv.VideoCapture(gst_pipeline(WIDTH, HEIGHT, FPS), cv.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Falha ao abrir libcamerasrc 1280x960 @30")
    return cap

# ---------- SAÍDAS ----------
BASE = Path("/home/pastre/projetos/data/label")
# exceção: peão mantém diretórios antigos
PAWN_WHITE_DIR = BASE / "labels_pawn" / "wpawn"
PAWN_BLACK_DIR = BASE / "labels_pawn" / "bpawn"
for d in (PAWN_WHITE_DIR, PAWN_BLACK_DIR):
    d.mkdir(parents=True, exist_ok=True)

# demais peças seguem padrão <plural>/<cor>
PLURALS = {
    "torre":  "torres",
    "cavalo": "cavalos",
    "bispo":  "bispos",
    "rainha": "rainhas",
    "rei":    "reis",
}
COLORS = ("brancas", "pretas")

def ensure_piece_dirs():
    for singular, plural in PLURALS.items():
        for cor in COLORS:
            (BASE / plural / cor).mkdir(parents=True, exist_ok=True)
ensure_piece_dirs()

# ---------- PARÂMETROS ----------
PREVIEW_WIN = "Camera"
WARP_SIZE   = 800     # warp quadrado
CROP_SIZE   = 96      # crop salvo
FONT        = cv.FONT_HERSHEY_SIMPLEX

# ---------- ESTADO ----------
select_mode = False
select_pts  = []          # 4 cantos: TL, TR, BL, BR
H = None                  # src->warp
H_inv = None              # warp->src
tile = None               # tamanho da casa no warp
latest_warp = None

color_idx = 0             # 0=brancas, 1=pretas  (tecla 1 alterna)
piece_key = "peao"        # tecla 2..7 seleciona
preview_ready = False

# ---------- GEOMETRIA ----------
def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    out = np.zeros((4,2), np.float32)
    out[0] = pts[np.argmin(s)]   # TL
    out[2] = pts[np.argmax(s)]   # BR
    out[1] = pts[np.argmin(d)]   # TR
    out[3] = pts[np.argmax(d)]   # BL
    return out

def compute_H(src_pts):
    dst = np.array([[0,0],[WARP_SIZE,0],[WARP_SIZE,WARP_SIZE],[0,WARP_SIZE]], np.float32)
    return cv.getPerspectiveTransform(order_corners(src_pts), dst)

def warp_frame(frame, H):
    return cv.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))

def transform_pts_xy(pts_xy, M):
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1,1,2)
    return cv.perspectiveTransform(pts, M).reshape(-1,2)

def draw_grid_on_camera(img, H_inv, cells=8):
    step = WARP_SIZE // cells
    for i in range(1, cells):
        p1, p2 = transform_pts_xy([(i*step,0),(i*step,WARP_SIZE)], H_inv).astype(int)
        cv.line(img, tuple(p1), tuple(p2), (0,255,0), 1, cv.LINE_AA)
        p1, p2 = transform_pts_xy([(0,i*step),(WARP_SIZE,i*step)], H_inv).astype(int)
        cv.line(img, tuple(p1), tuple(p2), (0,255,0), 1, cv.LINE_AA)

# ---------- PROCESSAMENTO ----------
def enhance_crop(bgr):
    # CLAHE + unsharp leve
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    h,s,v = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    v2 = clahe.apply(v)
    eq = cv.cvtColor(cv.merge([h,s,v2]), cv.COLOR_HSV2BGR)
    blur = cv.GaussianBlur(eq, (0,0), 1.0)
    sharp = cv.addWeighted(eq, 1.4, blur, -0.4, 0)
    return sharp

def crop_from_warp(warp_img, r, c):
    y0, x0 = r*tile, c*tile
    crop = warp_img[y0:y0+tile, x0:x0+tile]
    crop = crop[tile//8:tile*7//8, tile//8:tile*7//8]
    return cv.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv.INTER_AREA)

def current_color():
    return COLORS[color_idx]

def current_dirs_and_prefix():
    cor = current_color()
    if piece_key == "peao":
        if cor == "brancas":
            return PAWN_WHITE_DIR, "wpawn"
        else:
            return PAWN_BLACK_DIR, "bpawn"
    else:
        plural = PLURALS[piece_key]
        dir_ = BASE / plural / cor
        prefix = piece_key  # singular no nome do arquivo
        return dir_, prefix

# ---------- MOUSE ----------
def on_preview(event, x, y, flags, param):
    global select_pts, select_mode, H, H_inv, tile, latest_warp
    if event != cv.EVENT_LBUTTONDOWN:
        return

    if select_mode:
        if len(select_pts) < 4:
            select_pts.append((x, y))
            print(f"Ponto {len(select_pts)}: {x},{y}")
        if len(select_pts) == 4:
            H = compute_H(select_pts).astype(np.float64)
            H_inv = np.linalg.inv(H)
            tile = WARP_SIZE // 8
            select_pts.clear()
            select_mode = False
            print("Homografia definida.")
        return

    if H is None or latest_warp is None:
        return

    # clique na câmera -> célula no warp
    cx, cy = transform_pts_xy([(x, y)], H)[0]
    if not (0 <= cx < WARP_SIZE and 0 <= cy < WARP_SIZE):
        return
    c = int(cx // tile); r = int(cy // tile)

    crop = crop_from_warp(latest_warp, r, c)
    crop = enhance_crop(crop)

    out_dir, prefix = current_dirs_and_prefix()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{prefix}_r{r}_c{c}_{ts}.png"
    cv.imwrite(str(out_dir / fname), crop)
    print(f"Salvo: {out_dir / fname}")

# ---------- MAIN ----------
def main():
    global select_mode, preview_ready, H, H_inv, tile, latest_warp, color_idx, piece_key

    cap = open_cap()

    while True:
        ok, frame = cap.read()
        if not ok:
            if cv.waitKey(1) == ord('q'): break
            continue

        if not preview_ready:
            cv.namedWindow(PREVIEW_WIN)
            cv.setMouseCallback(PREVIEW_WIN, on_preview)
            preview_ready = True

        disp = frame.copy()

        # feedback da seleção
        for i, p in enumerate(select_pts):
            cv.circle(disp, p, 5, (0,0,255), -1)
            cv.putText(disp, str(i+1), (p[0]+6, p[1]-6), FONT, 0.6, (0,255,0), 2)

        # warp + grade projetada
        if H is not None:
            latest_warp = warp_frame(frame, H)
            draw_grid_on_camera(disp, H_inv, cells=8)

        # HUD
        hud1 = f"[1280x960@30]  M=selecionar  Q=sair"
        hud2 = f"Cor(1): {current_color()}   Peca(2..7): {piece_key}"
        hud3 = "2=peao  3=torre  4=cavalo  5=bispo  6=rainha  7=rei"
        cv.putText(disp, hud1, (10,30), FONT, 0.7, (255,255,255), 2)
        cv.putText(disp, hud2, (10,60), FONT, 0.7, (255,255,255), 2)
        cv.putText(disp, hud3, (10,90), FONT, 0.6, (255,255,255), 2)

        cv.imshow(PREVIEW_WIN, disp)

        k = cv.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('m'):
            H = None; H_inv = None; latest_warp = None; tile = None
            select_pts.clear(); select_mode = True
            print("Clique TL → TR → BL → BR na janela Camera.")
        elif k == ord('1'):
            color_idx = 1 - color_idx  # alterna
            print(f"Cor: {current_color()}")
        elif k == ord('2'):
            piece_key = "peao";   print("Peça: peao (wpawn/bpawn)")
        elif k == ord('3'):
            piece_key = "torre";  print("Peça: torre")
        elif k == ord('4'):
            piece_key = "cavalo"; print("Peça: cavalo")
        elif k == ord('5'):
            piece_key = "bispo";  print("Peça: bispo")
        elif k == ord('6'):
            piece_key = "rainha"; print("Peça: rainha")
        elif k == ord('7'):
            piece_key = "rei";    print("Peça: rei")

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
