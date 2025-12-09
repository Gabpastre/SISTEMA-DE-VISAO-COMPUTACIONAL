import cv2 as cv
import numpy as np
import time, os
from pathlib import Path

# ---------- MODOS DE CAPTURA ----------
MODES = [
    ("4:3 1640x1232 @30", 1640, 1232, 30),
    ("4:3 1280x960  @30", 1280,  960, 30),
    ("4:3 800x600   @30",  800,  600, 30),
    ("16:9 1280x720 @30", 1280,  720, 30),
]

def gst_pipeline(w, h, fps):
    return (
        f"libcamerasrc ! "
        f"video/x-raw,width={w},height={h},framerate={fps}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

def open_cap(mode_idx):
    label, w, h, fps = MODES[mode_idx]
    cap = cv.VideoCapture(gst_pipeline(w, h, fps), cv.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Falha ao abrir libcamerasrc em {label}.")
    return cap, label

# ---------- PARÂMETROS ----------
WARP_SIZE = 900
rotate_180 = False

# ---------- FUNÇÕES AUXILIARES ----------
def order_corners(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    out = np.zeros((4,2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out

def warp_board(frame_bgr, quad, size=WARP_SIZE):
    dst = np.array([[0,0],[size,0],[size,size],[0,size]], dtype=np.float32)
    M = cv.getPerspectiveTransform(order_corners(quad), dst)
    return cv.warpPerspective(frame_bgr, M, (size, size))

def draw_grid(img, cells=8):
    h, w = img.shape[:2]
    step = w // cells
    out = img.copy()
    for i in range(1, cells):
        x = i * step
        y = i * step
        cv.line(out, (x,0), (x,h), (0,255,0), 1, cv.LINE_AA)
        cv.line(out, (0,y), (w,y), (0,255,0), 1, cv.LINE_AA)
    return out

# ---------- MODO MANUAL ----------
manual_points = []
manual_mode = False
manual_ready = False

def on_mouse(event, x, y, flags, param):
    global manual_points, manual_mode, manual_ready
    if manual_mode:
        if event == cv.EVENT_LBUTTONDOWN:
            if len(manual_points) < 4:
                manual_points.append((x, y))
                print(f"Ponto {len(manual_points)} marcado: {x}, {y}")
            if len(manual_points) == 4:
                manual_ready = True

# ---------- MAIN ----------
def main():
    global manual_points, manual_mode, manual_ready, rotate_180

    mode_idx = 0
    cap, label = open_cap(mode_idx)

    save_dir = Path(os.path.expanduser("/home/pastre/projetos/board_caps"))
    save_dir.mkdir(parents=True, exist_ok=True)

    cv.namedWindow("Preview - Captura Manual")
    cv.setMouseCallback("Preview - Captura Manual", on_mouse)

    print("Pressione 'm' para modo manual de marcação.")
    print("Clique nos 4 cantos: top-left → top-right → bottom-left → bottom-right")
    print("Pressione 'r' para girar 180°, '1..4' para mudar resolução e 'q' para sair.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if cv.waitKey(1) == ord('q'):
                break
            continue

        if rotate_180:
            frame = cv.rotate(frame, cv.ROTATE_180)

        disp = frame.copy()

        # Marcação manual
        if manual_mode:
            for i, p in enumerate(manual_points):
                cv.circle(disp, p, 5, (0,0,255), -1)
                cv.putText(disp, str(i+1), (p[0]+5, p[1]-5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if manual_ready:
                warp = warp_board(frame, np.array(manual_points, dtype=np.float32))
                warp_grid = draw_grid(warp, 8)
                ts = time.strftime("%Y%m%d_%H%M%S")
                outpath = save_dir / f"manual_board_{ts}.png"
                cv.imwrite(str(outpath), warp)
                cv.imshow("Warp ortogonal + grade 8x8", warp_grid)
                print(f"Tabuleiro salvo em: {outpath}")
                manual_points.clear()
                manual_ready = False
                manual_mode = False

        txt = f"[{label}] M=manual  R=rotacao  1..4=modo  Q=sair"
        cv.putText(disp, txt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv.imshow("Preview - Captura Manual", disp)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            mode_idx = int(chr(key)) - 1
            cap.release()
            cap, label = open_cap(mode_idx)
        elif key == ord('r'):
            rotate_180 = not rotate_180
        elif key == ord('m'):
            print("Modo manual ativado — clique nos 4 cantos do tabuleiro.")
            manual_points.clear()
            manual_mode = True
            manual_ready = False

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
