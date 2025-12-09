# /home/pastre/projetos/train_all_color.py
# Uso: python3 /home/pastre/projetos/train_all_color.py
# Saídas:
#   /home/pastre/projetos/models/chess_all_color.joblib
#   /home/pastre/projetos/models/chess_all_color_meta.json

import os, glob, json
import numpy as np
import cv2
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

IMG_SIZE = (64,64)
MODEL_DIR = Path("/home/pastre/projetos/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "chess_all_color.joblib"
META_PATH  = MODEL_DIR / "chess_all_color_meta.json"

CLASS_DIRS = {
    "empty": ["/home/pastre/projetos/data/empty"],
    "wrook": ["/home/pastre/projetos/data/label/torres/brancas"],
    "brook": ["/home/pastre/projetos/data/label/torres/pretas"],
    "wking": ["/home/pastre/projetos/data/label/reis/brancas"],
    "bking": ["/home/pastre/projetos/data/label/reis/pretas"],
    "wknight": ["/home/pastre/projetos/data/label/cavalos/brancas"],
    "bknight": ["/home/pastre/projetos/data/label/cavalos/pretas"],
    "wbishop": ["/home/pastre/projetos/data/label/bispos/brancas"],
    "bbishop": ["/home/pastre/projetos/data/label/bispos/pretas"],
    "wqueen": ["/home/pastre/projetos/data/label/rainhas/brancas"],
    "bqueen": ["/home/pastre/projetos/data/label/rainhas/pretas"],
    "wpawn": ["/home/pastre/projetos/data/label/labels_pawn/wpawn"],
    "bpawn": ["/home/pastre/projetos/data/label/labels_pawn/bpawn"],
}
EXTS = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")

def iter_images(paths):
    for p in paths:
        for ext in EXTS:
            for f in glob.glob(os.path.join(p, ext)):
                yield f

def color_feats(bgr):
    img = cv2.resize(bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in cv2.split(hsv):
        h = cv2.calcHist([ch],[0],None,[16],[0,256]).ravel()
        h = h / (h.sum() + 1e-6)
        feats.append(h.astype(np.float32))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    means = np.array([l.mean(), a.mean(), b.mean()], dtype=np.float32)
    return np.concatenate(feats + [means])

def load_dataset():
    X, y = [], []
    counts = {}
    for cls, dirs in CLASS_DIRS.items():
        files = list(iter_images(dirs))
        counts[cls] = len(files)
        for fp in files:
            im = cv2.imread(fp, cv2.IMREAD_COLOR)
            if im is None: continue
            X.append(color_feats(im)); y.append(cls)
    return np.asarray(X, np.float32), np.asarray(y), counts

def main():
    X, y, counts = load_dataset()
    if X.size == 0: raise SystemExit("Dataset vazio.")
    print("Amostras por classe:")
    for k,v in counts.items(): print(f"{k:8s}: {v}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2", C=1.0, max_iter=5000, solver="lbfgs", multi_class="multinomial"))
    ])

    print("Treinando…")
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    print(classification_report(yte, yp, digits=3))

    labs = list(CLASS_DIRS.keys())
    cm = confusion_matrix(yte, yp, labels=labs)
    header = " ".join([f"{l:7s}" for l in labs])
    print("         " + header)
    for i, row in enumerate(cm):
        print(f"{labs[i]:7s} " + " ".join([f"{n:7d}" for n in row]))

    dump(clf, MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "labels": labs,
        "img_size": IMG_SIZE,
        "feat": "COLOR16_HSV+LAB_MEAN"
    }, ensure_ascii=False, indent=2))
    print("Modelo salvo:", MODEL_PATH)
    print("Meta salva:", META_PATH)

if __name__ == "__main__":
    main()
