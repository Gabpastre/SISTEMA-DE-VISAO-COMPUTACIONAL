import cv2 as cv
from pathlib import Path

# Diretórios
IN_DIR  = Path("/home/pastre/projetos/board_caps")
OUT_DIR = Path("/home/pastre/projetos/data/empty")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imgs = sorted(IN_DIR.glob("*.png"))
    if not imgs:
        print(f"Nenhum warp encontrado em {IN_DIR}")
        return

    count = 0
    for p in imgs:
        img = cv.imread(str(p), cv.IMREAD_COLOR)
        if img is None:
            print(f"Falha ao abrir {p}")
            continue

        h, w = img.shape[:2]
        tile = w // 8

        for r in range(8):
            for c in range(8):
                y0, x0 = r * tile, c * tile
                crop = img[y0:y0 + tile, x0:x0 + tile]
                # recorte interno para remover bordas
                crop = crop[tile//8:tile*7//8, tile//8:tile*7//8]
                crop = cv.resize(crop, (64, 64), interpolation=cv.INTER_AREA)
                filename = f"{p.stem}_r{r}_c{c}.png"
                cv.imwrite(str(OUT_DIR / filename), crop)
                count += 1

        print(f"{p.stem}: 64 recortes adicionados à pasta empty")

    print(f"Processo concluído. {count} imagens salvas em {OUT_DIR}")

if __name__ == "__main__":
    main()
