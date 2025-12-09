#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gravador de Partidas de Xadrez com Geração de PGN e Captura de Imagens
Sistema completo com análise visual dos lances
VERSÃO CORRIGIDA: Engine apenas sugere, não registra automaticamente
"""

import cv2
import numpy as np
import pickle
import os
import sys
import time
from datetime import datetime
import chess
import chess.engine

CAMERA_MODES = [
    ("4:3 1640x1232", 1640, 1232, 30),
    ("4:3 1280x960", 1280, 960, 30),
    ("4:3 800x600", 800, 600, 30),
]

def gst_pipeline(w, h, fps):
    return (
        f"libcamerasrc ! "
        f"video/x-raw,width={w},height={h},framerate={fps}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

DISPLAY_NAME = {
    'empty': '',
    'wpawn': 'P',  'bpawn': 'p',
    'wrook': 'R',  'brook': 'r',
    'wknight': 'N','bknight': 'n',
    'wbishop': 'B','bbishop': 'b',
    'wqueen': 'Q', 'bqueen': 'q',
    'wking': 'K',  'bking': 'k',
}

PIECE_TO_FEN = {
    'wpawn': 'P',  'bpawn': 'p',
    'wrook': 'R',  'brook': 'r',
    'wknight': 'N','bknight': 'n',
    'wbishop': 'B','bbishop': 'b',
    'wqueen': 'Q', 'bqueen': 'q',
    'wking': 'K',  'bking': 'k',
    'empty': ''
}

class ChessGameRecorder:
    SMOOTH_FRAMES = 3
    PROMO_CANDIDATES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

    def __init__(self):
        self.mode_idx = 0
        self.rotate_180 = False
        self.cap = None
        self.current_mode = None

        self.manual_points = []
        self.manual_mode = False
        self.manual_ready = False
        self.board_corners = None

        self.M = None
        self.M_inv = None

        self.base_path = '/home/pastre/tcc'
        self.model_path = f'{self.base_path}/models/chess_model.pkl'
        self.pgn_output_path = f'{self.base_path}/partidas'
        os.makedirs(self.pgn_output_path, exist_ok=True)

        self.model = None
        self.label_encoder = None
        self.expected_dim = None

        self.warped_size = 800

        # Estado do jogo
        self.board = chess.Board()
        self.game_active = False
        self.game_paused = False
        self.player_color = chess.WHITE
        self.last_board_state = None
        self.moves_list = []
        self.game_start_time = None

        # Flags
        self.waiting_for_verification = False
        self.waiting_for_color = False
        self.error_state = False
        self.error_details = []
        self.last_good_board_state = None

        # Engine - APENAS SUGESTÃO
        self.engine = None
        self.engine_path = "/usr/games/stockfish"
        self.engine_move = None  # Sugestão visual atual
        self.engine_thinking = False
        self.engine_suggested_move = None  # Última sugestão para comparação

        # Detecção
        self.last_labels_key = None
        self.last_labels_disp = None
        self.board_buffers = []

        # Sistema de captura de imagens da partida
        self.current_game_folder = None
        self.images_folder = None
        self.pgn_folder = None
        self.lance_counter = 0
        self.last_frame_capture = None

        self.open_camera()
        self.load_model_or_exit()

    # -------------------- Câmera e modelo --------------------

    def open_camera(self, mode_idx=None):
        if mode_idx is not None:
            self.mode_idx = mode_idx
        if self.cap is not None:
            self.cap.release()
        label, w, h, fps = CAMERA_MODES[self.mode_idx]
        self.current_mode = label
        self.cap = cv2.VideoCapture(gst_pipeline(w, h, fps), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print("Erro: não foi possível abrir a câmera.")
            sys.exit(1)
        time.sleep(2)

    def load_model_or_exit(self):
        if not os.path.exists(self.model_path):
            print(f"Erro: modelo não encontrado em {self.model_path}")
            print("Execute o treinamento primeiro")
            sys.exit(1)
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            if 'expected_dim' in data:
                self.expected_dim = data['expected_dim']
            else:
                self.expected_dim = getattr(self.model, 'n_features_in_', 207)
            print(f"Modelo carregado: {self.model_path} (dim={self.expected_dim})")
        except Exception as ex:
            print(f"Erro ao carregar modelo: {ex}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    def init_engine(self):
        try:
            if os.path.exists(self.engine_path):
                self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
                print("Engine Stockfish carregada")
            else:
                print(f"Aviso: Stockfish não encontrado em {self.engine_path}")
        except Exception as e:
            print(f"Erro ao inicializar engine: {e}")
            self.engine = None

    def close_engine(self):
        if self.engine:
            self.engine.quit()
            self.engine = None

    def get_engine_move(self):
        if not self.engine or not self.game_active:
            return None
        try:
            result = self.engine.play(self.board, chess.engine.Limit(time=1.0))
            return result.move
        except Exception as e:
            print(f"Erro ao obter lance da engine: {e}")
            return None

    # -------------------- Geometria e warp --------------------

    @staticmethod
    def order_corners(pts):
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        out = np.zeros((4, 2), dtype=np.float32)
        out[0] = pts[np.argmin(s)]
        out[2] = pts[np.argmax(s)]
        out[1] = pts[np.argmin(d)]
        out[3] = pts[np.argmax(d)]
        return out

    def compute_transforms(self):
        if self.board_corners is None:
            self.M = None
            self.M_inv = None
            return
        dst = np.array([[0, 0], [self.warped_size, 0],
                        [self.warped_size, self.warped_size], [0, self.warped_size]], dtype=np.float32)
        ordered = self.order_corners(self.board_corners)
        self.M = cv2.getPerspectiveTransform(ordered, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, ordered)

    def warp_board_from_frame(self, frame):
        if self.M is None:
            return None
        return cv2.warpPerspective(frame, self.M, (self.warped_size, self.warped_size))

    def on_mouse(self, event, x, y, flags, param):
        if self.manual_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.manual_points) < 4:
                    self.manual_points.append((x, y))
                if len(self.manual_points) == 4:
                    self.board_corners = np.array(self.manual_points, dtype=np.float32)
                    self.manual_ready = True
                    self.manual_mode = False
                    self.manual_points.clear()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.manual_points.clear()
                self.manual_ready = False

    # -------------------- Extração e classificação --------------------

    def extract_features(self, image):
        try:
            img_resized = cv2.resize(image, (64, 64))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_hsv  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            features = []
            for i in range(3):
                hist = cv2.calcHist([img_resized], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            hist_h = cv2.calcHist([img_hsv], [0], None, [30], [0, 180])
            hist_s = cv2.calcHist([img_hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([img_hsv], [2], None, [32], [0, 256])
            features.extend(hist_h.flatten())
            features.extend(hist_s.flatten())
            features.extend(hist_v.flatten())
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            features.extend([
                float(np.mean(sobelx)), float(np.std(sobelx)),
                float(np.mean(sobely)), float(np.std(sobely)),
                float(np.mean(np.abs(sobelx))), float(np.mean(np.abs(sobely)))
            ])
            moments = cv2.moments(img_gray)
            hu = cv2.HuMoments(moments).flatten()
            features.extend([float(v) for v in hu])
            features.extend([
                float(np.mean(img_gray)), float(np.std(img_gray)),
                float(np.min(img_gray)),  float(np.max(img_gray))
            ])
            return features
        except Exception as ex:
            print(f"Erro ao extrair features: {ex}")
            return None

    def align_features(self, feats):
        arr = np.array(feats, dtype=float)
        if len(arr) < self.expected_dim:
            arr = np.pad(arr, (0, self.expected_dim - len(arr)), mode='constant')
        elif len(arr) > self.expected_dim:
            arr = arr[:self.expected_dim]
        return arr

    def predict_square(self, square_img):
        if self.model is None:
            return '', 'empty'
        feats = self.extract_features(square_img)
        if feats is None:
            return '', 'empty'
        feats = self.align_features(feats).reshape(1, -1)
        pred = self.model.predict(feats)[0]
        try:
            key = self.label_encoder.inverse_transform([pred])[0]
        except Exception:
            key = pred
        return DISPLAY_NAME.get(key, key), key

    def classify_board_now(self, frame):
        if self.board_corners is None or self.M is None:
            return None
        warped = self.warp_board_from_frame(frame)
        step = warped.shape[1] // 8
        labels_key = []
        labels_disp = []
        for r in range(8):
            rk, rd = [], []
            for c in range(8):
                y0 = r * step
                x0 = c * step
                sq = warped[y0:y0 + step, x0:x0 + step]
                disp, key = self.predict_square(sq)
                rk.append(key)
                rd.append(disp)
            labels_key.append(rk)
            labels_disp.append(rd)
        return labels_key, labels_disp

    # -------------------- Suavização e burst estável --------------------

    def _majority_cell(self, vals):
        uniq, counts = np.unique(vals, return_counts=True)
        return uniq[np.argmax(counts)]

    def _majority_grid(self, grids):
        out = []
        for r in range(8):
            row = []
            for c in range(8):
                row.append(self._majority_cell([g[r][c] for g in grids]))
            out.append(row)
        return out

    def get_stable_detection(self, n=None):
        if self.board_corners is None or self.M is None:
            return None
        n = n or self.SMOOTH_FRAMES
        n = max(3, n)
        grids = []
        last_disp = None
        for _ in range(n):
            ok, fr = self.cap.read()
            if not ok:
                continue
            if self.rotate_180:
                fr = cv2.rotate(fr, cv2.ROTATE_180)
            res = self.classify_board_now(fr)
            if res is None:
                continue
            lk, ld = res
            grids.append(lk)
            last_disp = ld
            cv2.waitKey(1)
        if not grids:
            return None
        lk_stable = self._majority_grid(grids)
        disp_grid = []
        for r in range(8):
            row = []
            for c in range(8):
                row.append(DISPLAY_NAME.get(lk_stable[r][c], ''))
            disp_grid.append(row)
        return lk_stable, disp_grid

    # -------------------- Utilidades de grid --------------------

    @staticmethod
    def piece_to_key(piece: chess.Piece) -> str:
        if piece is None:
            return 'empty'
        color_prefix = 'w' if piece.color == chess.WHITE else 'b'
        type_map = {
            chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop',
            chess.ROOK: 'rook', chess.QUEEN: 'queen', chess.KING: 'king'
        }
        return color_prefix + type_map[piece.piece_type]

    def board_expected_grid(self, board_obj: chess.Board) -> list:
        grid = []
        for r in range(8):
            row = []
            for c in range(8):
                sq = chess.square(c, 7-r)
                row.append(self.piece_to_key(board_obj.piece_at(sq)))
            grid.append(row)
        return grid

    def grid_changes(self, old_grid, new_grid):
        diffs = []
        for r in range(8):
            for c in range(8):
                if old_grid[r][c] != new_grid[r][c]:
                    diffs.append((r, c))
        return diffs

    # -------------------- Detector de lance (regras estritas) --------------------

    def detect_move(self, old_labels, new_labels):
        if old_labels is None or new_labels is None:
            return None, []

        diffs_obs = self.grid_changes(old_labels, new_labels)
        set_obs = set(diffs_obs)
        if len(set_obs) == 0:
            return None, []

        before_grid = self.board_expected_grid(self.board)

        legal = list(self.board.legal_moves)
        expanded = []
        for mv in legal:
            piece = self.board.piece_at(mv.from_square)
            if piece and piece.piece_type == chess.PAWN and chess.square_rank(mv.to_square) in (0, 7):
                for promo in self.PROMO_CANDIDATES:
                    mvp = chess.Move(mv.from_square, mv.to_square, promotion=promo)
                    if mvp in legal or self.board.is_legal(mvp):
                        expanded.append(mvp)
            else:
                expanded.append(mv)

        def after_grid_for(mv):
            tb = self.board.copy()
            tb.push(mv)
            return self.board_expected_grid(tb)

        candidates = []
        for mv in expanded:
            ag = after_grid_for(mv)
            diffs_exp = self.grid_changes(before_grid, ag)
            set_exp = set(diffs_exp)
            if set_exp == set_obs:
                candidates.append((mv, ag))

        if not candidates:
            errors = []
            for (r, c) in set_obs:
                expected = before_grid[r][c]
                detected = new_labels[r][c]
                if expected != detected:
                    errors.append((r, c, expected, detected))
            return None, errors

        def consistent_with_detection(mv, ag):
            to_r = 7 - chess.square_rank(mv.to_square)
            to_c = chess.square_file(mv.to_square)
            return new_labels[to_r][to_c] == ag[to_r][to_c]

        filtered = [(mv, ag) for mv, ag in candidates if consistent_with_detection(mv, ag)]
        if not filtered:
            ag0 = candidates[0][1]
            errors = []
            for (r, c) in set_obs:
                errors.append((r, c, ag0[r][c], new_labels[r][c]))
            return None, errors

        mv, ag = filtered[0]
        residual = self.grid_changes(new_labels, ag)
        if len(residual) != 0:
            errors = []
            for (r, c) in residual:
                errors.append((r, c, ag[r][c], new_labels[r][c]))
            return None, errors

        return mv, []

    # -------------------- Verificação inicial --------------------

    def verify_initial_position(self, labels_key):
        expected_position = [
            ['brook','bknight','bbishop','bqueen','bking','bbishop','bknight','brook'],
            ['bpawn','bpawn','bpawn','bpawn','bpawn','bpawn','bpawn','bpawn'],
            ['empty']*8,
            ['empty']*8,
            ['empty']*8,
            ['empty']*8,
            ['wpawn','wpawn','wpawn','wpawn','wpawn','wpawn','wpawn','wpawn'],
            ['wrook','wknight','wbishop','wqueen','wking','wbishop','wknight','wrook']
        ]
        errors = []
        for r in range(8):
            for c in range(8):
                expected = expected_position[r][c]
                detected = labels_key[r][c]
                if expected != detected:
                    square_name = chess.square_name(chess.square(c, 7-r))
                    errors.append(f"{square_name}: esperado '{DISPLAY_NAME.get(expected, expected)}', detectado '{DISPLAY_NAME.get(detected, detected)}'")
        return errors

    # -------------------- Fluxo de partida --------------------

    def create_game_folders(self):
        """Cria estrutura de pastas partida1/Imagens e partida1/Arquivo_PGN"""
        partida_num = 1
        while True:
            folder_name = f"partida{partida_num}"
            full_path = os.path.join(self.pgn_output_path, folder_name)
            if not os.path.exists(full_path):
                break
            partida_num += 1
        
        self.current_game_folder = full_path
        self.images_folder = os.path.join(full_path, "Imagens")
        self.pgn_folder = os.path.join(full_path, "Arquivo_PGN")
        
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.pgn_folder, exist_ok=True)
        
        self.lance_counter = 0
        print(f"✓ Pasta criada: {folder_name}")
        print(f"  - Imagens: {self.images_folder}")
        print(f"  - PGN: {self.pgn_folder}")

    def start_new_game(self):
        self.board = chess.Board()
        self.moves_list = []
        self.game_start_time = datetime.now()
        self.game_active = False
        self.game_paused = False
        self.last_board_state = None
        self.engine_move = None
        self.engine_thinking = False
        self.engine_suggested_move = None
        print("\n=== NOVA PARTIDA ===")
        print("Pressione 'U' para verificar posição inicial")
        self.waiting_for_verification = True
        self.waiting_for_color = False
        self.create_game_folders()

    def set_player_color(self, color):
        self.player_color = color
        self.waiting_for_color = False
        self.game_active = True
        print(f"Você jogará com as {'BRANCAS' if color == chess.WHITE else 'PRETAS'}")
        print("Partida iniciada. Pressione U após cada lance.")
        
        # Gera primeira sugestão (oculta do jogador)
        if self.engine:
            self.engine_thinking = True
            self.engine_move = self.get_engine_move()
            self.engine_suggested_move = self.engine_move
            self.engine_thinking = False
        
        if self.last_labels_key:
            self.last_board_state = self.last_labels_key

    def is_player_turn(self):
        return self.board.turn == self.player_color

    def generate_pgn(self):
        if not self.moves_list:
            print("Nenhum lance para salvar")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.pgn_folder:
            filepath = os.path.join(self.pgn_folder, f"partida_{ts}.pgn")
        else:
            filepath = os.path.join(self.pgn_output_path, f"partida_{ts}.pgn")
        
        with open(filepath, 'w') as f:
            f.write('[Event "Partida Registrada"]\n')
            f.write('[Site "Sistema de Reconhecimento"]\n')
            f.write(f'[Date "{self.game_start_time.strftime("%Y.%m.%d")}"]\n')
            f.write('[Round "?"]\n')
            if self.player_color == chess.WHITE:
                f.write('[White "Jogador"]\n[Black "Stockfish"]\n')
            else:
                f.write('[White "Stockfish"]\n[Black "Jogador"]\n')
            result = "*"
            if self.board.is_checkmate():
                result = "1-0" if self.board.turn == chess.BLACK else "0-1"
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                result = "1/2-1/2"
            f.write(f'[Result "{result}"]\n\n')
            move_text = ""
            for i, move in enumerate(self.moves_list):
                if i % 2 == 0:
                    move_text += f"{i//2 + 1}. "
                move_text += f"{move} "
            move_text += result
            f.write(move_text + "\n")
        print(f"PGN salvo em: {filepath}")

    # ==================== SISTEMA DE CAPTURA DE IMAGENS ====================

    def draw_move_arrow(self, frame, move, color, thickness=4):
        """Desenha seta de lance no tabuleiro"""
        if self.M_inv is None:
            return frame
        display = frame.copy()
        step = self.warped_size // 8
        
        fsq, tsq = move.from_square, move.to_square
        fc, fr = chess.square_file(fsq), chess.square_rank(fsq)
        tc, tr = chess.square_file(tsq), chess.square_rank(tsq)
        fr_, fc_ = 7 - fr, fc
        tr_, tc_ = 7 - tr, tc
        
        pf = np.array([[[fc_ * step + step // 2, fr_ * step + step // 2]]], dtype=np.float32)
        pt = np.array([[[tc_ * step + step // 2, tr_ * step + step // 2]]], dtype=np.float32)
        pfo = cv2.perspectiveTransform(pf, self.M_inv)[0][0]
        pto = cv2.perspectiveTransform(pt, self.M_inv)[0][0]
        
        cv2.arrowedLine(display, (int(pfo[0]), int(pfo[1])), 
                       (int(pto[0]), int(pto[1])), color, thickness, tipLength=0.3)
        return display

    def save_move_image(self, frame, move_made, engine_suggested, san_move, is_engine_move=False):
        """
        Salva imagem LIMPA com sistema de setas e legenda:
        - VERDE: Lance ótimo (jogador fez o mesmo que engine sugeriu)
        - VERMELHA: Lance realizado pelo jogador (quando diferente da sugestão)
        - AMARELA: Lance sugerido pela engine / Lance da engine
        
        Se is_engine_move=True, apenas desenha seta amarela (sem comparação)
        """
        if self.images_folder is None:
            return
        
        self.lance_counter += 1
        display = frame.copy()
        
        if is_engine_move:
            # Lance da engine - apenas seta amarela
            display = self.draw_move_arrow(display, move_made, (0, 255, 255), 6)
        else:
            # Lance do jogador - com comparação
            same_move = (engine_suggested and move_made.uci() == engine_suggested.uci())
            
            if same_move:
                # ✅ SETA VERDE - Lance ótimo
                display = self.draw_move_arrow(display, move_made, (0, 255, 0), 6)
            else:
                # 🔴 SETA VERMELHA - Lance realizado
                display = self.draw_move_arrow(display, move_made, (0, 0, 255), 6)
                
                # 🟡 SETA AMARELA - Lance sugerido (se existe)
                if engine_suggested:
                    display = self.draw_move_arrow(display, engine_suggested, (0, 255, 255), 4)
        
        # ===== LEGENDA LIMPA NO CANTO INFERIOR DIREITO =====
        h, w = display.shape[:2]
        
        # Caixa de legenda compacta
        legend_w = 450
        legend_h = 110
        legend_x = w - legend_w - 15
        legend_y = h - legend_h - 15
        
        # Fundo semi-transparente
        overlay = display.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), 
                     (legend_x + legend_w, legend_y + legend_h), 
                     (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
        
        # Borda
        cv2.rectangle(display, (legend_x, legend_y), 
                     (legend_x + legend_w, legend_y + legend_h), 
                     (200, 200, 200), 2)
        
        # Título da legenda
        y_text = legend_y + 25
        cv2.putText(display, "LEGENDA:", 
                   (legend_x + 15, y_text), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2, cv2.LINE_AA)
        
        y_text += 30
        
        # Item VERMELHO
        cv2.circle(display, (legend_x + 25, y_text - 5), 8, (0, 0, 255), -1)
        cv2.putText(display, "LANCE REALIZADO", 
                   (legend_x + 45, y_text), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        
        y_text += 25
        
        # Item AMARELO
        cv2.circle(display, (legend_x + 25, y_text - 5), 8, (0, 255, 255), -1)
        cv2.putText(display, "MELHOR LANCE SUGERIDO PELO COMPUTADOR", 
                   (legend_x + 45, y_text), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 180), 2, cv2.LINE_AA)
        
        y_text += 25
        
        # Item VERDE
        cv2.circle(display, (legend_x + 25, y_text - 5), 8, (0, 255, 0), -1)
        cv2.putText(display, "O MELHOR LANCE FOI O REALIZADO", 
                   (legend_x + 45, y_text), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2, cv2.LINE_AA)
        
        # Salva arquivo
        filename = os.path.join(self.images_folder, f"lance{self.lance_counter}.jpg")
        cv2.imwrite(filename, display, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  📸 Imagem salva: lance{self.lance_counter}.jpg")

    # ==================== MÉTODOS VISUAIS APRIMORADOS ====================

    def draw_hud_background(self, frame, y_start, height, alpha=0.85):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (frame.shape[1], y_start + height), (20, 20, 20), -1)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def draw_status_badge(self, frame, x, y, text, bg_color, width=250):
        height = 40
        cv2.rectangle(frame, (x+3, y+3), (x+width+3, y+height+3), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x+width, y+height), bg_color, -1)
        lighter = tuple(min(255, int(c * 1.3)) for c in bg_color)
        cv2.rectangle(frame, (x, y), (x+width, y+height), lighter, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
        tx, ty = x + (width - tw) // 2, y + (height + th) // 2
        cv2.putText(frame, text, (tx+1, ty+1), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # -------------------- Overlays --------------------

    def draw_initial_position_guide(self, frame):
        if self.board_corners is None or self.M_inv is None:
            return frame
        display = frame.copy()
        step = self.warped_size // 8
        expected_position = [
            ['brook','bknight','bbishop','bqueen','bking','bbishop','bknight','brook'],
            ['bpawn']*8,
            ['empty']*8,
            ['empty']*8,
            ['empty']*8,
            ['empty']*8,
            ['wpawn']*8,
            ['wrook','wknight','wbishop','wqueen','wking','wbishop','wknight','wrook']
        ]
        for r in range(8):
            for c in range(8):
                piece_key = expected_position[r][c]
                if piece_key == 'empty':
                    continue
                label = DISPLAY_NAME.get(piece_key, piece_key)
                cx = c * step + step // 2
                cy = r * step + step // 2
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                pt_orig = cv2.perspectiveTransform(pt, self.M_inv)[0][0]
                x, y = int(pt_orig[0]), int(pt_orig[1])
                font = cv2.FONT_HERSHEY_SIMPLEX; fs = 0.6; th = 2
                (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
                cv2.rectangle(display, (x - tw // 2 - 6, y - tht // 2 - 6),
                              (x + tw // 2 + 6, y + tht // 2 + 6), (255, 200, 100), -1)
                cv2.rectangle(display, (x - tw // 2 - 6, y - tht // 2 - 6),
                              (x + tw // 2 + 6, y + tht // 2 + 6), (200, 100, 0), 2)
                cv2.putText(display, label, (x - tw // 2, y + tht // 2),
                            font, fs, (0, 0, 0), th, cv2.LINE_AA)
        return display

    def draw_error_correction_overlay(self, frame, errors_detected, last_good_state):
        if self.board_corners is None or self.M_inv is None:
            return frame
        display = frame.copy()
        step = self.warped_size // 8
        for r, c, expected_key, detected_key in errors_detected:
            cx = c * step + step // 2
            cy = r * step + step // 2
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            pt_orig = cv2.perspectiveTransform(pt, self.M_inv)[0][0]
            x, y = int(pt_orig[0]), int(pt_orig[1])
            square = np.array([
                [c * step, r * step],
                [(c + 1) * step, r * step],
                [(c + 1) * step, (r + 1) * step],
                [c * step, (r + 1) * step]
            ], dtype=np.float32).reshape(-1, 1, 2)
            square_orig = cv2.perspectiveTransform(square, self.M_inv).astype(int).reshape(-1, 2)
            cv2.line(display, tuple(square_orig[0]), tuple(square_orig[2]), (0, 0, 255), 4)
            cv2.line(display, tuple(square_orig[1]), tuple(square_orig[3]), (0, 0, 255), 4)
            cv2.polylines(display, [square_orig], True, (0, 0, 255), 3)
            expected_label = DISPLAY_NAME.get(expected_key, expected_key)
            if expected_label:
                font = cv2.FONT_HERSHEY_SIMPLEX; fs = 0.7; th = 2
                (tw, tht), _ = cv2.getTextSize(expected_label, font, fs, th)
                cv2.rectangle(display, (x - tw // 2 - 8, y - tht // 2 - 8),
                              (x + tw // 2 + 8, y + tht // 2 + 8), (255, 200, 0), -1)
                cv2.rectangle(display, (x - tw // 2 - 8, y - tht // 2 - 8),
                              (x + tw // 2 + 8, y + tht // 2 + 8), (200, 100, 0), 2)
                cv2.putText(display, expected_label, (x - tw // 2, y + tht // 2),
                            font, fs, (255, 255, 255), th, cv2.LINE_AA)
        return display

    def draw_board_coordinates(self, frame):
        if self.board_corners is None or self.M_inv is None:
            return frame
        display = frame.copy()
        step = self.warped_size // 8
        files = ['a','b','c','d','e','f','g','h']
        font = cv2.FONT_HERSHEY_SIMPLEX; fs = 0.7; th = 2
        for c, file_letter in enumerate(files):
            cx = c * step + step // 2
            pt_bottom = np.array([[[cx, self.warped_size + 15]]], dtype=np.float32)
            pt_top    = np.array([[[cx, -15]]], dtype=np.float32)
            pb = cv2.perspectiveTransform(pt_bottom, self.M_inv)[0][0]
            pt = cv2.perspectiveTransform(pt_top, self.M_inv)[0][0]
            (tw, tht), _ = cv2.getTextSize(file_letter, font, fs, th)
            x, y = int(pb[0]), int(pb[1])
            cv2.rectangle(display, (x - tw//2 - 3, y - tht//2 - 3),
                          (x + tw//2 + 3, y + tht//2 + 3), (255, 255, 255), -1)
            cv2.putText(display, file_letter, (x - tw//2, y + tht//2),
                        font, fs, (0, 100, 0), th, cv2.LINE_AA)
            x, y = int(pt[0]), int(pt[1])
            cv2.rectangle(display, (x - tw//2 - 3, y - tht//2 - 3),
                          (x + tw//2 + 3, y + tht//2 + 3), (255, 255, 255), -1)
            cv2.putText(display, file_letter, (x - tw//2, y + tht//2),
                        font, fs, (0, 100, 0), th, cv2.LINE_AA)
        for r in range(8):
            rank = str(8 - r)
            cy = r * step + step // 2
            pl = np.array([[[-15, cy]]], dtype=np.float32)
            pr = np.array([[[self.warped_size + 15, cy]]], dtype=np.float32)
            plb = cv2.perspectiveTransform(pl, self.M_inv)[0][0]
            prb = cv2.perspectiveTransform(pr, self.M_inv)[0][0]
            (tw, tht), _ = cv2.getTextSize(rank, font, fs, th)
            x, y = int(plb[0]), int(plb[1])
            cv2.rectangle(display, (x - tw//2 - 3, y - tht//2 - 3),
                          (x + tw//2 + 3, y + tht//2 + 3), (255, 255, 255), -1)
            cv2.putText(display, rank, (x - tw//2, y + tht//2),
                        font, fs, (0, 100, 0), th, cv2.LINE_AA)
            x, y = int(prb[0]), int(prb[1])
            cv2.rectangle(display, (x - tw//2 - 3, y - tht//2 - 3),
                          (x + tw//2 + 3, y + tht//2 + 3), (255, 255, 255), -1)
            cv2.putText(display, rank, (x - tw//2, y + tht//2),
                        font, fs, (0, 100, 0), th, cv2.LINE_AA)
        return display

    def overlay_piece_labels(self, frame, labels_disp):
        if self.board_corners is None or labels_disp is None or self.M_inv is None:
            return frame
        display = frame.copy()
        step = self.warped_size // 8
        for r in range(8):
            for c in range(8):
                label = labels_disp[r][c]
                if not label:
                    continue
                cx = c * step + step // 2
                cy = r * step + step // 2
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                pto = cv2.perspectiveTransform(pt, self.M_inv)[0][0]
                x, y = int(pto[0]), int(pto[1])
                font = cv2.FONT_HERSHEY_SIMPLEX; fs = 0.5; th = 2
                (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
                cv2.rectangle(display, (x - tw // 2 - 4, y - tht // 2 - 4),
                              (x + tw // 2 + 4, y + tht // 2 + 4), (255, 255, 255), -1)
                cv2.rectangle(display, (x - tw // 2 - 4, y - tht // 2 - 4),
                              (x + tw // 2 + 4, y + tht // 2 + 4), (0, 0, 0), 1)
                cv2.putText(display, label, (x - tw // 2, y + tht // 2),
                            font, fs, (0, 0, 0), th, cv2.LINE_AA)
        return display

    def draw_engine_suggestion(self, frame):
        """
        Desenha sugestão da engine na câmera (amarelo)
        IMPORTANTE: Só mostra quando for TURNO DA ENGINE (para jogador executar)
        """
        if not self.engine_move:
            return frame
        
        # Só mostra se NÃO for turno do jogador (ou seja, é turno da engine)
        if self.is_player_turn():
            return frame
        
        display = frame.copy()
        fsq = self.engine_move.from_square
        tsq = self.engine_move.to_square
        fc, fr = chess.square_file(fsq), chess.square_rank(fsq)
        tc, tr = chess.square_file(tsq), chess.square_rank(tsq)
        fr_, fc_ = 7 - fr, fc
        tr_, tc_ = 7 - tr, tc
        
        if self.M_inv is None:
            return display
        
        step = self.warped_size // 8
        pf = np.array([[[fc_ * step + step // 2, fr_ * step + step // 2]]], dtype=np.float32)
        pt = np.array([[[tc_ * step + step // 2, tr_ * step + step // 2]]], dtype=np.float32)
        pfo = cv2.perspectiveTransform(pf, self.M_inv)[0][0]
        pto = cv2.perspectiveTransform(pt, self.M_inv)[0][0]
        
        # Círculos origem (amarelo) e destino (verde)
        cv2.circle(display, (int(pfo[0]), int(pfo[1])), 20, (0, 255, 255), 3)
        cv2.circle(display, (int(pto[0]), int(pto[1])), 20, (0, 255, 0), 3)
        
        # Seta amarela indicando o movimento
        cv2.arrowedLine(display, (int(pfo[0]), int(pfo[1])), 
                       (int(pto[0]), int(pto[1])), (0, 255, 255), 4, tipLength=0.3)
        
        return display

    # -------------------- Loop principal --------------------

    def run(self):
        cv2.namedWindow("Chess Game Recorder")
        cv2.setMouseCallback("Chess Game Recorder", self.on_mouse)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                continue
            if self.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            self.compute_transforms()
            display = frame.copy()

            if self.manual_mode:
                for i, p in enumerate(self.manual_points):
                    cv2.circle(display, p, 6, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(display, str(i+1), (p[0]+8, p[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if self.board_corners is not None:
                cv2.polylines(display, [self.board_corners.astype(int)], True, (0, 255, 0), 2)
                display = self.draw_board_coordinates(display)

            if self.waiting_for_verification:
                display = self.draw_initial_position_guide(display)
            elif self.error_state and self.error_details:
                display = self.draw_error_correction_overlay(display, self.error_details, self.last_good_board_state)
            elif self.board_corners is not None and self.last_labels_disp is not None:
                display = self.overlay_piece_labels(display, self.last_labels_disp)

            # Mostra sugestão da engine na câmera APENAS quando for turno dela
            if self.game_active and not self.game_paused and self.engine_move and not self.error_state:
                display = self.draw_engine_suggestion(display)

            # ===== HUD APRIMORADO =====
            h, w = display.shape[:2]

            # Barra superior
            display = self.draw_hud_background(display, 0, 90, 0.90)
            cv2.putText(display, f"[{self.current_mode}]", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            instruction_text = ""
            instruction_color = (255, 255, 255)

            if self.manual_mode:
                instruction_text = f"Clique no canto {len(self.manual_points)+1}/4 do tabuleiro (Direito = resetar)"
                instruction_color = (100, 255, 255)
            elif self.error_state:
                instruction_text = "CORRIJA as pecas (Vermelho=erro, Azul=correto) e pressione [U]"
                instruction_color = (100, 100, 255)
            elif self.waiting_for_verification:
                instruction_text = "Posicione as pecas conforme labels AZUIS e pressione [U]"
                instruction_color = (255, 200, 100)
            elif self.waiting_for_color:
                instruction_text = "Pressione [W] para BRANCAS ou [B] para PRETAS"
                instruction_color = (100, 255, 100)
            elif self.game_active and not self.game_paused:
                if self.is_player_turn():
                    instruction_text = "SEU TURNO: Faca seu lance e pressione [U]"
                    instruction_color = (100, 200, 255)
                else:
                    # Turno da engine
                    if self.engine_move:
                        instruction_text = f"TURNO ENGINE: Execute {self.engine_move.uci()} no tabuleiro e pressione [U]"
                        instruction_color = (255, 180, 100)
                    else:
                        instruction_text = "TURNO ENGINE: Aguardando calculo..."
                        instruction_color = (255, 200, 100)
            elif self.game_paused:
                instruction_text = "PAUSADO: [P] para continuar | [G] para salvar PGN"
                instruction_color = (255, 255, 100)
            elif not self.game_active:
                if self.board_corners is None:
                    instruction_text = "Pressione [M] para marcar os 4 cantos do tabuleiro"
                    instruction_color = (200, 200, 200)
                else:
                    instruction_text = "Pressione [S] para iniciar nova partida"
                    instruction_color = (100, 255, 100)

            if instruction_text:
                cv2.putText(display, instruction_text, (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, instruction_color, 2, cv2.LINE_AA)

            # Status badge (canto superior direito)
            x_status = w - 270
            if self.error_state:
                self.draw_status_badge(display, x_status, 15, "ERRO DETECTADO", (0, 0, 180))
            elif self.waiting_for_verification:
                self.draw_status_badge(display, x_status, 15, "VERIFICAR POSICAO", (180, 120, 0))
            elif self.waiting_for_color:
                self.draw_status_badge(display, x_status, 15, "ESCOLHER COR", (0, 150, 0))
            elif self.game_active:
                if self.game_paused:
                    self.draw_status_badge(display, x_status, 15, "PAUSADO", (180, 140, 0))
                else:
                    turn_text = "BRANCAS" if self.board.turn == chess.WHITE else "PRETAS"
                    lance_num = f"Lance #{len(self.moves_list)+1}"
                    self.draw_status_badge(display, x_status, 15, f"{turn_text} | {lance_num}", (0, 120, 50))
                    
                    # Badge adicional mostrando de quem é o turno
                    if self.is_player_turn():
                        self.draw_status_badge(display, x_status-220, 15, "SEU TURNO", (20, 100, 160), 200)
                    else:
                        self.draw_status_badge(display, x_status-220, 15, "TURNO ENGINE", (160, 80, 0), 200)
            else:
                self.draw_status_badge(display, x_status, 15, "AGUARDANDO", (80, 80, 80))

            # Rodapé
            display = self.draw_hud_background(display, h-35, 35, 0.95)
            cv2.putText(display, "[M]Marcar [U]Atualizar [S]Iniciar [P]Pausar [G]PGN [R]Rotar [1/2/3]Res [Q]Sair",
                        (20, h-13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

            # Captura frame para possível salvamento
            if self.game_active and not self.game_paused and not self.error_state:
                self.last_frame_capture = display.copy()

            cv2.imshow("Chess Game Recorder", display)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                break
            elif k == ord('r'):
                self.rotate_180 = not self.rotate_180
            elif k in [ord('1'), ord('2'), ord('3')]:
                idx = int(chr(k)) - 1
                if idx < len(CAMERA_MODES):
                    self.open_camera(idx)

            if k == ord('m'):
                self.manual_points.clear()
                self.manual_mode = True
                self.manual_ready = False
            if self.manual_ready:
                self.manual_ready = False

            if k == ord('s'):
                if not self.game_active:
                    if self.engine is None:
                        self.init_engine()
                    self.start_new_game()

            if self.waiting_for_color:
                if k == ord('w'):
                    self.set_player_color(chess.WHITE)
                elif k == ord('b'):
                    self.set_player_color(chess.BLACK)

            if k == ord('p') and self.game_active:
                self.game_paused = not self.game_paused
                print(f"Partida {'pausada' if self.game_paused else 'retomada'}")

            if k == ord('u'):
                if self.board_corners is not None:
                    result = self.get_stable_detection(self.SMOOTH_FRAMES)
                    if result is not None:
                        new_labels_key, new_labels_disp = result
                        self.last_labels_key = new_labels_key
                        self.last_labels_disp = new_labels_disp
                        print("\n=== Estado do tabuleiro atualizado ===")

                        if self.error_state:
                            expected_now = self.board_expected_grid(self.board)
                            diffs = self.grid_changes(expected_now, new_labels_key)
                            if len(diffs) == 0:
                                print("Posição corrigida")
                                self.error_state = False
                                self.error_details = []
                                self.last_board_state = new_labels_key
                            else:
                                self.error_details = []
                                for (r, c) in diffs:
                                    self.error_details.append((r, c, expected_now[r][c], new_labels_key[r][c]))
                                print(f"Ainda há {len(self.error_details)} erro(s)")
                            continue

                        if self.waiting_for_verification:
                            errors = self.verify_initial_position(new_labels_key)
                            if len(errors) == 0:
                                print("Posição inicial correta")
                                self.waiting_for_verification = False
                                self.waiting_for_color = True
                            else:
                                for e in errors:
                                    print(f"- {e}")
                            continue

                        if self.game_active and not self.game_paused:
                            if self.last_board_state is None:
                                print("Estado inicial registrado")
                                self.last_board_state = new_labels_key
                                self.last_good_board_state = new_labels_key
                            else:
                                move, errors = self.detect_move(self.last_board_state, new_labels_key)

                                if move:
                                    mover_color = self.board.turn
                                    piece = self.board.piece_at(move.from_square)
                                    if not piece or piece.color != mover_color:
                                        self.error_state = True
                                        self.error_details = errors if errors else []
                                        print("Movimento rejeitado: peça de cor errada")
                                        continue

                                    if move in self.board.legal_moves:
                                        # Verifica se é turno do jogador ou da engine
                                        is_player_move = (self.board.turn == self.player_color)
                                        
                                        # Captura a sugestão ANTES de fazer o lance (só para turno do jogador)
                                        engine_suggestion = self.engine_suggested_move if is_player_move else None
                                        
                                        # Registra o lance
                                        san_move = self.board.san(move)
                                        self.board.push(move)
                                        self.moves_list.append(san_move)
                                        
                                        if is_player_move:
                                            print(f"Lance do JOGADOR: {san_move} ({move.uci()})")
                                        else:
                                            print(f"Lance da ENGINE executado: {san_move} ({move.uci()})")
                                        
                                        # Salva imagem SEMPRE (com comportamento diferente)
                                        if self.last_frame_capture is not None:
                                            if is_player_move:
                                                # Lance do jogador - com comparação
                                                self.save_move_image(self.last_frame_capture, move, 
                                                                   engine_suggestion, san_move, is_engine_move=False)
                                            else:
                                                # Lance da engine - apenas seta amarela
                                                self.save_move_image(self.last_frame_capture, move, 
                                                                   None, san_move, is_engine_move=True)
                                        
                                        self.last_board_state = new_labels_key
                                        self.last_good_board_state = new_labels_key

                                        if self.board.is_game_over():
                                            print("Fim da partida")
                                            self.game_active = False
                                            self.engine_move = None
                                            self.engine_suggested_move = None
                                        else:
                                            # Gera NOVA sugestão (sempre após qualquer lance)
                                            if self.engine:
                                                self.engine_thinking = True
                                                self.engine_move = self.get_engine_move()
                                                self.engine_suggested_move = self.engine_move
                                                self.engine_thinking = False
                                                
                                                # Se agora é turno da engine, mostra qual lance ela escolheu
                                                if self.board.turn != self.player_color:
                                                    print(f"→ ENGINE vai jogar: {self.engine_move.uci()}")
                                                    print("  Execute este lance no tabuleiro e pressione [U]")
                                            else:
                                                self.engine_move = None
                                                self.engine_suggested_move = None
                                    else:
                                        print(f"Lance ilegal: {move.uci()}")
                                        self.error_state = True
                                        self.error_details = errors if errors else []
                                elif errors:
                                    self.error_state = True
                                    self.error_details = errors
                                    print("Corrija as peças e pressione U")
                                else:
                                    print("Nenhuma mudança significativa")

            if k == ord('g'):
                self.generate_pgn()

        self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        self.close_engine()
        cv2.destroyAllWindows()

def main():
    try:
        ChessGameRecorder().run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Erro: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()