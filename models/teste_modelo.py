#!/usr/bin/env python3
"""
Validação hold-out do Sistema de Reconhecimento de Xadrez

Gera:
- Acurácia global em hold-out (80% treino / 20% teste, estratificado)
- Relatório de classificação por classe (sklearn)
- Matriz de confusão colorida (fundo branco, números em negrito)
- Tabela em texto com: Classe, Imagens (teste), Acertos, Acurácia, Precisão, Recall, F1-score
  + totais de imagens de treino e de teste
"""

import os
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import seaborn as sns
except ImportError:
    sns = None


class ChessModelTester:
    def __init__(self):
        self.model = None
        self.label_encoder = None

        # Pastas do dataset por classe
        self.dataset_paths = {
            'wrook':  '/home/pastre/projetos/data/label/torres/brancas',
            'brook':  '/home/pastre/projetos/data/label/torres/pretas',
            'wking':  '/home/pastre/projetos/data/label/reis/brancas',
            'bking':  '/home/pastre/projetos/data/label/reis/pretas',
            'wknight': '/home/pastre/projetos/data/label/cavalos/brancas',
            'bknight': '/home/pastre/projetos/data/label/cavalos/pretas',
            'wbishop': '/home/pastre/projetos/data/label/bispos/brancas',
            'bbishop': '/home/pastre/projetos/data/label/bispos/pretas',
            'wqueen': '/home/pastre/projetos/data/label/rainhas/brancas',
            'bqueen': '/home/pastre/projetos/data/label/rainhas/pretas',
            'wpawn':  '/home/pastre/projetos/data/label/labels_pawn/wpawn',
            'bpawn':  '/home/pastre/projetos/data/label/labels_pawn/bpawn',
            'empty':  '/home/pastre/projetos/data/empty'
        }

        # Nomes legíveis
        self.piece_names = {
            'wpawn':  'Peão Branco',
            'bpawn':  'Peão Preto',
            'wrook':  'Torre Branca',
            'brook':  'Torre Preta',
            'wknight': 'Cavalo Branco',
            'bknight': 'Cavalo Preto',
            'wbishop': 'Bispo Branco',
            'bbishop': 'Bispo Preto',
            'wqueen': 'Rainha Branca',
            'bqueen': 'Rainha Preta',
            'wking':  'Rei Branco',
            'bking':  'Rei Preto',
            'empty':  'Vazio'
        }

    # ---------- EXTRAÇÃO DE FEATURES ----------

    def extract_features(self, image):
        """Extrai vetor de 208 features da imagem (mesmo pipeline do treino)."""
        try:
            img_resized = cv2.resize(image, (64, 64))

            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_hsv  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

            features = []

            # Histogramas BGR
            for i in range(3):
                hist = cv2.calcHist([img_resized], [i], None, [32], [0, 256])
                features.extend(hist.flatten())

            # Histogramas HSV
            hist_h = cv2.calcHist([img_hsv], [0], None, [30], [0, 180])
            hist_s = cv2.calcHist([img_hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([img_hsv], [2], None, [32], [0, 256])
            features.extend(hist_h.flatten())
            features.extend(hist_s.flatten())
            features.extend(hist_v.flatten())

            # Gradientes Sobel
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

            features.extend([
                np.mean(sobelx), np.std(sobelx),
                np.mean(sobely), np.std(sobely),
                np.mean(np.abs(sobelx)), np.mean(np.abs(sobely))
            ])

            # Momentos de Hu
            moments = cv2.moments(img_gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend(hu_moments)

            # Estatísticas básicas
            features.extend([
                np.mean(img_gray),
                np.std(img_gray),
                np.min(img_gray),
                np.max(img_gray),
                np.median(img_gray)
            ])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            print(f"Erro ao extrair features: {e}")
            return None

    # ---------- DATASET COMPLETO ----------

    def load_full_dataset(self):
        """Carrega todas as imagens do dataset e monta X, y."""
        X = []
        y = []

        for label, folder in self.dataset_paths.items():
            folder_path = Path(folder)
            if not folder_path.exists():
                print(f"Aviso: pasta não encontrada: {folder_path}")
                continue

            for img_name in os.listdir(folder_path):
                img_path = folder_path / img_name
                if not img_path.is_file():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Aviso: não foi possível ler {img_path}")
                    continue

                feat = self.extract_features(img)
                if feat is None:
                    continue

                X.append(feat)
                y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        print(f"Total de amostras carregadas: {len(y)}")
        return X, y

    # ---------- AVALIAÇÃO HOLD-OUT ----------

    def evaluate_holdout(self, test_size=0.2, random_state=42):
        """Treina RF em 80% e avalia em 20% estratificado."""
        print("Carregando dataset completo...")
        X, y = self.load_full_dataset()
        if len(y) == 0:
            print("Nenhuma amostra encontrada.")
            return

        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)

        print("Realizando train_test_split (hold-out)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size=test_size,
            random_state=random_state,
            stratify=y_enc
        )

        print(f"Total imagens treino: {len(y_train)}")
        print(f"Total imagens teste : {len(y_test)}")

        print("Treinando RandomForest no conjunto de treinamento...")
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        print("Gerando predições no conjunto de teste...")
        y_pred = self.model.predict(X_test)

        acc = (y_pred == y_test).mean() * 100.0
        print("\n" + "=" * 60)
        print(f"ACURÁCIA GLOBAL (HOLD-OUT {int(test_size*100)}%): {acc:.2f}%")
        print("=" * 60)

        target_names = [self.piece_names[label]
                        for label in self.label_encoder.classes_]

        print("\nRELATÓRIO DE CLASSIFICAÇÃO POR CLASSE (sklearn):")
        print(classification_report(
            y_test, y_pred,
            target_names=target_names,
            zero_division=0
        ))

        labels = np.arange(len(self.label_encoder.classes_))
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        self.plot_confusion_matrix(cm, out_path="confusion_matrix_holdout.png")
        self.print_metrics_table(cm, len(y_train), len(y_test))

    # ---------- MATRIZ DE CONFUSÃO COLORIDA ----------

    def plot_confusion_matrix(self, cm, out_path="confusion_matrix_holdout.png"):
        """
        Plota e salva a matriz de confusão com fundo branco, mapa de cores
        e números em negrito e legíveis.
        """
        class_labels = [self.piece_names[label]
                        for label in self.label_encoder.classes_]

        fig = plt.figure(figsize=(10, 8), facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')

        if sns is not None:
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                xticklabels=class_labels,
                yticklabels=class_labels,
                cmap="Blues",               # mapa de cores
                cbar=True,
                annot_kws={"weight": "bold", "color": "black"}
            )
        else:
            im = ax.imshow(cm, interpolation='nearest', cmap="Blues")
            ax.set_xticks(range(len(class_labels)))
            ax.set_yticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=45, ha='right')
            ax.set_yticklabels(class_labels)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]),
                            ha='center', va='center',
                            color='black', fontweight='bold')

        plt.xlabel('Classe Predita')
        plt.ylabel('Classe Verdadeira')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Matriz de confusão salva em: {out_path}")

    # ---------- TABELA DE MÉTRICAS EM TEXTO ----------

    def print_metrics_table(self, cm, train_count, test_count):
        """
        Imprime tabela em texto com:
        Classe, Imagens (teste), Acertos, Acurácia, Precisão, Recall, F1-score
        + totais de treino e teste.
        """
        class_labels = list(self.label_encoder.classes_)
        n = len(class_labels)

        # Cabeçalho
        header = [
            "Classe",
            "Imagens_teste",
            "Acertos",
            "Acurácia(%)",
            "Precisão",
            "Recall",
            "F1-score"
        ]

        rows = []

        for i in range(n):
            label = class_labels[i]
            nome = self.piece_names[label]

            total = int(cm[i, :].sum())
            correct = int(cm[i, i])
            acc = correct / total if total > 0 else 0.0

            col_sum = int(cm[:, i].sum())
            prec = correct / col_sum if col_sum > 0 else 0.0
            rec = acc
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            else:
                f1 = 0.0

            rows.append([
                nome,
                str(total),
                str(correct),
                f"{acc*100:.1f}",
                f"{prec:.2f}",
                f"{rec:.2f}",
                f"{f1:.2f}"
            ])

        total_imgs_all = int(np.sum(cm))
        total_correct_all = int(np.trace(cm))
        global_acc = total_correct_all / total_imgs_all if total_imgs_all > 0 else 0.0
        global_prec = global_acc
        global_rec = global_acc
        global_f1 = global_acc

        rows.append([
            "TOTAL",
            str(total_imgs_all),
            str(total_correct_all),
            f"{global_acc*100:.1f}",
            f"{global_prec:.2f}",
            f"{global_rec:.2f}",
            f"{global_f1:.2f}"
        ])

        # Cálculo de larguras das colunas
        col_widths = [len(h) for h in header]
        for row in rows:
            for j, cell in enumerate(row):
                col_widths[j] = max(col_widths[j], len(cell))

        # Função auxiliar para formatar linha
        def fmt_row(vals):
            return " | ".join(
                vals[j].ljust(col_widths[j]) for j in range(len(vals))
            )

        sep = "-+-".join("-" * w for w in col_widths)

        print("\nTABELA DE MÉTRICAS (HOLD-OUT):\n")
        print(fmt_row(header))
        print(sep)
        for row in rows:
            print(fmt_row(row))

        print("\nTotal de imagens de treino:", train_count)
        print("Total de imagens de teste :", test_count)


if __name__ == "__main__":
    tester = ChessModelTester()
    tester.evaluate_holdout(test_size=0.2, random_state=42)
