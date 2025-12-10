# Sistema de Visão Computacional para Aprendizado de Xadrez

Repositório do projeto de TCC em Engenharia Elétrica (FHO – Fundação Hermínio Ometto), cujo objetivo é desenvolver um sistema tutor de xadrez utilizando visão computacional e técnicas de aprendizado de máquina para reconhecer peças em um tabuleiro físico e auxiliar no aprendizado do jogo.
---

## Visão geral

O sistema captura imagens de um tabuleiro de xadrez físico por meio de uma câmera conectada a uma Raspberry Pi 4, realiza o processamento das imagens para:

- Detectar o tabuleiro
- Reconhecer as peças e suas posições
- Converter a posição para uma representação interna (FEN/PGN)
- Enviar a posição para uma engine de xadrez (Stockfish)
- Gerar recomendações didáticas e feedback para analise de partida.

---

## Arquitetura do sistema

Componentes principais:

- **Hardware**
  - Raspberry Pi 4 (4 GB RAM)
  - Câmera (Raspberry Pi Camera v2 ou webcam USB equivalente)
  - Monitor/TV para exibir interface e feedback

- **Software**
  - Raspberry Pi OS
  - Python 3
  - OpenCV para visão computacional
  - Stockfish (engine de xadrez)
  - PyGame para interface gráfica

---

## Requisitos

### Sistema

- Raspberry Pi OS (ou outra distro Linux compatível)
- Python 3.9+
- Acesso à internet (para instalação de pacotes e, se desejado, atualização do Stockfish)

### Dependências

Instalação via pip:

pip install opencv-python numpy pygame python-chess
pip install scikit-learn

#Informações acadêmicas

Projeto desenvolvido como Trabalho de Conclusão de Curso em Engenharia Elétrica no Centro Universitário da Fundação Hermínio Ometto (FHO), Araras/SP.



