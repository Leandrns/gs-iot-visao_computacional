# Tradutor de Sinais EVA

Projeto desenvolvido para a **Global Solution FIAP** no tema **Space Connect**, na disciplina de **Physical Computing IoT & IoB**.

## Integrantes

| Nome | RM |
|---|---|
| Caio Alexandre dos Santos | 558460 |
| Leandro do Nascimento Souza | 558893 |
| Rafael de Mônaco Maniezo | 556079 |
| Vinicius Rozas Pannuci de Paula Cont | 555338 |

## Descrição

Durante as atividades extra-veiculares (EVAs / spacewalks), astronautas usam sinais de mão e pose como comunicação redundante quando o áudio do rádio falha por interferência ou distância. NASA e MDRS (Mars Desert Research Station) treinam e documentam esses sinais.

Esse projeto é um sistema de visão computacional que captura vídeo da webcam em tempo real e reconhece esses sinais oficiais usando MediaPipe (detecção de mãos e pose) combinado com regras geométricas sobre os landmarks. Cada sinal reconhecido é exibido em um HUD na tela com seu significado.

### Sinais reconhecidos

Sinais de mão:
- **OK** (👌) — "Tudo bem"
- **WAIT** (✊ punho fechado) — "Atenção / perigo"
- **THUMBS UP** (👍) — "Status bom"
- **THUMBS DOWN** (👎) — "Status ruim"

Sinais de pose (corpo todo):
- **EMERGENCY** — Ambos os braços cruzados no peito ("me leve à airlock")
- **AIR** — Um braço cruzado no peito ("checar oxigênio")
- **STOP** — Braço dobrado 90° para cima ("pare!")
- **OK FORMAL** — Mão tocando o topo da cabeça (versão formal NASA do OK)

Uma camada de suavização por votação majoritária em janela deslizante (8 frames, 60% de consenso) elimina o flicker entre frames quando o classificador fica incerto.

## Bibliotecas utilizadas

- **[MediaPipe](https://ai.google.dev/edge/mediapipe)** — Tasks API (`HandLandmarker` + `PoseLandmarker`) para extração dos 21 landmarks da mão e 33 landmarks do corpo
- **[OpenCV](https://opencv.org/)** — captura da webcam, manipulação de frames e desenho do HUD
- **NumPy** — dependência transitiva das duas acima

## Como executar

### Pré-requisitos
- Python **3.10**
- Webcam

### Instalação

```powershell
# 1. Criar e ativar o ambiente virtual
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Baixar os modelos .task do MediaPipe (~13 MB, faz só uma vez)
python download_models.py
```

> Se o PowerShell bloquear a ativação do venv, rode:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### Execução

```powershell
python main.py
```

A janela da webcam abre com o HUD. Faça os sinais na frente da câmera para vê-los reconhecidos. Pressione **Q** para sair.

## Estrutura do projeto

```
.
├── main.py                  # Loop principal: webcam + MediaPipe + HUD
├── gesture_classifier.py    # Classificador rule-based + smoother temporal
├── overlay.py               # HUD e desenho dos landmarks
├── download_models.py       # Baixa os modelos .task do Google
├── requirements.txt
└── models/                  # Modelos baixados (criado pelo download_models.py)
```
