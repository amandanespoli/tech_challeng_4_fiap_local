# Classificacao de Raio-X via Video

Modulo que processa um video local, extrai frames e classifica imagens de raio-X pulmonar em tempo real utilizando um modelo CNN (ResNet50) treinado com Transfer Learning.

## Como Funciona

O pipeline segue 4 etapas:

### 1. Carregamento do Modelo

O script importa o classificador de raio-X (`xray_classifier.py`) que carrega um modelo Keras/ResNet50 (~194MB) treinado para identificar 4 categorias pulmonares:

| Classe | Descricao |
|--------|-----------|
| 0 | Covid-19 |
| 1 | Normal |
| 2 | Pneumonia Viral |
| 3 | Pneumonia Bacteriana |

### 2. Leitura do Video

O video local e aberto com `cv2.VideoCapture`. Caso o caminho contenha caracteres especiais (acentos, cedilha), o script automaticamente copia o arquivo para um diretorio temporario como fallback — contornando uma limitacao conhecida do OpenCV no Windows.

### 3. Amostragem e Classificacao dos Frames

Para manter boa performance, o classificador nao processa todos os frames. A cada **30 frames** (~1 vez por segundo em videos a 30 FPS), o script:

1. Captura o frame atual do video
2. Converte de BGR (OpenCV) para RGB (PIL)
3. Redimensiona para 256x256 pixels e normaliza os valores de pixel para [0, 1]
4. Passa a imagem pelo modelo ResNet50, que retorna as probabilidades para cada classe
5. Seleciona a classe com maior probabilidade como resultado

Nos frames intermediarios, o ultimo resultado de classificacao permanece exibido.

### 4. Exibicao dos Resultados

O video e exibido em uma janela com overlay contendo:

- **Classe principal** e confianca (em verde)
- **Probabilidades de todas as classes** (em branco)
- **Contador de frames** (canto inferior)

Ao finalizar (fim do video ou tecla `q`), um resumo da ultima classificacao e impresso no console.

## Como Executar

```bash
# Ativar o ambiente virtual
conda activate envjarvis

# Executar a partir do diretorio Video/
cd Video
python video.py
```

### Controles

- **q** — Encerra o video e exibe o resumo final

## Estrutura de Arquivos

```
Tech Challenge 6IADT - Fase 4/
├── xray_classifier.py                  # Classificador de raio-X (ResNet50)
├── Departamento_Medico/
│   └── melhor_modelo.keras             # Modelo treinado (~194MB)
└── Video/
    ├── video.py                        # Script principal
    ├── Video 2026-02-07 at 11.29.01.mp4  # Video de exemplo
    └── README.md
```

## Dependencias

- **opencv-python** — Leitura de video e exibicao de frames
- **Pillow** — Conversao de frames para formato compativel com o modelo
- **tensorflow** — Inferencia do modelo ResNet50
- **numpy** — Manipulacao de arrays de imagem

## Configuracao

No inicio do `video.py`, a constante `CLASSIFY_EVERY_N_FRAMES` controla a frequencia de classificacao:

```python
CLASSIFY_EVERY_N_FRAMES = 30  # Classificar 1 frame a cada 30
```

Diminuir o valor aumenta a frequencia de classificacao (mais lento). Aumentar o valor reduz a frequencia (mais rapido, menos atualizacoes).
