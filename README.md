# Classificacao de Doencas Pulmonares - Chatbot Multimodal

Sistema de classificacao automatica de imagens de raio-X toracico utilizando Deep Learning (ResNet50) com interface de chatbot multimodal inspirado no conceito "Jarvis".

**Tech Challenge 6IADT - Fase 4 | FIAP Pos-Tech**

---

## Sobre o Projeto

Este chatbot integra inteligencia artificial com processamento de imagens medicas para auxiliar na identificacao de condicoes pulmonares a partir de raios-X. O sistema combina:

- **Classificacao por Deep Learning** (ResNet50) para analise de imagens
- **RAG (Retrieval-Augmented Generation)** com ChromaDB para informacoes de saude
- **Interacao multimodal**: texto, audio, imagem, video e captura de tela

> **IMPORTANTE:** Este sistema e apenas informativo e NAO substitui o diagnostico de um profissional de saude qualificado.

---

## Classes de Classificacao

| Classe | Descricao | Codigo |
|--------|-----------|--------|
| **Covid-19** | Pacientes diagnosticados com COVID-19 | 0 |
| **Healthy** | Pacientes saudaveis (sem doenca pulmonar) | 1 |
| **Viral Pneumonia** | Pneumonia causada por virus | 2 |
| **Bacterial Pneumonia** | Pneumonia causada por bacterias | 3 |

---

## Arquitetura

```
                         +-------------------+
                         |     Frontend      |
                         |  (HTML/CSS/JS)    |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |   Flask Backend   |
                         |   (chatbot.py)    |
                         +--------+----------+
                                  |
              +-------------------+-------------------+
              |                   |                   |
    +---------v-------+  +-------v--------+  +-------v--------+
    |    ResNet50      |  |   OpenAI API   |  |   ChromaDB     |
    | Classificador    |  | GPT-4o-mini    |  | RAG Saude      |
    | de Raio-X        |  | Whisper (STT)  |  | (Embeddings)   |
    | (256x256 input)  |  | TTS-1 (Voz)   |  |                |
    +---------+--------+  +-------+--------+  +-------+--------+
              |                   |                   |
              +-------------------+-------------------+
                                  |
                         +--------v----------+
                         |  Resposta JSON    |
                         |  + Audio (TTS)    |
                         +-------------------+
```

---

## Funcionalidades

### 1. Classificacao de Raio-X por Imagem

Upload de imagens de raio-X toracico para classificacao automatica.

- **Formatos aceitos:** PNG, JPG, JPEG, GIF, BMP, WEBP
- **Validacao:** GPT-4o Vision verifica se a imagem e realmente um raio-X toracico
- **Modelo:** ResNet50 com Transfer Learning (entrada 256x256 pixels)
- **Saida:** Classe predita, confianca e probabilidades de todas as classes
- **Complemento:** Informacoes de saude relacionadas via RAG/ChromaDB

### 2. Classificacao de Raio-X por Video

Upload de videos contendo imagens de raio-X para analise frame a frame.

- **Formatos aceitos:** MP4, AVI, MOV, MKV, WMV, WEBM
- **Amostragem:** 1 classificacao a cada 30 frames (~1/segundo a 30fps)
- **Exibicao em tempo real:** Janela OpenCV com overlays de classificacao
  - Classe principal em verde com porcentagem de confianca
  - Probabilidades de todas as classes em branco
  - Contador de frames no canto inferior
- **Agregacao:** Classe dominante (moda) + confianca media + probabilidades medias
- **Interacao:** Pressione 'q' para encerrar antecipadamente

### 3. Deteccao na Tela (Screen Capture)

O chatbot captura a tela do monitor para identificar raios-X abertos.

- **Ativacao:** Pergunte "o que tem na minha tela?"
- **Processo:** Minimiza o navegador, captura a tela, analisa a imagem
- **Multi-monitor:** Suporte para monitor secundario
- Se detectar raio-X: classifica automaticamente
- Se nao for raio-X: descreve o conteudo da tela via GPT-4o Vision

### 4. Chat por Texto

Conversacao textual com o chatbot para perguntas gerais e de saude.

- **Classificacao automatica de mensagens:**
  - Perguntas gerais → GPT-4o-mini
  - Perguntas de saude → RAG com ChromaDB
  - Follow-up de raio-X → Resposta contextualizada com diagnostico anterior (valido por 30 minutos)

### 5. Interacao por Audio

Gravacao de voz para enviar perguntas ao chatbot.

- **Dispositivos:** Fone de ouvido ou microfone do computador
- **Transcricao:** Whisper API (modelo whisper-1)
- **Resposta em voz:** TTS-1 da OpenAI (voz "alloy"), ativavel via checkbox "Voz"

### 6. Upload de PDF

Processamento de documentos PDF para conversacao contextualizada.

- Documentos sao divididos em chunks de 1000 caracteres (overlap de 500)
- Embeddings armazenados no ChromaDB para busca semantica
- Consultas retornam os 5 trechos mais relevantes (threshold 0.3)

---

## Rotas da API

| Metodo | Rota | Descricao |
|--------|------|-----------|
| GET | `/` | Interface do chatbot |
| POST | `/send_message` | Enviar mensagem de texto (+ deteccao de tela) |
| POST | `/upload_xray` | Upload e analise de imagem de raio-X |
| POST | `/upload_video` | Upload e analise de video de raio-X |
| POST | `/start_recording` | Iniciar gravacao (fone de ouvido) |
| POST | `/stop_recording` | Parar e transcrever (fone de ouvido) |
| POST | `/start_recording_mic` | Iniciar gravacao (microfone do computador) |
| POST | `/stop_recording_mic` | Parar e transcrever (microfone do computador) |
| POST | `/text_to_speech` | Converter texto em audio |
| POST | `/save_settings` | Salvar preferencias do usuario |

---

## Tecnologias Utilizadas

| Categoria | Tecnologia |
|-----------|------------|
| **Backend** | Flask (Python) |
| **Modelo de Classificacao** | ResNet50 (Transfer Learning) - TensorFlow/Keras |
| **LLM** | OpenAI GPT-4o-mini |
| **Speech-to-Text** | OpenAI Whisper |
| **Text-to-Speech** | OpenAI TTS-1 |
| **Banco Vetorial** | ChromaDB |
| **Embeddings** | OpenAI Embeddings API |
| **Processamento de Video** | OpenCV (cv2) |
| **Captura de Tela** | mss |
| **Audio** | PyAudio, SoundDevice, SciPy |
| **Frontend** | HTML, CSS, JavaScript, Font Awesome |

---

## Estrutura do Projeto

```
Tech Challenge 6IADT - Fase 4/
├── chatbot.py                  # Aplicacao Flask principal (rotas e logica)
├── xray_classifier.py          # Classificador ResNet50 (singleton)
├── gravar_e_transcrever.py     # Bridge: audio + video para o chatbot
├── create_db.py                # Criacao do banco ChromaDB
├── settings.json               # Preferencias do usuario
├── .env                        # Chaves de API (nao versionado)
│
├── templates/
│   └── index.html              # Interface do chatbot (single-page)
│
├── static/
│   ├── bot-avatar.png          # Avatar do bot
│   └── user-avatar.png         # Avatar do usuario
│
├── Departamento_Medico/
│   ├── melhor_modelo.keras     # Modelo ResNet50 treinado (~194MB)
│   ├── Departamento_Medico.ipynb  # Notebook de treinamento
│   ├── Dataset/                # Dataset de treinamento
│   └── Test/                   # Imagens de teste
│
├── Gravador/                   # Modulo de gravacao de audio (PyAudio)
├── Transcrever/                # Modulo de transcricao (Whisper)
├── Video/
│   └── video.py                # Classificador de video standalone
│
├── Captura/                    # Modulo de captura de tela
├── data/
│   └── file.pdf                # Documento de saude para RAG
│
├── chromasaude/                # Banco ChromaDB persistido
└── uploads/                    # Arquivos temporarios de upload
```

---

## Como Executar

### Pre-requisitos

- Python 3.10+
- Chave de API da OpenAI

### Instalacao

1. Clone o repositorio:
```bash
git clone <url-do-repositorio>
cd "Tech Challenge 6IADT - Fase 4"
```

2. Instale as dependencias:
```bash
pip install flask openai python-dotenv langchain langchain-openai langchain-community chromadb tensorflow opencv-python pillow pyaudio sounddevice scipy mss pygetwindow
```

3. Configure o arquivo `.env`:
```
OPENAI_API_KEY=sk-proj-sua-chave-aqui
```

4. Crie o banco ChromaDB (primeira execucao):
```bash
python create_db.py
```

5. Execute o chatbot:
```bash
python chatbot.py
```

6. Acesse no navegador:
```
http://localhost:5000
```

---

## Como Usar

1. **Upload de Raio-X:** Clique em "Escolher imagem" e depois em "Analisar Raio-X"
2. **Analise de Video:** Selecione um video de raio-X e clique em "Analisar Video"
3. **Deteccao na Tela:** Abra uma imagem de raio-X no monitor e pergunte "o que tem na minha tela?"
4. **Perguntas de Follow-up:** Apos a classificacao, pergunte "explique mais sobre este diagnostico"
5. **Dispositivo de audio:** Selecione entre "Fone de Ouvido" ou "Microfone do Computador"
6. **Voz:** Marque a caixa "Voz" para ouvir as respostas em audio
7. **Microfone:** Clique no botao do microfone para gravar sua pergunta; clique novamente para parar e enviar
8. **Enviar:** Envia a mensagem digitada no campo de texto
9. **Deletar:** Limpa o historico de mensagens do chat
