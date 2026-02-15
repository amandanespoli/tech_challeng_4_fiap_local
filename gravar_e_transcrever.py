"""
Integra gravador.py e transcrever.py para uso no chatbot.
Grava audio com sounddevice e transcreve com Whisper API.
Processa videos de raio-X para classificacao de frames.
"""
import sys
import threading
from pathlib import Path

# Adiciona pastas ao path para imports
sys.path.insert(0, str(Path(__file__).parent / "Gravador"))
sys.path.insert(0, str(Path(__file__).parent / "Transcrever"))
sys.path.insert(0, str(Path(__file__).parent / "Video"))

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import queue
import cv2
from PIL import Image
import shutil
from collections import Counter

from transcrever import transcrever_audio, ARQUIVO_AUDIO, ARQUIVO_SAIDA
from video import open_video
from xray_classifier import get_classifier


def detectar_dispositivo_audio():
    """Detecta dispositivo de entrada padrão e seus canais suportados"""
    try:
        # Obtém dispositivo de entrada padrão
        device_info = sd.query_devices(kind='input')
        device_id = sd.default.device[0]  # ID do dispositivo de entrada padrão
        max_channels = int(device_info['max_input_channels'])

        # Usa no máximo 2 canais, ou menos se dispositivo não suportar
        channels = min(2, max_channels) if max_channels > 0 else 1

        # Tenta usar taxa nativa do dispositivo, fallback para 16000
        sample_rate = int(device_info.get('default_samplerate', 16000))

        print(f"Dispositivo detectado: {device_info['name']}")
        print(f"Canais suportados: {max_channels}, usando: {channels}")
        print(f"Taxa de amostragem: {sample_rate}")

        return device_id, sample_rate, channels
    except Exception as e:
        print(f"Erro ao detectar dispositivo: {e}. Usando valores padrão.")
        return None, 16000, 1


# Detecta dispositivo automaticamente
ID_MICROFONE, RATE, CHANNELS = detectar_dispositivo_audio()

# Estado global
audio_queue = queue.Queue()
gravacao_frames = []
is_recording = False
recording_thread = None


def _callback(indata, frames, time, status):
    """Callback do sounddevice para captura de audio"""
    if is_recording:
        audio_queue.put(indata.copy())


def iniciar_gravacao():
    """Inicia gravacao em thread separada usando sounddevice"""
    global is_recording, gravacao_frames, recording_thread

    # Limpa fila e frames anteriores
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    is_recording = True
    gravacao_frames = []

    def _gravar():
        global gravacao_frames
        try:
            with sd.InputStream(samplerate=RATE, device=ID_MICROFONE,
                              channels=CHANNELS, callback=_callback, dtype='int16'):
                print(f"Gravando com microfone do computador (device {ID_MICROFONE})...")
                while is_recording:
                    try:
                        data = audio_queue.get(timeout=0.5)
                        gravacao_frames.append(data)
                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"Erro na gravacao: {e}")

    recording_thread = threading.Thread(target=_gravar, daemon=True)
    recording_thread.start()
    return True


def parar_gravacao_e_transcrever():
    """Para gravacao, salva WAV e transcreve usando Whisper API"""
    global is_recording, gravacao_frames

    is_recording = False

    # Aguarda thread finalizar
    if recording_thread:
        recording_thread.join(timeout=2)

    if len(gravacao_frames) == 0:
        print("Nenhum frame gravado")
        return None

    print(f"Processando {len(gravacao_frames)} frames de audio...")

    # Salva WAV na pasta Gravador
    audio_concatenado = np.concatenate(gravacao_frames, axis=0)
    write(str(ARQUIVO_AUDIO), RATE, audio_concatenado)
    print(f"Audio salvo em: {ARQUIVO_AUDIO}")

    # Transcreve usando a funcao do transcrever.py
    transcricao = transcrever_audio(ARQUIVO_AUDIO)
    print(f"Transcricao: {transcricao}")

    # Salva TXT
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
        f.write(transcricao)
    print(f"Transcricao salva em: {ARQUIVO_SAIDA}")

    return transcricao


#################################### VIDEO RAIO-X ####################################

MIN_CONFIDENCE_THRESHOLD = 0.4  # Confianca minima para incluir frame na agregacao
TARGET_CLASSIFICATIONS_PER_SECOND = 2  # Classificacoes por segundo de video
MIN_CLASSIFY_INTERVAL = 3  # Minimo de frames entre classificacoes
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'webm'}


def allowed_video_file(filename):
    """Verifica se o arquivo tem extensao de video permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def extract_xray_region(frame):
    """
    Detecta e extrai a regiao do raio-X de um frame de gravacao de tela.

    Usa threshold de Otsu + deteccao de contornos para encontrar a maior
    regiao retangular (o raio-X) e recorta-la, removendo elementos de UI.
    Retorna o frame original se nenhuma regiao valida for encontrada.
    """
    h, w = frame.shape[:2]
    frame_area = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu separa regioes claras (raio-X) de escuras (fundo/UI)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operacoes morfologicas para unir regioes fragmentadas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return frame

    # Filtrar por area minima (pelo menos 15% do frame)
    min_area = frame_area * 0.15
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid_contours:
        return frame

    largest = max(valid_contours, key=cv2.contourArea)
    x, y, rw, rh = cv2.boundingRect(largest)

    # Se a regiao ocupa >90% do frame, nao recortar (ja e o raio-X direto)
    if (rw * rh) > frame_area * 0.9:
        return frame

    # Margem para nao cortar bordas do raio-X
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    rw = min(w - x, rw + 2 * margin)
    rh = min(h - y, rh + 2 * margin)

    return frame[y:y+rh, x:x+rw]


def enhance_xray_frame(frame):
    """
    Normaliza um frame de raio-X extraido de video para compensar
    artefatos de compressao e diferencas de contraste.

    1. Converte para escala de cinza (raio-X e inerentemente grayscale,
       remove artefatos de cor da compressao de video)
    2. Aplica CLAHE para normalizar contraste (padrao em imagens medicas)
    3. Retorna em BGR (3 canais) para compatibilidade com o pipeline
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE normaliza contraste local, recuperando detalhes perdidos na compressao
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def processar_video_xray(video_path):
    """
    Processa um video de raio-X, classificando frames periodicamente.

    Usa amostragem adaptativa ao FPS do video e votacao ponderada
    por confianca para agregar resultados dos frames classificados.

    Args:
        video_path (str): Caminho absoluto para o arquivo de video.

    Returns:
        dict com final_classification, frame_results, classification_counts, etc.
    """
    classifier = get_classifier()

    if not classifier.is_model_loaded():
        return {
            'success': False,
            'error': 'Modelo de classificacao nao carregado'
        }

    # Abrir video usando open_video do Video/video.py (com fallback non-ASCII)
    cap, temp_dir = open_video(video_path)

    if not cap.isOpened():
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return {
            'success': False,
            'error': f'Nao foi possivel abrir o video: {video_path}'
        }

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Amostragem adaptativa baseada no FPS real do video
        if fps > 0:
            classify_every_n = max(MIN_CLASSIFY_INTERVAL, int(fps / TARGET_CLASSIFICATIONS_PER_SECOND))
        else:
            classify_every_n = 10

        print(f"Video: {fps:.0f} FPS, {total_frames} frames, "
              f"classificando a cada {classify_every_n} frames "
              f"(~{fps/max(classify_every_n, 1):.1f} classificacoes/s)")

        frame_count = 0
        frame_results = []
        classification_names = []
        last_result = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Classificar a cada N frames (intervalo adaptativo ao FPS)
            if frame_count % classify_every_n == 0:
                # Extrair regiao do raio-X (remove UI de gravacoes de tela)
                xray_frame = extract_xray_region(frame)
                # Normalizar contraste e remover artefatos de cor
                xray_frame = enhance_xray_frame(xray_frame)
                rgb_frame = cv2.cvtColor(xray_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                result = classifier.classify(pil_image)

                if result['success']:
                    frame_results.append({
                        'frame_number': frame_count,
                        'class_name': result['class_name'],
                        'confidence': result['confidence'],
                        'all_probabilities': result['all_probabilities']
                    })
                    classification_names.append(result['class_name'])
                    last_result = result
                    print(f"Frame {frame_count}: {result['class_name']} "
                          f"({result['confidence']*100:.1f}%)")

            # Sobrepor resultado da classificacao no frame
            if last_result is not None:
                class_name = last_result['class_name']
                confidence = last_result['confidence']
                label = f"{class_name}: {confidence*100:.1f}%"

                # Fundo escuro para legibilidade
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(frame, (10, 10), (20 + text_w, 40 + text_h),
                              (0, 0, 0), -1)

                # Classe principal em verde
                cv2.putText(frame, label, (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Todas as probabilidades em branco
                y_offset = 70
                for cls_name, prob in last_result['all_probabilities'].items():
                    prob_text = f"{cls_name}: {prob*100:.1f}%"
                    cv2.putText(frame, prob_text, (15, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 25

            # Contador de frames
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                        (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

            cv2.imshow("Classificador de Raio-X", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        if not frame_results:
            return {
                'success': False,
                'error': 'Nenhum frame foi classificado com sucesso'
            }

        # Filtrar frames com confianca baixa (artefatos de compressao)
        reliable_results = [
            fr for fr in frame_results
            if fr['confidence'] >= MIN_CONFIDENCE_THRESHOLD
        ]

        # Fallback: se todos abaixo do threshold, usar todos
        if not reliable_results:
            reliable_results = frame_results

        filtered_count = len(frame_results) - len(reliable_results)
        if filtered_count > 0:
            print(f"Filtrados {filtered_count} frames com confianca "
                  f"abaixo de {MIN_CONFIDENCE_THRESHOLD*100:.0f}%")

        # Votacao ponderada: somar probabilidades por classe
        class_labels = ['Covid-19', 'Normal', 'Pneumonia Viral', 'Pneumonia Bacteriana']
        weighted_scores = {label: 0.0 for label in class_labels}

        for fr in reliable_results:
            for label in class_labels:
                weighted_scores[label] += fr['all_probabilities'].get(label, 0.0)

        # Classe com maior soma ponderada vence
        dominant_class = max(weighted_scores, key=weighted_scores.get)

        # Normalizar para obter probabilidades medias
        total_frames_used = len(reliable_results)
        avg_probabilities = {
            label: weighted_scores[label] / total_frames_used
            for label in class_labels
        }
        avg_confidence = avg_probabilities[dominant_class]

        # Manter class_counts para compatibilidade
        class_counts = Counter(fr['class_name'] for fr in reliable_results)

        print(f"Votacao ponderada: {dominant_class} "
              f"(confianca media: {avg_confidence*100:.1f}%)")

        return {
            'success': True,
            'final_classification': {
                'success': True,
                'class_name': dominant_class,
                'confidence': avg_confidence,
                'all_probabilities': avg_probabilities
            },
            'total_frames_analyzed': len(frame_results),
            'total_frames_reliable': len(reliable_results),
            'total_frames_video': total_frames,
            'fps': fps,
            'frame_results': frame_results,
            'classification_counts': dict(class_counts)
        }

    finally:
        # Limpar arquivo temporario se foi usado (mesmo padrao de video.py)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
