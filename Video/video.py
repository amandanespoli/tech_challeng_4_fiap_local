import sys
import os
import cv2
import tempfile
import shutil
from PIL import Image

# Adicionar diretorio pai ao sys.path para importar xray_classifier
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, PARENT_DIR)

from xray_classifier import get_classifier

# Configuracao
VIDEO_PATH = os.path.join(SCRIPT_DIR, 'Video 2026-02-07 at 11.29.01.mp4')
CLASSIFY_EVERY_N_FRAMES = 30  # ~1 classificacao por segundo a 30fps


def open_video(path):
    """Abre video com fallback para paths com caracteres nao-ASCII (Windows)."""
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap, None
    # Fallback: copiar para path temporario ASCII-only
    print("Fallback: copiando video para caminho temporario...")
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'video.mp4')
    shutil.copy2(path, temp_path)
    cap = cv2.VideoCapture(temp_path)
    return cap, temp_dir


def main():
    # Carregar modelo de classificacao
    print("Carregando modelo de classificacao de raio-X...")
    classifier = get_classifier()

    if not classifier.is_model_loaded():
        print("ERRO: Modelo nao foi carregado. Verifique o caminho do modelo.")
        sys.exit(1)

    print("Modelo carregado com sucesso!")

    # Abrir video local
    cap, temp_dir = open_video(VIDEO_PATH)

    if not cap.isOpened():
        print(f"ERRO: Nao foi possivel abrir o video: {VIDEO_PATH}")
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {fps:.0f} FPS, {width}x{height}, {total_frames} frames")
    print("Pressione 'q' para sair.\n")

    frame_count = 0
    last_result = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Classificar a cada N frames
        if frame_count % CLASSIFY_EVERY_N_FRAMES == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            result = classifier.classify(pil_image)

            if result['success']:
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

            # Classe principal
            cv2.putText(frame, label, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Todas as probabilidades
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

    # Limpar arquivo temporario se foi usado
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Resumo final
    if last_result is not None:
        print(f"\n--- Resultado Final ---")
        print(f"Classificacao: {last_result['class_name']} "
              f"({last_result['confidence']*100:.1f}%)")
        print("Probabilidades:")
        for cls_name, prob in last_result['all_probabilities'].items():
            print(f"  {cls_name}: {prob*100:.1f}%")
    else:
        print("\nNenhuma classificacao realizada.")


if __name__ == '__main__':
    main()
