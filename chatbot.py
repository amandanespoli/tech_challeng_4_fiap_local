from flask import Flask, render_template, request, jsonify
from PIL import Image
import pyaudio
import threading
from openai import OpenAI
import wave
import io
import json
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import mss
import base64
from io import BytesIO
import pygetwindow as gw
import time
import atexit

# LangChain (stack moderno)
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
import logging

# Importar classificador de raio-X
from xray_classifier import get_classifier

# Importar integração gravador + transcrever (microfone do computador) + video
from gravar_e_transcrever import (
    iniciar_gravacao,
    parar_gravacao_e_transcrever,
    processar_video_xray,
    allowed_video_file,
    ALLOWED_VIDEO_EXTENSIONS
)


# Configuração de logging
logging.basicConfig(level=logging.INFO)

# Variáveis globais adicionais
pergunta_num = 0
session_question_count = 0

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Configurações iniciais
app = Flask(__name__)

load_dotenv()

# Configurações de áudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Inicialização do cliente OpenAI
client = OpenAI()

class ChatBot:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.frames_lock = threading.Lock()
        self.is_recording = False
        self.recording_timeout = 30
        self.audio_device_index = None
        self.chat_history = []
        self.settings = self.load_settings()

        # Contexto do último raio-X analisado (para follow-up)
        self.last_xray_result = None
        self.last_xray_timestamp = None

        # Detect and validate audio device
        self._initialize_audio_device()
        
    def load_settings(self):
        settings_file = 'settings.json'
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as file:
                return json.load(file)
        return {
            "selected_voice": "alloy",
            "hear_response": True
        }

    def _initialize_audio_device(self):
        """
        Detect and validate available audio input device.
        Sets self.audio_device_index to the best available device or None.
        """
        try:
            # Try to get system default input device
            default_input = self.p.get_default_input_device_info()

            if default_input and default_input['maxInputChannels'] > 0:
                self.audio_device_index = default_input['index']
                logging.info(f"Audio device detected: {default_input['name']}")
                return

            # Fallback: Search for any available input device
            device_count = self.p.get_device_count()
            for i in range(device_count):
                device_info = self.p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    self.audio_device_index = i
                    logging.info(f"Using fallback audio device: {device_info['name']}")
                    return

            # No input devices found
            self.audio_device_index = None
            logging.warning("No audio input devices detected")

        except Exception as e:
            logging.error(f"Error detecting audio device: {e}")
            self.audio_device_index = None

    def test_audio_device(self):
        """
        Test if the audio device can be opened for recording.
        Returns: (success: bool, error_message: str)
        """
        if self.audio_device_index is None:
            return False, "No audio input device available. Please connect a microphone."

        try:
            # Attempt to open device briefly
            test_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.audio_device_index,
                frames_per_buffer=CHUNK
            )
            test_stream.stop_stream()
            test_stream.close()
            return True, "Audio device is working"
        except Exception as e:
            error_str = str(e)
            # Parse common PortAudio error codes
            if "-9996" in error_str:
                return False, "No microphone found. Please connect a microphone and restart."
            elif "-9997" in error_str:
                return False, "Microphone is already in use by another application."
            elif "-9988" in error_str:
                return False, "Microphone does not support the required audio format."
            else:
                return False, f"Microphone error: {error_str}"

    def transcribe_audio(self, audio_file):
        try:
            audio_file.seek(0)

            # Validate file size
            audio_file.seek(0, 2)  # Seek to end
            file_size = audio_file.tell()
            audio_file.seek(0)  # Reset to start

            if file_size < 1000:  # Less than 1KB
                logging.error(f"Audio file too small: {file_size} bytes")
                return ""

            logging.info(f"Transcribing audio: {file_size} bytes")

            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

            logging.info(f"Transcription: {response.text}")
            return response.text
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return ""

    def get_response(self, user_message):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """You are a helpful assistant that classifies user messages.
                        Your answer must be a JSON with two fields: "type" and "content".
                        The "content" field must ALWAYS be a string containing the user's question or message.

                        Classification rules:
                        - If it's a general question not related to health, type is 'normal' and content is your answer.
                        - If the user asks about health topics (saúde, doenças, sintomas, tratamentos, epidemiologia, etc.), type is 'saude' and content is the user's question as a string.
                        - If the user asks to click or point to something, type is 'click' and content must be in english starting with 'point to the...'.
                        - If the user asks about the screen or image, type is 'image' and content is the user's question as a string.
                        - If the user asks about a previous X-ray result, diagnosis, classification result, or wants more information about a detected condition (covid, pneumonia, etc.), type is 'xray_followup' and content is the user's question as a string.
                        - Keywords for xray_followup: "diagnóstico", "raio-x", "resultado", "classificação", "explique mais", "o que significa", "covid", "pneumonia", "pulmão", "radiografia", "condição detectada".

                        Example response: {"type": "saude", "content": "Qual a definição e epidemiologia?"}
                        Example response: {"type": "xray_followup", "content": "Explique mais sobre a pneumonia detectada"}"""},
                    {"role": "assistant", "content": "\n".join(self.chat_history)},
                    {"role": "user", "content": user_message}
                ]
            )

            json_response = json.loads(response.choices[0].message.content)

            if json_response.get('type') == 'normal':
                return {'type': 'normal', 'content': json_response.get('content')}

            elif json_response.get('type') == 'saude':
                rag_response = self.get_ragsaude_response(json_response.get('content'))
                # Extrair apenas o conteúdo da resposta do RAG - Saúde
                if isinstance(rag_response, dict):
                    return {'type': 'saude', 'content': rag_response.get('content', '')}
                return {'type': 'saude', 'content': str(rag_response)}
            
            elif json_response.get('type') == 'image':
                resposta = self.ler_tela(json_response.get('content'))
                # Verificar se é raio-X detectado na tela
                if isinstance(resposta, dict) and resposta.get('is_xray'):
                    return {
                        'type': 'xray_screen',
                        'classification': resposta['classification'],
                        'health_info': resposta['health_info']
                    }
                return {'type': 'image', 'content': resposta}

            elif json_response.get('type') == 'xray_followup':
                followup = self.get_xray_followup_response(json_response.get('content'))
                return {'type': 'xray_followup', 'content': followup}

        except Exception as e:
            return {'type': 'error', 'content': f"Sorry, I couldn't get a response. Error: {e}"}

#################################### SAÚDE ####################################
    def get_ragsaude_response(self, question):
        # Garantir que question é uma string
        if isinstance(question, dict):
            question = question.get('content', str(question))
        if not isinstance(question, str):
            question = str(question)

        # Caminho para a base de dados do Chroma
        CHROMA_PATH_SAUDE = "chromasaude"

        api_key = os.getenv('OPENAI_API_KEY')
        # Inicialização do Chroma e do modelo de embedding
        embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
        db = Chroma(persist_directory=CHROMA_PATH_SAUDE, embedding_function=embedding_function)

        # Pesquisar no banco de dados
        results = db.similarity_search_with_relevance_scores(question, k=5)
        print("Resultados de relevância:", results)
        print_formatted_results(results)

        # Verifique se há resultados relevantes
        if len(results) == 0 or results[0][1] < 0.3:
            print("Nenhum resultado relevante encontrado ou abaixo do limiar de 0.3.")
            return {'type': 'saude', 'content': "Desculpe, não encontrei informações relevantes nos documentos carregados."}

        # Estrutura do prompt com o contexto preenchido
        PROMPT_TEMPLATE_SAUDE = """
        # Role
        Você é um assistente virtual de saúde, especializado em fornecer informações educativas sobre saúde, bem-estar e qualidade de vida.

        # Task
        Sua tarefa é interpretar a pergunta do usuário e fornecer uma resposta clara, precisa, educada e direta, mantendo um tom profissional e acolhedor.

        # Specifics
        - A resposta deve conter no máximo 1200 caracteres. Não ultrapasse esse limite.
        - Utilize apenas as informações contidas no contexto fornecido sobre saúde.
        - Se não houver informação suficiente no contexto para responder à pergunta, diga: "Desculpe, mas não consigo ajudar com as informações disponíveis. Por favor, consulte um profissional de saúde."
        - IMPORTANTE: Sempre inclua uma recomendação para que o usuário consulte um profissional de saúde (médico, enfermeiro, nutricionista, etc.) para diagnósticos, tratamentos ou orientações personalizadas.
        - Nunca forneça diagnósticos médicos ou prescrições de medicamentos.
        - Seja empático e compreensivo com as preocupações de saúde do usuário.

        # Context
        Use o seguinte contexto para responder à questão da forma mais clara e precisa possível.
        Contexto: {context}

        Pergunta: {question}
        Resposta:

        """
        # Concatenar o conteúdo dos documentos relevantes
        conteudo = "\n\n".join([doc.page_content for doc, _score in results])
        print("Conteúdo extraído para o contexto:", conteudo)

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_SAUDE)
        prompt = prompt_template.format(context=conteudo, question=question)

        completion = client.chat.completions.create(
            temperature=0.5,
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
        )

        return {'type': 'saude', 'content': completion.choices[0].message.content}

#################################### RAIO-X FOLLOW-UP ####################################
    def get_xray_followup_response(self, question):
        """
        Responde perguntas de follow-up sobre o último raio-X analisado.
        Combina o contexto da classificação com busca no ChromaDB.
        """
        # Verificar se há contexto de raio-X recente
        if self.last_xray_result is None:
            return "Não encontrei nenhuma análise de raio-X recente. Por favor, faça o upload de um raio-X ou pergunte sobre a sua tela com um raio-X aberto."

        # Verificar se o contexto não está muito antigo (30 minutos)
        if time.time() - self.last_xray_result.get('timestamp', 0) > 1800:
            return "A análise de raio-X anterior já expirou (mais de 30 minutos). Por favor, faça uma nova análise."

        # Obter classificação anterior
        classification = self.last_xray_result.get('classification', {})
        class_name = classification.get('class_name', 'desconhecida')
        confidence = classification.get('confidence', 0)

        # Criar contexto enriquecido para a busca
        enriched_query = f"{class_name} {question}"

        # Buscar informações adicionais no ChromaDB
        rag_response = self.get_ragsaude_response(enriched_query)
        additional_info = rag_response.get('content', '') if isinstance(rag_response, dict) else str(rag_response)

        # Construir resposta contextualizada
        FOLLOWUP_PROMPT = f"""Com base na análise de raio-X anterior que detectou {class_name} com {confidence*100:.1f}% de confiança,
responda a seguinte pergunta do usuário de forma clara e educativa.

Contexto da análise anterior:
- Classificação: {class_name}
- Confiança: {confidence*100:.1f}%
- Informações de saúde: {self.last_xray_result.get('health_info', 'Não disponível')}

Informações adicionais do banco de dados:
{additional_info}

Pergunta do usuário: {question}

Responda de forma clara, empática e sempre inclua a recomendação de consultar um profissional de saúde.
IMPORTANTE: Este é apenas um resultado informativo e NÃO substitui o diagnóstico de um médico."""

        completion = client.chat.completions.create(
            temperature=0.5,
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": FOLLOWUP_PROMPT},
                {"role": "user", "content": question},
            ],
        )

        return completion.choices[0].message.content

#######################################################################################################

    def minimize_browser_windows(self):
        """Minimiza janelas de navegadores para captura de tela limpa"""
        browser_keywords = ['chrome', 'firefox', 'edge', 'opera', 'brave', 'safari', 'localhost', '127.0.0.1']
        minimized_windows = []

        try:
            for window in gw.getAllWindows():
                title_lower = window.title.lower()
                for keyword in browser_keywords:
                    if keyword in title_lower and not window.isMinimized:
                        window.minimize()
                        minimized_windows.append(window)
                        logging.info(f"Minimized window: {window.title}")
                        break
        except Exception as e:
            logging.error(f"Error minimizing windows: {e}")

        return minimized_windows

    def restore_windows(self, windows):
        """Restaura janelas que foram minimizadas"""
        for window in windows:
            try:
                window.restore()
                window.activate()
                logging.info(f"Restored window: {window.title}")
            except Exception as e:
                logging.error(f"Error restoring window {window.title}: {e}")

    def ler_tela(self, message):
        """
        Captures the screen and analyzes it.
        If the screen contains an X-ray, classifies it automatically.
        Works with single or multiple monitor setups.
        Automatically minimizes browser windows before capture.
        """
        # Minimizar janelas do navegador antes da captura
        minimized_windows = self.minimize_browser_windows()
        time.sleep(0.4)  # Aguardar animação de minimização (400ms)

        try:
            image = self.capture_screen(width=1920, height=1080)

            # Verificar se a tela contém um raio-X
            classifier = get_classifier()

            if classifier.is_model_loaded() and classifier.is_xray_image(image):
                # Detectou raio-X na tela - classificar automaticamente
                result = classifier.classify(image)

                if result['success']:
                    # Buscar informações de saúde
                    disease_query = classifier.get_disease_query(result['class_name'])
                    health_info = self.get_ragsaude_response(disease_query)
                    health_content = health_info.get('content', '') if isinstance(health_info, dict) else ''

                    # Armazenar contexto para follow-up
                    self.last_xray_result = {
                        'classification': result,
                        'health_info': health_content,
                        'timestamp': time.time()
                    }
                    self.last_xray_timestamp = time.time()

                    # Retornar dados estruturados (igual ao /upload_xray)
                    return {
                        'is_xray': True,
                        'classification': result,
                        'health_info': health_content
                    }

            # Não é raio-X - comportamento original
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                }
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message.content

        except Exception as e:
            logging.error(f"Error capturing or analyzing screen: {e}")
            return f"Desculpe, não consegui capturar a tela. Erro: {str(e)}"
        finally:
            # Restaurar janelas do navegador (sempre executado)
            time.sleep(0.2)  # Pequeno delay antes de restaurar
            self.restore_windows(minimized_windows)

    def capture_screen(self, width=None, height=None, monitor_index=None):
        """
        Capture a screenshot from specified monitor or primary monitor.

        Args:
            width: Width of capture region (None = full monitor width)
            height: Height of capture region (None = full monitor height)
            monitor_index: Which monitor to capture (None = auto-select best available)
                          1 = primary, 2 = secondary, etc.

        Returns:
            PIL Image object

        Raises:
            ValueError: If no monitors are detected
        """
        with mss.mss() as sct:
            monitors = sct.monitors

            # Validate monitors exist
            if len(monitors) < 2:
                raise ValueError("No monitors detected on this system.")

            # Auto-select monitor if not specified
            if monitor_index is None:
                # Prefer secondary monitor if available, otherwise use primary
                if len(monitors) >= 3:
                    monitor_index = 2  # Secondary monitor
                    logging.info("Using secondary monitor for screen capture")
                else:
                    monitor_index = 1  # Primary monitor
                    logging.info("Using primary monitor for screen capture (no secondary monitor detected)")

            # Validate selected monitor exists
            if monitor_index >= len(monitors):
                # Fallback to primary if requested monitor doesn't exist
                logging.warning(f"Monitor {monitor_index} not found, falling back to primary monitor")
                monitor_index = 1

            selected_monitor = monitors[monitor_index]

            # Build capture region
            capture_region = {
                "top": selected_monitor["top"],
                "left": selected_monitor["left"],
                "width": width if width else selected_monitor["width"],
                "height": height if height else selected_monitor["height"]
            }

            logging.info(f"Capturing from monitor {monitor_index}: {selected_monitor['width']}x{selected_monitor['height']}")

            screenshot = sct.grab(capture_region)
            return Image.frombytes("RGB", screenshot.size, screenshot.rgb)

    def cleanup(self):
        """Clean up PyAudio resources on shutdown"""
        try:
            if self.is_recording:
                self.is_recording = False
                time.sleep(0.5)  # Wait for thread to finish
            self.p.terminate()
            logging.info("Audio resources cleaned up")
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# Função para formatar resultados de busca
def print_formatted_results(results):
    formatted_results = []
    for i, (doc, score) in enumerate(results, 1):
        result = {
            "number": i,
            "length": len(doc.page_content),
            "score": score,
            "content": doc.page_content
        }
        formatted_results.append(result)

    # Save results to a JSON file
    with open('static/pdf_results.json', 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)

# Rota para análise de raio-X
@app.route('/upload_xray', methods=['POST'])
def upload_xray():
    """
    Rota para upload e análise de imagem de raio-X.
    Retorna classificação da doença e informações de saúde relacionadas.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    if not allowed_image_file(file.filename):
        return jsonify({
            'error': 'Formato inválido',
            'message': 'Por favor, envie uma imagem (PNG, JPG, JPEG, GIF, BMP ou WEBP)'
        }), 400

    try:
        # Carregar imagem com PIL
        image = Image.open(file.stream)

        # Obter classificador (singleton)
        classifier = get_classifier()

        if not classifier.is_model_loaded():
            return jsonify({
                'error': 'Modelo não disponível',
                'message': 'O modelo de classificação de raio-X não foi carregado.'
            }), 500

        # Verificar se é uma imagem de raio-X usando GPT-4o Vision
        is_xray = classifier.is_xray_image(image)

        if not is_xray:
            return jsonify({
                'type': 'not_xray',
                'content': 'A imagem enviada não parece ser um raio-X de tórax. Por favor, envie uma radiografia de tórax válida.'
            })

        # Classificar o raio-X
        result = classifier.classify(image)

        if not result['success']:
            return jsonify({
                'error': 'Falha na classificação',
                'message': result.get('error', 'Erro desconhecido')
            }), 500

        # Buscar informações de saúde no ChromaDB
        disease_query = classifier.get_disease_query(result['class_name'])
        health_info = chatbot.get_ragsaude_response(disease_query)
        health_content = health_info.get('content', '') if isinstance(health_info, dict) else str(health_info)

        # Armazenar contexto para follow-up
        chatbot.last_xray_result = {
            'classification': result,
            'health_info': health_content,
            'timestamp': time.time()
        }
        chatbot.last_xray_timestamp = time.time()

        # Adicionar ao histórico do chat
        chatbot.chat_history.append(f"[Raio-X Analisado]: {result['class_name']} ({result['confidence']*100:.1f}% confiança)")

        logging.info(f"Raio-X classificado: {result['class_name']} ({result['confidence']*100:.1f}%)")

        return jsonify({
            'type': 'xray',
            'classification': result,
            'health_info': health_content,
            'follow_up_hint': 'Você pode perguntar mais sobre este diagnóstico.'
        })

    except Exception as e:
        logging.error(f"Erro ao processar raio-X: {e}")
        return jsonify({
            'error': 'Erro ao processar imagem',
            'message': str(e)
        }), 500

# Rota para análise de vídeo de raio-X
@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    Rota para upload e analise de video de raio-X.
    Processa frames do video, classifica e retorna resultado agregado
    com informacoes de saude relacionadas.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    if not allowed_video_file(file.filename):
        return jsonify({
            'error': 'Formato invalido',
            'message': f'Por favor, envie um video ({", ".join(ALLOWED_VIDEO_EXTENSIONS)})'
        }), 400

    video_path = None
    try:
        # Salvar video temporariamente na pasta uploads
        filename = secure_filename(file.filename)
        video_path = os.path.join('uploads', filename)
        file.save(video_path)

        # Processar video (funcao de gravar_e_transcrever.py)
        result = processar_video_xray(video_path)

        # Remover arquivo temporario
        try:
            os.remove(video_path)
        except OSError:
            pass

        if not result['success']:
            return jsonify({
                'error': 'Falha no processamento do video',
                'message': result.get('error', 'Erro desconhecido')
            }), 500

        # Extrair classificacao final (agregada)
        final_classification = result['final_classification']

        # Buscar informacoes de saude no ChromaDB (mesmo padrao do /upload_xray)
        disease_query = get_classifier().get_disease_query(final_classification['class_name'])
        health_info = chatbot.get_ragsaude_response(disease_query)
        health_content = health_info.get('content', '') if isinstance(health_info, dict) else str(health_info)

        # Armazenar contexto para follow-up (mesmo padrao do /upload_xray)
        chatbot.last_xray_result = {
            'classification': final_classification,
            'health_info': health_content,
            'timestamp': time.time()
        }
        chatbot.last_xray_timestamp = time.time()

        # Adicionar ao historico do chat
        chatbot.chat_history.append(
            f"[Video Raio-X Analisado]: {final_classification['class_name']} "
            f"({final_classification['confidence']*100:.1f}% confianca, "
            f"{result['total_frames_analyzed']} frames analisados)"
        )

        logging.info(
            f"Video raio-X classificado: {final_classification['class_name']} "
            f"({final_classification['confidence']*100:.1f}%) - "
            f"{result['total_frames_analyzed']} frames"
        )

        return jsonify({
            'type': 'video_xray',
            'classification': final_classification,
            'health_info': health_content,
            'video_stats': {
                'total_frames_video': result['total_frames_video'],
                'total_frames_analyzed': result['total_frames_analyzed'],
                'fps': result.get('fps', 0),
                'classification_counts': result['classification_counts']
            },
            'frame_results': result['frame_results'],
            'follow_up_hint': 'Voce pode perguntar mais sobre este diagnostico.'
        })

    except Exception as e:
        logging.error(f"Erro ao processar video de raio-X: {e}")
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
        except OSError:
            pass
        return jsonify({
            'error': 'Erro ao processar video',
            'message': str(e)
        }), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('text', '')
        #voice = data.get('voice', 'alloy')
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Usando sempre 'alloy'
            input=text,
            response_format="wav"
        )
        
        audio_data = response.content
        audio_file = io.BytesIO(audio_data)
        audio_file.seek(0)
        
        # Play the WAV audio using PyAudio
        wf = wave.open(audio_file, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return jsonify({
            'status': 'success',
            'audio': audio_file
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/save_settings', methods=['POST'])
def save_settings():
    data = request.json
    chatbot.settings.update(data)
    with open('settings.json', 'w') as file:
        json.dump(chatbot.settings, file)
    return jsonify({'status': 'success'})

# Instância global do chatbot
chatbot = ChatBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    chatbot.chat_history.append(f"You: {message}")
    response = chatbot.get_response(message)

    if response.get('type') == 'xray_screen':
        history_text = f"[Raio-X Detectado]: {response['classification']['class_name']}"
    else:
        history_text = response.get('content', str(response))
    chatbot.chat_history.append(f"Bot: {history_text}")

    return jsonify(response)

@app.route('/start_recording', methods=['POST'])
def start_recording():
    if chatbot.is_recording:
        return jsonify({'error': 'Already recording'}), 400

    # Validate audio device availability
    if chatbot.audio_device_index is None:
        return jsonify({
            'error': 'No audio input device available',
            'message': 'Please connect a microphone and restart the application.'
        }), 400

    # Test device before starting
    device_ok, device_msg = chatbot.test_audio_device()
    if not device_ok:
        return jsonify({
            'error': 'Audio device not accessible',
            'message': device_msg
        }), 400

    # Initialize recording state
    chatbot.is_recording = True
    with chatbot.frames_lock:
        chatbot.frames = []

    def record_audio():
        audio_stream = None
        start_time = time.time()

        try:
            # Open audio stream with validated device
            audio_stream = chatbot.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=chatbot.audio_device_index,
                frames_per_buffer=CHUNK
            )

            logging.info("Recording started")

            while chatbot.is_recording:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > chatbot.recording_timeout:
                    logging.warning(f"Recording stopped: {chatbot.recording_timeout}s timeout")
                    chatbot.is_recording = False
                    break

                try:
                    # Read audio chunk
                    data = audio_stream.read(CHUNK, exception_on_overflow=False)
                    with chatbot.frames_lock:
                        chatbot.frames.append(data)
                except OSError as e:
                    if chatbot.is_recording:
                        logging.error(f"Audio read error: {e}")
                        chatbot.is_recording = False
                    break

            logging.info(f"Recording stopped: {len(chatbot.frames)} frames captured")

        except Exception as e:
            logging.error(f"Recording thread error: {e}")
            chatbot.is_recording = False
        finally:
            if audio_stream:
                try:
                    audio_stream.stop_stream()
                    audio_stream.close()
                except Exception as e:
                    logging.error(f"Error closing audio stream: {e}")

    # Start recording thread
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

    return jsonify({'status': 'Recording started'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    if not chatbot.is_recording:
        return jsonify({'error': 'Not currently recording'}), 400

    # Signal thread to stop
    chatbot.is_recording = False

    # Wait for thread to finish (max 2 seconds)
    max_wait = 2.0
    wait_interval = 0.1
    elapsed = 0
    while elapsed < max_wait:
        time.sleep(wait_interval)
        elapsed += wait_interval

    # Get frames safely
    with chatbot.frames_lock:
        frames_copy = chatbot.frames[:]

    # Validate recording length
    if len(frames_copy) == 0:
        return jsonify({
            'error': 'No audio recorded',
            'message': 'Recording failed. Please check your microphone and try again.'
        }), 400

    # Calculate duration
    frame_count = len(frames_copy)
    duration = (frame_count * CHUNK) / RATE

    if duration < 0.5:
        return jsonify({
            'error': 'Recording too short',
            'message': f'Recording was only {duration:.1f} seconds. Please speak for at least 1 second.'
        }), 400

    try:
        # Create WAV file
        audio_data = b''.join(frames_copy)
        audio_file = io.BytesIO()
        with wave.open(audio_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(chatbot.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        audio_file.name = "output.wav"

        logging.info(f"Audio file created: {duration:.1f}s, {len(audio_data)} bytes")

        # Transcribe audio
        transcript = chatbot.transcribe_audio(audio_file)
        if not transcript:
            return jsonify({
                'error': 'Transcription failed',
                'message': 'Could not understand the audio. Please speak clearly and try again.'
            }), 400

        # Get bot response
        chatbot.chat_history.append(f"You: {transcript}")
        response = chatbot.get_response(transcript)

        # Tratar diferentes tipos de resposta para o histórico
        if response.get('type') == 'xray_screen':
            history_text = f"[Raio-X Detectado na Tela]: {response['classification']['class_name']}"
        else:
            history_text = response.get('content', str(response))
        chatbot.chat_history.append(f"Bot: {history_text}")

        return jsonify({
            'transcript': transcript,
            'response': response
        })

    except Exception as e:
        logging.error(f"Error processing recording: {e}")
        return jsonify({
            'error': 'Processing failed',
            'message': f'Failed to process recording: {str(e)}'
        }), 500


@app.route('/start_recording_mic', methods=['POST'])
def start_recording_mic():
    """Inicia gravacao usando microfone do computador (sounddevice)"""
    try:
        iniciar_gravacao()
        return jsonify({'status': 'Recording started with computer mic'})
    except Exception as e:
        logging.error(f"Error starting computer mic recording: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stop_recording_mic', methods=['POST'])
def stop_recording_mic():
    """Para gravacao e transcreve usando microfone do computador"""
    try:
        transcript = parar_gravacao_e_transcrever()
        if not transcript:
            return jsonify({'error': 'No audio recorded'}), 400

        # Usa o mesmo fluxo de resposta do chatbot
        response = chatbot.get_response(transcript)
        chatbot.chat_history.append(f"You: {transcript}")

        # Tratar diferentes tipos de resposta para o historico
        if response.get('type') == 'xray_screen':
            history_text = f"[Raio-X Detectado na Tela]: {response['classification']['class_name']}"
        else:
            history_text = response.get('content', str(response))
        chatbot.chat_history.append(f"Bot: {history_text}")

        return jsonify({
            'transcript': transcript,
            'response': response
        })
    except Exception as e:
        logging.error(f"Error processing computer mic recording: {e}")
        return jsonify({'error': str(e)}), 500


def cleanup_on_exit():
    """Cleanup function called on program exit"""
    logging.info("Shutting down...")
    chatbot.cleanup()

atexit.register(cleanup_on_exit)

if __name__ == '__main__':
    app.run(debug=True)