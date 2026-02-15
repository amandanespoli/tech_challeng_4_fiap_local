"""
Modulo de Classificacao de Raio-X Pulmonar
==========================================
Este modulo fornece funcionalidades para classificar imagens de raio-X
de torax em 4 categorias de doencas pulmonares usando um modelo CNN
treinado com Transfer Learning (ResNet50).

Classes de classificacao:
- 0: Covid-19
- 1: Normal
- 2: Pneumonia Viral
- 3: Pneumonia Bacteriana
"""

import os
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import logging
from dotenv import load_dotenv

# Carregar variaveis de ambiente
load_dotenv()

# Configuracao de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Departamento_Medico", "melhor_modelo.keras")
IMAGE_SIZE = (256, 256)

CLASS_LABELS = {
    0: 'Covid-19',
    1: 'Normal',
    2: 'Pneumonia Viral',
    3: 'Pneumonia Bacteriana'
}

# Queries otimizadas para ChromaDB por tipo de doenca
DISEASE_QUERIES = {
    'Covid-19': 'covid-19 coronavirus sintomas tratamento doenca pulmonar respiratoria',
    'Normal': 'saude pulmonar prevencao doencas respiratorias cuidados pulmao saudavel',
    'Pneumonia Viral': 'pneumonia viral infeccao pulmonar virus sintomas tratamento respiratorio',
    'Pneumonia Bacteriana': 'pneumonia bacteriana infeccao bacteria antibioticos tratamento pulmonar'
}


class XRayClassifier:
    """
    Classificador de imagens de raio-X pulmonar.

    Utiliza um modelo Keras treinado com Transfer Learning (ResNet50)
    para classificar imagens em 4 categorias de doencas pulmonares.
    """

    def __init__(self):
        """Inicializa o classificador carregando o modelo e o cliente OpenAI."""
        self.model = None
        self.client = None
        self._load_model()
        self._initialize_openai_client()

    def _load_model(self):
        """
        Carrega o modelo Keras do arquivo.

        O modelo e carregado de forma lazy para evitar overhead
        se nao for utilizado.
        """
        try:
            # Suprimir warnings do TensorFlow
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')

            from tensorflow.keras.models import load_model

            if not os.path.exists(MODEL_PATH):
                logger.error(f"Modelo nao encontrado em: {MODEL_PATH}")
                return

            self.model = load_model(MODEL_PATH, compile=False)
            logger.info(f"Modelo de raio-X carregado com sucesso de: {MODEL_PATH}")

            # Recompilar manualmente com os parâmetros corretos
            logger.info("Recompilando modelo...")
            self.model.compile(
                optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Modelo de raio-X carregado com sucesso!")
            logger.info(f"Arquitetura do modelo: Input {self.model.input_shape}, Output {self.model.output_shape}")
        
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de raio-X: {e}")
            self.model = None

    def _initialize_openai_client(self):
        """Inicializa o cliente OpenAI para deteccao de raio-X."""
        try:
            from openai import OpenAI
            self.client = OpenAI()
            logger.info("Cliente OpenAI inicializado para deteccao de raio-X")
        except Exception as e:
            logger.error(f"Erro ao inicializar cliente OpenAI: {e}")
            self.client = None

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Pre-processa uma imagem PIL para entrada no modelo.

        Args:
            image: Imagem PIL a ser processada

        Returns:
            numpy.ndarray: Imagem preprocessada com shape (1, 256, 256, 3)
        """
        # Converter para RGB se necessario
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Redimensionar para o tamanho esperado pelo modelo
        image = image.resize(IMAGE_SIZE)

        # Converter para array numpy
        img_array = np.array(image)

        # Normalizar valores de pixel para [0, 1]
        img_array = img_array / 255.0

        # Adicionar dimensao de batch: (1, 256, 256, 3)
        img_array = img_array.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

        return img_array

    def classify(self, image: Image.Image) -> dict:
        """
        Classifica uma imagem de raio-X.

        Args:
            image: Imagem PIL do raio-X

        Returns:
            dict: Dicionario com resultado da classificacao:
                - success: bool indicando sucesso
                - class_name: nome da classe detectada
                - class_id: ID numerico da classe
                - confidence: confianca da predicao (0-1)
                - all_probabilities: probabilidades para todas as classes
                - error: mensagem de erro (se aplicavel)
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Modelo de classificacao nao carregado'
            }

        try:
            # Pre-processar imagem
            processed = self.preprocess_image(image)

            # Fazer predicao
            predictions = self.model.predict(processed, verbose=0)

            # Obter classe com maior probabilidade
            class_id = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][class_id])
            class_name = CLASS_LABELS[class_id]

            # Construir dicionario de probabilidades
            all_probabilities = {
                CLASS_LABELS[i]: float(predictions[0][i])
                for i in range(len(CLASS_LABELS))
            }

            logger.info(f"Raio-X classificado: {class_name} ({confidence*100:.1f}%)")

            return {
                'success': True,
                'class_name': class_name,
                'class_id': class_id,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }

        except Exception as e:
            logger.error(f"Erro na classificacao: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def is_xray_image(self, image: Image.Image) -> bool:
        """
        Detecta se uma imagem e um raio-X de torax usando GPT-4o Vision.

        Esta funcao utiliza o modelo de visao da OpenAI para determinar
        se a imagem fornecida e uma radiografia de torax, evitando
        classificar erroneamente imagens que nao sao raio-X.

        Args:
            image: Imagem PIL a ser analisada

        Returns:
            bool: True se a imagem for um raio-X de torax, False caso contrario
        """
        if self.client is None:
            logger.warning("Cliente OpenAI nao disponivel para deteccao de raio-X")
            return False

        try:
            # Converter imagem para base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Consultar GPT-4o Vision
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analise esta imagem e determine se e um raio-X (radiografia) de torax/pulmao.

Um raio-X de torax tipicamente mostra:
- Imagem medica em tons de cinza
- Costelas e campos pulmonares visiveis
- Silhueta do coracao no centro
- Aparencia diagnostica/medica clara

Responda APENAS com 'YES' se for um raio-X de torax, ou 'NO' se nao for.
Nao inclua nenhum outro texto na resposta."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                }
                            }
                        ]
                    }
                ],
                max_tokens=10
            )

            answer = response.choices[0].message.content.strip().upper()
            is_xray = answer == 'YES'

            logger.info(f"Deteccao de raio-X: {'Sim' if is_xray else 'Nao'}")
            return is_xray

        except Exception as e:
            logger.error(f"Erro na deteccao de raio-X: {e}")
            return False

    def get_disease_query(self, disease_class: str) -> str:
        """
        Gera uma query otimizada para busca no ChromaDB baseada na doenca detectada.

        Args:
            disease_class: Nome da classe da doenca (ex: 'Covid-19')

        Returns:
            str: Query otimizada para busca semantica
        """
        return DISEASE_QUERIES.get(
            disease_class,
            f"informacoes sobre {disease_class} doenca pulmonar"
        )

    def is_model_loaded(self) -> bool:
        """Verifica se o modelo foi carregado com sucesso."""
        return self.model is not None

    def get_class_labels(self) -> dict:
        """Retorna o mapeamento de IDs para nomes de classes."""
        return CLASS_LABELS.copy()


# Instancia global para uso no chatbot (carregamento lazy)
_classifier_instance = None

def get_classifier() -> XRayClassifier:
    """
    Retorna a instancia global do classificador.

    Utiliza padrao singleton para evitar carregar o modelo
    multiplas vezes.
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = XRayClassifier()
    return _classifier_instance
