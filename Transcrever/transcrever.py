"""
Transcreve audios da pasta Gravador usando API Whisper da OpenAI.
Salva a transcricao em transcricao.txt (substitui a cada execucao).
"""
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Carrega variaveis de ambiente (.env na raiz do projeto)
load_dotenv(Path(__file__).parent.parent / ".env")

# Cliente OpenAI
client = OpenAI()

# Caminhos
PASTA_GRAVADOR = Path(__file__).parent.parent / "Gravador"
ARQUIVO_AUDIO = PASTA_GRAVADOR / "gravacao_oficial.wav"
ARQUIVO_SAIDA = Path(__file__).parent / "transcricao.txt"


def transcrever_audio(caminho_audio):
    """Transcreve um arquivo de audio usando API Whisper da OpenAI."""
    with open(caminho_audio, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text


def main():
    # Verifica se arquivo existe
    if not ARQUIVO_AUDIO.exists():
        print(f"Erro: Arquivo de audio nao encontrado: {ARQUIVO_AUDIO}")
        return

    print(f"Transcrevendo: {ARQUIVO_AUDIO}")

    # Transcreve
    transcricao = transcrever_audio(ARQUIVO_AUDIO)

    # Salva em TXT (substitui se existir)
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
        f.write(transcricao)

    print(f"Transcricao salva em: {ARQUIVO_SAIDA}")
    print(f"\nTexto transcrito:\n{transcricao}")


if __name__ == "__main__":
    main()
