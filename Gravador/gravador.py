import sounddevice as sd
from scipy.io.wavfile import write
import queue
import sys
import numpy as np

# --- CONFIGURAÇÕES QUE FUNCIONARAM NO SEU PC ---
ID_MICROFONE = 5      # O ID que deu certo no teste WDM-KS
RATE = 48000          # Taxa nativa
CHANNELS = 2          # 2 Canais (Estéreo) é o mais seguro para ID 5. 
                      # Se der erro, mude para 4 (já que você testou 4 antes).
ARQUIVO_SAIDA = "gravacao_oficial.wav"
# -----------------------------------------------

# Fila para transferir áudio do driver para o Python sem travar
q = queue.Queue()

def callback(indata, frames, time, status):
    """Essa função é chamada pelo driver a cada fração de segundo"""
    if status:
        print(status, file=sys.stderr)
    # Coloca uma cópia dos dados na fila
    q.put(indata.copy())

print(f"--- GRAVADOR MODERNO (SoundDevice) ---")
print(f"Dispositivo: {ID_MICROFONE} | Taxa: {RATE}")
print("Pressione Ctrl+C para PARAR a gravação.")
print("----------------------------------------")

# Lista para guardar todo o áudio
gravacao_total = []

try:
    # Abre o microfone em modo "Stream" (Fluxo contínuo)
    with sd.InputStream(samplerate=RATE,
                        device=ID_MICROFONE,
                        channels=CHANNELS,
                        callback=callback,
                        dtype='int16'): # int16 é o padrão de CD/WAV
        
        print("🔴 GRAVANDO... (Fale agora!)")
        
        # Loop infinito que mantém o programa rodando
        while True:
            # Pega o áudio da fila e guarda na lista
            # O timeout permite que o Ctrl+C seja detectado
            data = q.get() 
            gravacao_total.append(data)

except KeyboardInterrupt:
    print("\n\n⏹️ Parando gravação...")

except Exception as e:
    print(f"\n❌ ERRO CRÍTICO: {e}")
    if "Invalid number of channels" in str(e):
        print("DICA: Tente mudar a variável CHANNELS para 4 no código.")

# --- SALVAMENTO ---
print("Processando arquivo...")
if len(gravacao_total) > 0:
    # Junta todos os pedacinhos em um único bloco de áudio
    audio_concatenado = np.concatenate(gravacao_total, axis=0)
    
    # Salva usando scipy (mais robusto que a lib wave nativa)
    write(ARQUIVO_SAIDA, RATE, audio_concatenado)
    print(f"✅ SUCESSO! Arquivo salvo em: {ARQUIVO_SAIDA}")
else:
    print("⚠️ Nada foi gravado.")