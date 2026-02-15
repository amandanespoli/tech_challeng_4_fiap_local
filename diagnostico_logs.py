"""
Diagnóstico Detalhado - Ver Logs Completos do Carregamento
"""

import logging
import sys

# Configurar logging para mostrar TUDO
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s',
    stream=sys.stdout
)

print("=" * 80)
print("🔍 DIAGNÓSTICO DETALHADO - CARREGAMENTO DO MODELO")
print("=" * 80)

# 1. Verificar versões
print("\n1️⃣ VERSÕES INSTALADAS:")
print("-" * 80)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    import keras
    print(f"✅ Keras: {keras.__version__}")
except ImportError as e:
    print(f"⚠️  Keras standalone: {e}")

# 2. Verificar se arquivo do modelo existe
print("\n2️⃣ VERIFICANDO ARQUIVO DO MODELO:")
print("-" * 80)

import os

model_paths = [
    os.path.join("Departamento_Medico", "melhor_modelo.keras"),
    os.path.join("Departamento_Médico", "melhor_modelo.keras"),
]

model_found = None
for path in model_paths:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ Arquivo encontrado: {path}")
        print(f"   Tamanho: {size:.2f} MB")
        model_found = path
        break

if not model_found:
    print("❌ Arquivo do modelo NÃO encontrado!")
    sys.exit(1)

# 3. Tentar carregar o modelo DIRETAMENTE
print("\n3️⃣ TENTANDO CARREGAR MODELO DIRETAMENTE:")
print("-" * 80)

from tensorflow.keras.models import load_model

print(f"⏳ Carregando: {model_found}")
print("   (isso pode demorar alguns segundos...)\n")

try:
    # Tentar carregar normalmente
    model = load_model(model_found)
    print("✅ MODELO CARREGADO COM SUCESSO!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
except Exception as e:
    print(f"❌ ERRO ao carregar modelo:")
    print(f"   Tipo: {type(e).__name__}")
    print(f"   Mensagem: {str(e)}\n")
    
    # Tentar com compile=False
    print("⏳ Tentando com compile=False...")
    try:
        model = load_model(model_found, compile=False)
        print("✅ MODELO CARREGADO (sem compilação)!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Recompilar
        print("\n⏳ Recompilando modelo...")
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Modelo recompilado!")
        
    except Exception as e2:
        print(f"❌ ERRO mesmo com compile=False:")
        print(f"   Tipo: {type(e2).__name__}")
        print(f"   Mensagem: {str(e2)}\n")
        
        # Mostrar traceback completo
        import traceback
        print("\n📋 TRACEBACK COMPLETO:")
        print("-" * 80)
        traceback.print_exc()
        print("-" * 80)

# 4. Tentar importar xray_classifier
print("\n4️⃣ TENTANDO IMPORTAR xray_classifier:")
print("-" * 80)

try:
    from xray_classifier import get_classifier
    print("✅ Módulo importado")
    
    print("\n⏳ Obtendo instância do classificador...")
    classifier = get_classifier()
    
    if classifier.is_model_loaded():
        print("✅ CLASSIFICADOR CARREGOU O MODELO COM SUCESSO!")
        info = classifier.get_model_info()
        print(f"\n📊 Informações:")
        import json
        print(json.dumps(info, indent=2))
    else:
        print("❌ CLASSIFICADOR NÃO CARREGOU O MODELO")
        info = classifier.get_model_info()
        print(f"\n📋 Informações de erro:")
        import json
        print(json.dumps(info, indent=2))
        
except Exception as e:
    print(f"❌ ERRO ao importar/usar xray_classifier:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ Diagnóstico concluído!")
print("=" * 80)
