#!/usr/bin/env python3
"""
Diagnóstico Avançado - Modelo de Raio-X
Verifica TODOS os possíveis problemas
"""

import os
import sys

print("=" * 80)
print("🔍 DIAGNÓSTICO AVANÇADO DO MODELO DE RAIO-X")
print("=" * 80)

# 1. Informações do Sistema
print("\n1️⃣ INFORMAÇÕES DO SISTEMA")
print("-" * 80)
print(f"   Python version: {sys.version}")
print(f"   Sistema operacional: {sys.platform}")
print(f"   Diretório de trabalho: {os.getcwd()}")
print(f"   Caminho do script: {os.path.abspath(__file__)}")

# 2. Verificar se xray_classifier.py existe
print("\n2️⃣ VERIFICANDO ARQUIVO xray_classifier.py")
print("-" * 80)

xray_file = "xray_classifier.py"
if os.path.exists(xray_file):
    print(f"   ✅ Arquivo encontrado: {xray_file}")
    
    # Ler linha 31 para ver o caminho do modelo
    try:
        with open(xray_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) >= 31:
                linha_31 = lines[30].strip()  # linha 31 é índice 30
                print(f"   📄 Linha 31: {linha_31}")
                
                # Extrair caminho
                if 'MODEL_PATH' in linha_31:
                    print(f"   ℹ️  Esta linha define o caminho do modelo")
    except Exception as e:
        print(f"   ⚠️  Erro ao ler arquivo: {e}")
else:
    print(f"   ❌ Arquivo NÃO encontrado: {xray_file}")
    print(f"   💡 Você está no diretório correto?")

# 3. Procurar pasta do modelo
print("\n3️⃣ PROCURANDO PASTA DO MODELO")
print("-" * 80)

base_dir = os.getcwd()
print(f"   Procurando em: {base_dir}")

# Listar todas as pastas
all_folders = [d for d in os.listdir(base_dir) if os.path.isdir(d)]
print(f"   Total de pastas encontradas: {len(all_folders)}")

# Procurar pastas que podem conter o modelo
model_folders = [d for d in all_folders if 'depart' in d.lower() or 'medico' in d.lower() or 'modelo' in d.lower()]

if model_folders:
    print(f"   ✅ Pastas relacionadas encontradas:")
    for folder in model_folders:
        print(f"      📁 {folder}")
        folder_path = os.path.join(base_dir, folder)
        try:
            files = os.listdir(folder_path)
            keras_files = [f for f in files if f.endswith('.keras') or f.endswith('.h5')]
            if keras_files:
                print(f"         ✅ Arquivos de modelo encontrados:")
                for kf in keras_files:
                    full_path = os.path.join(folder_path, kf)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    print(f"            • {kf} ({size_mb:.2f} MB)")
            else:
                print(f"         ⚠️  Nenhum arquivo .keras ou .h5 encontrado")
                print(f"         📂 Conteúdo: {files[:5]}...")  # Mostrar primeiros 5
        except Exception as e:
            print(f"         ❌ Erro ao listar: {e}")
else:
    print(f"   ❌ NENHUMA pasta relacionada ao modelo encontrada!")
    print(f"   💡 Pastas disponíveis:")
    for folder in all_folders[:10]:  # Mostrar primeiras 10
        print(f"      • {folder}")

# 4. Verificar TensorFlow
print("\n4️⃣ VERIFICANDO TENSORFLOW")
print("-" * 80)

try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow instalado")
    print(f"   📦 Versão: {tf.__version__}")
    
    # Verificar GPU (opcional)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   🎮 GPUs disponíveis: {len(gpus)}")
    else:
        print(f"   💻 Rodando em CPU")
        
except ImportError as e:
    print(f"   ❌ TensorFlow NÃO instalado")
    print(f"   ⚠️  Erro: {e}")
    print(f"   💡 Instale com: pip install tensorflow>=2.10.0")

# 5. Verificar OpenAI
print("\n5️⃣ VERIFICANDO OPENAI")
print("-" * 80)

try:
    import openai
    print(f"   ✅ Biblioteca OpenAI instalada")
    print(f"   📦 Versão: {openai.__version__}")
    
    # Verificar chave API
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"   ✅ OPENAI_API_KEY encontrada (.env)")
        print(f"   🔑 Começa com: {api_key[:10]}...")
    else:
        print(f"   ⚠️  OPENAI_API_KEY NÃO encontrada no .env")
        
except ImportError as e:
    print(f"   ❌ Biblioteca OpenAI NÃO instalada")
    print(f"   💡 Instale com: pip install openai")

# 6. Tentar importar xray_classifier
print("\n6️⃣ TENTANDO IMPORTAR xray_classifier")
print("-" * 80)

try:
    sys.path.insert(0, base_dir)
    from xray_classifier import get_classifier
    print(f"   ✅ Módulo importado com sucesso")
    
    # Tentar obter classificador
    print(f"   ⏳ Tentando obter instância do classificador...")
    classifier = get_classifier()
    
    # Verificar se modelo foi carregado
    if classifier.is_model_loaded():
        print(f"   ✅ MODELO CARREGADO COM SUCESSO!")
        info = classifier.get_model_info()
        print(f"   📊 Informações:")
        for key, value in info.items():
            print(f"      • {key}: {value}")
    else:
        print(f"   ❌ MODELO NÃO FOI CARREGADO")
        print(f"   ℹ️  Verifique os logs acima para detalhes")
        
        # Tentar obter info do modelo
        try:
            info = classifier.get_model_info()
            print(f"   📋 Informações de erro:")
            for key, value in info.items():
                print(f"      • {key}: {value}")
        except:
            pass
            
except ImportError as e:
    print(f"   ❌ Erro ao importar xray_classifier")
    print(f"   ⚠️  {e}")
except Exception as e:
    print(f"   ❌ Erro durante importação/carregamento")
    print(f"   ⚠️  {type(e).__name__}: {e}")

# 7. Verificar estrutura esperada
print("\n7️⃣ VERIFICANDO ESTRUTURA ESPERADA DO PROJETO")
print("-" * 80)

expected_files = [
    'chatbot.py',
    'xray_classifier.py',
    'gravar_e_transcrever.py',
    'create_db.py',
    'requirements.txt',
    '.env'
]

expected_folders = [
    'Departamento_Medico',  # ou variações
    'templates',
    'static',
    'data',
    'chromasaude'
]

print("   Arquivos esperados:")
for file in expected_files:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"      {status} {file}")

print("\n   Pastas esperadas:")
for folder in expected_folders:
    exists = os.path.exists(folder) and os.path.isdir(folder)
    status = "✅" if exists else "❌"
    print(f"      {status} {folder}")

# 8. Recomendações
print("\n" + "=" * 80)
print("📋 RECOMENDAÇÕES")
print("=" * 80)

# Análise do que falta
missing_model = not any('depart' in d.lower() for d in all_folders)
missing_tensorflow = False
missing_openai = False

try:
    import tensorflow
except:
    missing_tensorflow = True

try:
    import openai
except:
    missing_openai = True

if missing_model:
    print("\n🔴 PROBLEMA CRÍTICO: Pasta do modelo não encontrada!")
    print("   A pasta 'Departamento_Medico' (ou variação) não existe.")
    print("   ")
    print("   SOLUÇÕES:")
    print("   1. ✅ Certifique-se de que você está no diretório RAIZ do projeto")
    print("   2. ✅ A pasta deve estar no mesmo nível que chatbot.py")
    print("   3. ✅ Estrutura correta:")
    print("      ")
    print("      seu-projeto/")
    print("      ├── chatbot.py")
    print("      ├── xray_classifier.py")
    print("      └── Departamento_Medico/")
    print("          └── melhor_modelo.keras  (arquivo ~194MB)")
    print("   ")
    print("   4. ✅ Se a pasta está em outro lugar, copie-a para o diretório atual")

if missing_tensorflow:
    print("\n🟡 TensorFlow não instalado:")
    print("   pip install tensorflow>=2.10.0")

if missing_openai:
    print("\n🟡 OpenAI não instalado:")
    print("   pip install openai")

if not missing_model and not missing_tensorflow and not missing_openai:
    print("\n✅ Tudo parece estar instalado corretamente!")
    print("   Se ainda houver erro, verifique os logs detalhados acima.")

print("\n" + "=" * 80)
print("✅ Diagnóstico concluído!")
print("=" * 80)