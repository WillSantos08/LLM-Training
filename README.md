# 🤖 Cephalon Luna

> A lightweight, personal AI assistant based on Transformer architecture

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com)

## 📋 Visão Geral

**Cephalon Luna** é um assistente de IA pessoal baseado em arquitetura Transformer, otimizado para rodar eficientemente em CPU com recursos limitados. Treine seu próprio modelo com seus documentos e converse com ele!

### ✨ Características Principais

- **🧠 Transformer Customizável**: Implementação própria otimizada para CPU
- **📚 Suporte a Múltiplos Formatos**: TXT, PDF, DOCX, JSON, CSV, MD
- **⚡ Processamento Automático**: Converta arquivos brutos em dados de treino
- **🎓 Fácil de Treinar**: Configure e treine em minutos
- **💬 Chat Interativo**: Interface simples de linha de comando
- **💾 Checkpoints**: Salve e retome treinamento a qualquer momento
- **🎛️ Configurações Flexíveis**: Modelos pequeno, médio e grande
- **🚀 CPU ou GPU**: Funciona em qualquer dispositivo

---

## 🚀 Início Rápido

### Requisitos

- **Python**: 3.12+
- **RAM**: Mínimo 4GB (recomendado 8GB)
- **Armazenamento**: 5GB
- **CPU**: 2+ cores
- **GPU** (opcional): Para treino mais rápido

### 1️⃣ Instalação

```bash
# Clone ou baixe o projeto
cd cephalon_luna

# Crie um ambiente virtual
python -m venv .venv

# Ative o ambiente
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Instale dependências
pip install -r requirements.txt
```

### 2️⃣ Prepare seus Dados

Opção A: Use dados de exemplo

```bash
cd cephalon_luna
python main.py
# Selecione opção 4 para criar dados de exemplo
```

Opção B: Use seus próprios arquivos

```bash
# Coloque seus arquivos em:
cephalon_luna/data/raw/seu_arquivo.txt
```

### 3️⃣ Treine o Modelo

```bash
cd cephalon_luna

# Treine com arquivo bruto
python train.py --raw data/raw/seu_arquivo.txt --epochs 5

# Ou com dados já processados
python train.py --data data/processed/seu_dado.json --epochs 5
```

### 4️⃣ Converse com seu Assistente

```bash
cd cephalon_luna

# Chat interativo
python test.py --model models/trained/final_model.pth

# Ou teste com um prompt específico
python test.py --model models/trained/final_model.pth --prompt "O que você sabe?"
```

## 📁 Estrutura do Projeto

```text
cephalon_luna/
├── config/                      # Configurações e lógica principal
│   ├── settings/               # Classes de configuração
│   │   ├── constants.py       # Constantes padrão
│   │   ├── base.py            # Configuração base
│   │   ├── small.py           # Config otimizada para CPU
│   │   └── large.py           # Config para GPU
│   ├── transformer/            # Arquitetura do modelo
│   │   ├── attention.py       # Multi-head attention
│   │   ├── feed_forward.py    # Feed-forward network
│   │   └── model.py           # Modelo principal
│   └── data/                   # Processamento de dados
│       ├── dataset.py         # Dataset class
│       ├── loader.py          # DataLoader creator
│       ├── processor.py       # Document processor
│       └── trainer.py         # Training logic
├── data/
│   ├── raw/                    # Documentos originais (seus arquivos)
│   ├── processed/              # Dados processados para treino
│   └── cache/                  # Cache temporário
├── models/
│   ├── checkpoints/            # Checkpoints durante treino
│   └── trained/                # Modelos finais treinados
├── train.py                    # Script de treinamento
├── test.py                     # Script de teste/chat
├── process_text.py             # Processador de arquivos
└── main.py                     # Menu interativo
```

## 💻 Uso Detalhado

### Treinamento

Com arquivo bruto (recomendado)

```bash
cd cephalon_luna

# Processa arquivo e treina automaticamente
python train.py --raw data/raw/documento.txt --config small --epochs 5
```

Com dados já processados

```bash
python train.py --data data/processed/dados.json --config small --epochs 5
```

Com parâmetros customizados

```bash
python train.py \
  --raw data/raw/documento.txt \
  --config small \
  --epochs 10 \
  --batch-size 8
```

### Teste e Chat

Modo interativo

```bash
python test.py --model models/trained/final_model.pth
```

```text
You: O que você aprendeu?
Assistant: [resposta do modelo]

You: Como funciona IA?
Assistant: [resposta do modelo]
```

Teste com prompt único

```bash
python test.py \
  --model models/trained/final_model.pth \
  --prompt "Explique machine learning"
```

Com temperatura customizada

```bash
python test.py \
  --model models/trained/final_model.pth \
  --temperature 0.5 \
  --max-tokens 100
```

Menu Interativo

```bash
python main.py
```

Opções:

- Treinar modelo
- Chat com modelo
- Processar documentos
- Criar dados de exemplo
- Sair

## 📊 Formatos de Dados Suportados

JSON (Recomendado)

```json
[
  {
    "question": "Sua pergunta?",
    "answer": "A resposta correspondente..."
  },
  {
    "question": "Outra pergunta?",
    "answer": "Outra resposta..."
  }
]
```

Arquivos Brutos

- TXT: Arquivos de texto simples
- PDF: Documentos PDF
- DOCX: Arquivos Word
- CSV: Dados estruturados
- MD: Arquivos Markdown

Todos são processados automaticamente para formato JSON!

## 🎛️ Configurações

**Small (Padrão - CPU)**

- Embedding Dimension: 256
- Layers: 3
- Attention Heads: 4
- Batch Size: 4
- Tempo por época: ~10 min (10 exemplos)
- Memória: ~500MB

**Large (GPU)**

- Embedding Dimension: 768
- Layers: 12
- Attention Heads: 12
- Batch Size: 32
- Tempo por época: ~2 min (1000 exemplos)
- Memória: ~2GB

### Customizar

Edite `config.yaml` na raiz do projeto:

```yaml
model:
  embedding_dim: 512
  num_layers: 6
  num_heads: 8

training:
  learning_rate: 3e-4
  batch_size: 8
  num_epochs: 10
```

## 🔧 Troubleshooting

**Out of Memory**

```bash
# Reduzir tamanho do batch
python train.py --batch-size 2

# Ou usar modelo menor
python train.py --config small
```

**Arquivo não encontrado**

```bash
# Certifique-se de que está no diretório correto
cd cephalon_luna

# E que o arquivo existe
ls data/raw/seu_arquivo.txt
```

**Modelo não carrega**

```bash
# Verifique o caminho
python test.py --model models/trained/final_model.pth

# Use a mesma configuração usada no treino
python test.py --model models/trained/final_model.pth --config small
```

**Treinamento muito lento**

```bash
# Use GPU se disponível - edite config.yaml:
device: "cuda"

# Ou reduza o tamanho do modelo
python train.py --config small
```

## 📈 Dicas para Melhor Performance

### Dados de Qualidade

- ✅ Use 100+ exemplos de treino
- ✅ Mantenha consistência nas respostas
- ✅ Diversifique as perguntas
- ✅ Evite dados duplicados

### Treino Eficiente

- ✅ Comece com --config small
- ✅ Treine por 5-10 épocas
- ✅ Use --batch-size 4 para CPU
- ✅ Monitore o progresso

### Melhorar Respostas

```bash
# Aumentar épocas
python train.py --raw dados.txt --epochs 20

# Usar modelo maior (requer GPU)
python train.py --raw dados.txt --config large

# Ajustar temperatura (mais criativo)
python test.py --model final_model.pth --temperature 0.8
```

## 🏗️ Arquitetura Técnica

Transformer Decoder-Only

```text
Input Tokens
    ↓
Embedding (vocab → embedding_dim)
    ↓
[Transformer Block × num_layers]
    ├─ Multi-Head Attention
    ├─ Layer Norm + Residual
    ├─ Feed-Forward
    └─ Layer Norm + Residual
    ↓
Output Projection (embedding_dim → vocab)
    ↓
Logits → Softmax → Próximo Token
```

Componentes Principais

- Token Embedding: Converte índices em vetores densos
- Positional Embedding: Adiciona informação de posição
- Multi-Head Attention: Mecanismo de atenção múltipla
- Feed-Forward: Rede posição-wise
- Layer Normalization: Estabilização
- Residual Connections: Fluxo de gradientes

Para mais detalhes, veja docs/ARCHITECTURE.md

## 🔗 Uso como Biblioteca Python

```python
from cephalon_luna import ConfigSmall, TransformerModel, Trainer, create_dataloader
import torch

# Setup
config = ConfigSmall()
model = TransformerModel(config)

# Load data
dataloader, tokenizer = create_dataloader(
    'data/processed/dados.json',
    tokenizer_name=config.tokenizer_name,
    batch_size=config.batch_size,
    max_length=config.max_sequence_length
)

# Train
trainer = Trainer(model, config)
trainer.train(dataloader, num_epochs=5)

# Save
trainer.save_model('final_model.pth')

# Generate
prompt_ids = tokenizer.encode("Olá", return_tensors='pt').to(config.device)
generated = model.generate(prompt_ids, max_new_tokens=50)
response = tokenizer.decode(generated[0], skip_special_tokens=True)
print(response)
```

Para mais exemplos, veja docs/API.md

## 📚 Documentação

- QUICK_START.md - Guia de início rápido
- ARCHITECTURE.md - Detalhes técnicos
- API.md - Referência de API Python

## 🤝 Contribuindo

Sugestões e melhorias são bem-vindas!

- Fork o projeto
- Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
- Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
- Push para a branch (`git push origin feature/AmazingFeature`)
- Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo LICENSE para detalhes.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...

## 🙏 Agradecimentos

- Baseado na arquitetura Transformer de "Attention is All You Need"
- Utiliza PyTorch para computação
- Tokenizador via HuggingFace Transformers
- Inspirado em projetos open-source de LLM
