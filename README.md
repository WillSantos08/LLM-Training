# 🌙 Cephalon Luna

> Assistente pessoal construída do zero com PyTorch.
> Aprende a partir de arquivos Markdown e evolui a cada ciclo de treinamento.

---

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura](#arquitetura)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Como Ensinar a Luna](#como-ensinar-a-luna)
- [Configuração](#configuração)
- [Ciclo de Treinamento](#ciclo-de-treinamento)
- [Avaliação](#avaliação)
- [Dependências](#dependências)
- [Licença](#licença)

---

## Sobre o Projeto

Cephalon Luna é uma assistente pessoal baseada em um modelo de linguagem **construído do zero** utilizando PyTorch puro.

Diferente de projetos que utilizam modelos pré-treinados como base, o Cephalon Luna implementa toda a arquitetura Transformer manualmente, desde o mecanismo de atenção até o tokenizer BPE, permitindo total controle sobre o aprendizado.

O modelo aprende exclusivamente a partir de arquivos Markdown escritos pelo próprio usuário, seguindo um ciclo contínuo de:

```
Escreva → Treine → Teste → Avalie → Melhore → Repita
```

### Destaques

- Transformer GPT-style implementado do zero com PyTorch
- Tokenizer próprio baseado em vocabulário de subpalavras
- Aprende com arquivos Markdown simples
- Ciclo contínuo de treinamento com checkpoints automáticos
- Avaliação visual com gráficos gerados pelo matplotlib
- Configuração centralizada via `config.yaml`
- Suporte a CPU, GPU (CUDA) e Apple Silicon (MPS)

---

## Arquitetura

O Cephalon Luna é um modelo de linguagem causal (GPT-style) composto pelos seguintes componentes:

### Modelo

```
Token IDs
    ↓
Token Embedding + Positional Embedding
    ↓
Dropout
    ↓
[ LayerNorm → CausalAttention → Residual ] × N camadas
[ LayerNorm → FeedForward     → Residual ]
    ↓
LayerNorm Final
    ↓
LM Head (Linear → vocab_size)
    ↓
Logits → próximo token
```

### Componentes Internos

| Componente | Descrição |
|---|---|
| `CausalAttention` | Multi-head self-attention com máscara causal via `scaled_dot_product_attention` |
| `FeedForward` | MLP com ativação GELU e dropout |
| `TransformerBlock` | Bloco completo com pre-norm e conexões residuais |
| `LunaModel` | Modelo completo com weight tying entre embedding e lm_head |

### Tokenizer

O tokenizer é baseado em vocabulário de subpalavras com tokens especiais dedicados ao formato de conversa:

| Token | ID | Uso |
|---|---|---|
| `<pad>` | 0 | Padding de sequências |
| `<unk>` | 1 | Token desconhecido |
| `<bos>` | 2 | Início de sequência |
| `<eos>` | 3 | Fim de sequência |
| `<usr>` | 4 | Início da fala do usuário |
| `<sep>` | 5 | Separador pergunta / resposta |

### Formato de Treinamento

Cada par do Markdown é convertido para o seguinte formato antes de ser tokenizado:

```
<usr>pergunta<sep>resposta<eos>
```

---

## Estrutura do Projeto

```
.
├── .gitignore
├── README.md
├── config.yaml                        ← configuração central
├── requirements.txt                   ← dependências
│
└── cephalon_luna/
    │
    ├── train.py                       ← executa o treinamento
    ├── test.py                        ← interface de chat
    ├── evaluate.py                    ← gráficos e métricas
    │
    ├── config/                        ← configurações tipadas
    │   ├── __init__.py
    │   ├── settings.py                ← lê e valida o config.yaml
    │   ├── model_config.py            ← dataclass da arquitetura
    │   └── train_config.py            ← dataclasses de treino
    │
    ├── core/                          ← núcleo do modelo
    │   ├── __init__.py
    │   ├── tokenizer.py               ← tokenizer de subpalavras
    │   ├── model.py                   ← Transformer completo
    │   ├── dataset.py                 ← Dataset e DataLoaders
    │   └── trainer.py                 ← loop treino → avaliação → checkpoint
    │
    ├── parser/                        ← leitura dos markdowns
    │   ├── __init__.py
    │   └── markdown_parser.py         ← extrai pares pergunta/resposta
    │
    ├── interface/                     ← interação com o usuário
    │   ├── __init__.py
    │   └── repl.py                    ← chat interativo no terminal
    │
    ├── data/
    │   ├── raw/                       ← seus arquivos .md (você escreve aqui)
    │   ├── processed/                 ← corpus.txt (gerado automaticamente)
    │   └── tokenizer/                 ← tokenizer.json (gerado automaticamente)
    │
    └── models/
        ├── checkpoints/               ← epoch_005.pt, epoch_010.pt ...
        ├── latest/                    ← model.pt (melhor modelo salvo)
        └── logs/                      ← history.json, evaluation.png
```

### O que é gerado automaticamente

```
data/processed/corpus.txt             ← corpus extraído dos .md
data/tokenizer/tokenizer.json         ← vocabulário treinado
models/latest/model.pt                ← melhor modelo salvo
models/checkpoints/epoch_NNN.pt       ← checkpoints periódicos
models/logs/history.json              ← histórico de métricas
models/logs/evaluation.png            ← gráficos (com --save)
```

### O que você escreve

```
config.yaml                           ← ajuste do modelo
cephalon_luna/data/raw/*.md           ← conhecimento da Luna
```

---

## Requisitos

- Python 3.10 ou superior
- pip

### Hardware Suportado

| Hardware | Suporte |
|---|---|
| CPU | ✅ Qualquer máquina |
| GPU NVIDIA (CUDA) | ✅ Recomendado para modelos maiores |
| Apple Silicon (MPS) | ✅ Mac M1, M2, M3, M4 |

---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/WillSantos08/cephalon-luna.git
cd cephalon-luna
```

### 2. Criar o ambiente virtual

```bash
python -m venv .venv
```

### 3. Ativar o ambiente

```bash
# Linux / Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Instalar dependências

```bash
pip install -r requirements.txt
```

### 5. Verificar instalação

```bash
python -c "
import torch
import yaml
import matplotlib

print('torch      :', torch.__version__)
print('yaml       :', yaml.__version__)
print('matplotlib :', matplotlib.__version__)
print('CUDA       :', torch.cuda.is_available())
print('MPS        :', torch.backends.mps.is_available())
"
```

---

## Como Usar

### Treinar o modelo

```bash
# Treinar do zero
python cephalon_luna/train.py

# Continuar de onde parou
python cephalon_luna/train.py --resume
```

### Testar e conversar

```bash
python cephalon_luna/test.py
```

### Avaliar a evolução

```bash
# Exibir gráficos na tela
python cephalon_luna/evaluate.py

# Salvar gráficos em PNG
python cephalon_luna/evaluate.py --save
```

### Comandos disponíveis no chat

| Comando | Descrição |
|---|---|
| `sair` | Encerra a sessão |
| `/temp 0.7` | Ajusta a temperature em tempo real |
| `/topk 40` | Ajusta o top-k em tempo real |

---

## Como Ensinar a Luna

Todo o conhecimento da Luna vem dos arquivos `.md` em `cephalon_luna/data/raw/`.

### Formato do arquivo

```markdown
---
categoria: nome_da_categoria
prioridade: alta
---

# Título do Arquivo

## Seção

Pergunta: sua pergunta aqui
Resposta: a resposta que você quer que Luna aprenda
```

### Regras

- Toda `Resposta:` precisa vir logo após uma `Pergunta:`
- Pode ter quantos pares quiser por arquivo
- Pode ter quantos arquivos `.md` quiser
- O frontmatter `---` é opcional mas recomendado para organização

### Exemplo

```markdown
---
categoria: programacao
prioridade: alta
---

# Python

Pergunta: O que é Python?
Resposta: Python é uma linguagem de programação de alto nível, Operador. É conhecida pela sintaxe limpa e é amplamente usada em inteligência artificial, automação e desenvolvimento web.

Pergunta: O que é uma lista em Python?
Resposta: Uma lista em Python é uma estrutura de dados ordenada e mutável, Operador. Você pode criar uma com colchetes, por exemplo: minha_lista = [1, 2, 3].
```

### Após adicionar ou editar arquivos

```bash
python cephalon_luna/train.py --resume
```

### Data Augmentation

O sistema gera automaticamente variações dos pares de treino quando `data.augment: true` no `config.yaml`. Isso aumenta a diversidade dos dados sem que você precise escrever mais exemplos.

---

## Configuração

Todos os ajustes são feitos no `config.yaml` na raiz do projeto.

### Caminhos

```yaml
paths:
  raw_data:        "data/raw"
  processed_data:  "data/processed"
  tokenizer:       "data/tokenizer/tokenizer.json"
  latest_model:    "models/latest/model.pth"
  checkpoints:     "models/checkpoints"
  logs:            "models/logs"
```

### Modelo

| Parâmetro | Descrição | Padrão |
|---|---|---|
| `vocab_size` | Tamanho do vocabulário | `2000` |
| `context_len` | Janela de contexto em tokens | `256` |
| `d_model` | Dimensão interna do modelo | `256` |
| `num_heads` | Cabeças de atenção | `8` |
| `num_layers` | Blocos Transformer | `6` |
| `d_ff` | Dimensão da camada feed-forward | `1024` |
| `dropout` | Taxa de regularização | `0.1` |

### Guia de tamanho do modelo

| Situação | d_model | num_heads | num_layers | d_ff |
|---|---|---|---|---|
| Poucos dados (< 500 pares) | `128` | `4` | `4` | `512` |
| Médio (500 a 2000 pares) | `256` | `8` | `6` | `1024` |
| Muitos dados (> 2000 pares) | `512` | `8` | `8` | `2048` |

### Treinamento

| Parâmetro | Descrição | Padrão |
|---|---|---|
| `epochs` | Voltas completas sobre os dados | `20` |
| `batch_size` | Sequências por passo | `16` |
| `lr` | Learning rate | `3e-4` |
| `grad_clip` | Gradient clipping | `1.0` |
| `resume` | Continuar do último checkpoint | `false` |

### Scheduler de Learning Rate

| Tipo | Comportamento |
|---|---|
| `cosine` | Decaimento suave em curva cossenoide (recomendado) |
| `linear` | Decaimento linear simples |
| `none` | Learning rate fixo durante todo o treino |

### Geração

| Parâmetro | Descrição | Padrão |
|---|---|---|
| `temperature` | Criatividade da resposta | `0.7` |
| `top_k` | Tokens candidatos por passo | `40` |
| `max_new_tokens` | Tamanho máximo da resposta | `120` |

### Guia de temperature

| Faixa | Comportamento |
|---|---|
| `0.1 → 0.4` | Respostas certeiras e repetitivas |
| `0.5 → 0.7` | Equilíbrio entre foco e variedade (recomendado) |
| `0.8 → 1.2` | Respostas mais criativas e variadas |
| `> 1.2` | Respostas caóticas |

### Checkpoints

| Parâmetro | Descrição | Padrão |
|---|---|---|
| `save_every_n` | Salvar checkpoint a cada N epochs | `5` |
| `keep_last_n` | Manter apenas os N checkpoints mais recentes | `3` |

### Hardware

| Parâmetro | Opções | Padrão |
|---|---|---|
| `device` | `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `seed` | Semente para reprodutibilidade | `42` |

---

## Ciclo de Treinamento

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   1. Escreva pares no .md                               │
│              ↓                                          │
│   2. python cephalon_luna/train.py                      │
│              ↓                                          │
│      ┌─ TREINO ──────────────────────┐                  │
│      │  forward → backward → step    │                  │
│      └───────────────────────────────┘                  │
│              ↓                                          │
│      ┌─ AVALIAÇÃO ───────────────────┐                  │
│      │  val loss + perplexidade      │                  │
│      └───────────────────────────────┘                  │
│              ↓                                          │
│      ┌─ CHECKPOINT ──────────────────┐                  │
│      │  salva se melhorou            │                  │
│      └───────────────────────────────┘                  │
│              ↓                                          │
│      ┌─ AMOSTRAS ────────────────────┐                  │
│      │  gera respostas ao vivo       │                  │
│      └───────────────────────────────┘                  │
│              ↓                                          │
│   3. python cephalon_luna/test.py                       │
│              ↓                                          │
│   4. Identificou respostas ruins?                       │
│              ↓                                          │
│   5. Corrija ou adicione exemplos no .md                │
│              ↓                                          │
│      Volte ao passo 2                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Interpretando o output do treino

```
  ┌─ EPOCH 1/20 ─ TREINO ─────────────
    Epoch 1 │  20.0% │ loss=4.1200 │ 2.1s
    Epoch 1 │ 100.0% │ loss=3.0100 │ 10.6s
  │  Train Loss : 3.4060 │ PPL : 30.17
  ├─ EPOCH 1/20 ─ AVALIAÇÃO ──────────
  │  Val Loss   : 3.2100 │ PPL : 24.78 ⭐ Melhor!
  │  💾 Melhor modelo → models/latest/model.pt
  ├─ EPOCH 1/20 ─ AMOSTRAS ─────────
  │  💬 Quem é você?
  │  🤖 Sou Cephalon Luna...
```

| Sinal | Significado |
|---|---|
| `loss` caindo | Modelo aprendendo ✅ |
| `PPL` caindo | Modelo melhorando ✅ |
| `⭐ Melhor!` | Novo melhor modelo salvo ✅ |
| `loss` estável por muitas epochs | Aumentar `lr` ⚠️ |
| `val loss` subindo enquanto `train loss` cai | Overfitting, reduza `epochs` ou aumente `dropout` ❌ |

---

## Avaliação

O `evaluate.py` gera os seguintes gráficos a partir do `history.json`:

| Gráfico | Descrição |
|---|---|
| Loss por Epoch | Train loss vs Val loss ao longo do treino |
| Perplexidade | Evolução da perplexidade no conjunto de validação |
| Δ Val Loss | Variação da val loss entre epochs (verde = melhorou) |
| Learning Rate | Curva do scheduler ao longo do treino |
| Tabela Resumo | Métricas finais consolidadas |

```bash
# Visualizar na tela
python cephalon_luna/evaluate.py

# Salvar em PNG em models/logs/evaluation.png
python cephalon_luna/evaluate.py --save
```

---

## Dependências

| Lib | Versão | Uso |
|---|---|---|
| `torch` | latest | Motor de deep learning e Transformer |
| `numpy` | latest | Operações numéricas |
| `pyyaml` | latest | Leitura e validação do config.yaml |
| `matplotlib` | latest | Geração de gráficos de avaliação |

```bash
pip install -r requirements.txt
```

---

## Instalação com GPU

### CUDA 11.8

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### CUDA 12.1

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Apple Silicon

```bash
pip install -r requirements.txt
# suporte MPS já incluso no pacote padrão do PyTorch
```

---

## Licença

MIT
