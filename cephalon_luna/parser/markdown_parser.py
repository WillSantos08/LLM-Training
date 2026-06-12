import os
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class KnowledgeEntry:
    """Um par pergunta-resposta extraído do markdown."""
    pergunta:  str
    resposta:  str
    categoria: str = "geral"
    arquivo:   str = ""

    def to_training_text(self) -> str:
        """
        Converte para texto formatado para treino.
        Formato: <usr>pergunta<sep>resposta<eos>
        """
        return (
            f"<usr>{self.pergunta.strip()}"
            f"<sep>{self.resposta.strip()}"
            f"<eos>"
        )


@dataclass
class KnowledgeBase:
    """Coleção de entradas de conhecimento."""
    entries: List[KnowledgeEntry] = field(default_factory=list)

    def add(self, entry: KnowledgeEntry):
        self.entries.append(entry)

    def to_corpus(self) -> str:
        """Converte tudo para um corpus de treino."""
        return "\n".join(e.to_training_text() for e in self.entries)

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"KnowledgeBase({len(self.entries)} entradas)"


class MarkdownParser:
    """
    Lê arquivos .md e extrai pares de treino.
    """

    def __init__(self, data_dir: str = "cephalon_luna/data/raw"):
        self.data_dir = data_dir

    def parse_all(self, verbose: bool = True) -> KnowledgeBase:
        """
        Lê todos os .md do diretório e retorna KnowledgeBase.
        """
        kb = KnowledgeBase()

        if not os.path.exists(self.data_dir):
            print(f"  ⚠️  Diretório não encontrado: {self.data_dir}")
            return kb

        md_files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.endswith(".md")
        ])

        if not md_files:
            print(f"  ⚠️  Nenhum .md encontrado em {self.data_dir}")
            return kb

        if verbose:
            print(f"  📂 Encontrados {len(md_files)} arquivo(s):")

        for filename in md_files:
            path    = os.path.join(self.data_dir, filename)
            entries = self._parse_file(path)
            for e in entries:
                kb.add(e)

            if verbose:
                print(f"     ✅ {filename} → {len(entries)} pares")

        if verbose:
            print(f"\n  📊 Total: {len(kb)} pares de treino")

        return kb

    def _parse_file(self, path: str) -> List[KnowledgeEntry]:
        """Extrai entradas de um arquivo .md."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        filename  = os.path.basename(path)
        categoria = self._extract_categoria(content)
        entries   = self._extract_pairs(content, categoria, filename)

        return entries

    def _extract_categoria(self, content: str) -> str:
        """Extrai categoria do frontmatter YAML."""
        match = re.search(r"categoria:\s*(.+)", content)
        if match:
            return match.group(1).strip()
        return "geral"

    def _extract_pairs(
        self,
        content:   str,
        categoria: str,
        filename:  str,
    ) -> List[KnowledgeEntry]:
        """Extrai todos os pares Pergunta/Resposta do conteúdo."""
        entries          = []
        lines            = content.split("\n")
        current_pergunta = None
        i                = 0

        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("Pergunta:"):
                current_pergunta = line[len("Pergunta:"):].strip()

            elif line.startswith("Resposta:") and current_pergunta:
                resposta = line[len("Resposta:"):].strip()

                # Resposta pode continuar nas próximas linhas
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if (
                        next_line
                        and not next_line.startswith("Pergunta:")
                        and not next_line.startswith("#")
                        and not next_line.startswith("---")
                    ):
                        resposta += " " + next_line
                        j += 1
                    else:
                        break

                entries.append(KnowledgeEntry(
                    pergunta  = current_pergunta,
                    resposta  = resposta,
                    categoria = categoria,
                    arquivo   = filename,
                ))
                current_pergunta = None

            i += 1

        return entries

    def augment(self, kb: KnowledgeBase) -> KnowledgeBase:
        """
        Data augmentation: gera variações das perguntas.
        """
        augmented = KnowledgeBase()
        prefixos  = ["Me diga, ", "Pode me explicar, "]

        for entry in kb.entries:
            augmented.add(entry)

            if len(entry.pergunta) < 50:
                for prefixo in prefixos:
                    nova = KnowledgeEntry(
                        pergunta  = prefixo + entry.pergunta[0].lower() + entry.pergunta[1:],
                        resposta  = entry.resposta,
                        categoria = entry.categoria,
                        arquivo   = entry.arquivo,
                    )
                    augmented.add(nova)

        return augmented