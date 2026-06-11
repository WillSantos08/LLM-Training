import json
import re
import os
from typing import List, Dict


class LunaTokenizer:
    """
    Tokenizer baseado em vocabulário de subpalavras.
    Simples o suficiente para entender, eficaz para o projeto.
    """

    SPECIAL = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<usr>": 4,
        "<sep>": 5,
    }

    def __init__(self):
        self.vocab:        Dict[str, int] = {}
        self.id_to_token:  Dict[int, str] = {}
        self.trained:      bool           = False

    def train(self, corpus: str, max_vocab: int = 2000):
        """Constrói vocabulário a partir do corpus."""
        print(f"  🔤 Treinando tokenizer...")

        self.vocab = dict(self.SPECIAL)

        tokens = self._extract_tokens(corpus)

        freq: Dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])

        for token, _ in sorted_tokens:
            if len(self.vocab) >= max_vocab:
                break
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Garantir que todos os caracteres estão no vocab
        for char in set(corpus):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.trained     = True

        print(f"  ✅ Vocabulário: {len(self.vocab)} tokens")
        return self

    def _extract_tokens(self, text: str) -> List[str]:
        """Extrai tokens do texto mantendo especiais intactos."""
        special_pattern = "|".join(re.escape(s) for s in self.SPECIAL.keys())
        parts           = re.split(f"({special_pattern})", text)
        tokens          = []

        for part in parts:
            if part in self.SPECIAL:
                tokens.append(part)
            else:
                words = re.findall(
                    r"[a-záàâãéèêíïóôõúüçA-ZÁÀÂÃÉÈÊÍÏÓÔÕÚÜÇ]+"
                    r"|[^a-záàâãéèêíïóôõúüçA-ZÁÀÂÃÉÈÊÍÏÓÔÕÚÜÇ\s]"
                    r"|\s+",
                    part
                )
                tokens.extend(words)

        return [t for t in tokens if t]

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """Texto → lista de IDs."""
        if not self.trained:
            raise RuntimeError("Tokenizer não treinado!")

        tokens = self._extract_tokens(text)
        ids    = []

        if add_special:
            ids.append(self.SPECIAL["<bos>"])

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                for char in token:
                    ids.append(
                        self.vocab.get(char, self.SPECIAL["<unk>"])
                    )

        if add_special:
            ids.append(self.SPECIAL["<eos>"])

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """IDs → texto."""
        special_ids = set(self.SPECIAL.values())
        parts       = []

        for id_ in ids:
            if skip_special and id_ in special_ids:
                continue
            parts.append(self.id_to_token.get(id_, "<unk>"))

        return "".join(parts)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, ensure_ascii=False)
        print(f"  💾 Tokenizer salvo : {path}")

    @classmethod
    def load(cls, path: str) -> "LunaTokenizer":
        t = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        t.vocab        = data["vocab"]
        t.id_to_token  = {int(v): k for k, v in t.vocab.items()}
        t.trained      = True
        print(f"  ✅ Tokenizer carregado : {len(t.vocab)} tokens")
        return t

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.SPECIAL["<pad>"]

    @property
    def eos_id(self) -> int:
        return self.SPECIAL["<eos>"]

    @property
    def sep_id(self) -> int:
        return self.SPECIAL["<sep>"]