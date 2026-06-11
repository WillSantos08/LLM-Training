import torch

class LunaREPL:
    """Read-Eval-Print Loop para conversar com a Luna."""

    def __init__(
        self,
        model,
        tokenizer,
        device:      str   = "cpu",
        temperature: float = 0.7,
        top_k:       int   = 40,
        max_tokens:  int   = 120,
    ):
        self.model       = model.to(device)
        self.tokenizer   = tokenizer
        self.device      = device
        self.temperature = temperature
        self.top_k       = top_k
        self.max_tokens  = max_tokens

    def ask(self, pergunta: str) -> str:
        """Envia uma pergunta e retorna a resposta gerada."""
        prompt = f"<usr>{pergunta.strip()}<sep>"
        ids    = self.tokenizer.encode(prompt)
        ids_t  = torch.tensor([ids], device=self.device)

        out = self.model.generate(
            ids_t,
            max_new_tokens = self.max_tokens,
            temperature    = self.temperature,
            top_k          = self.top_k,
            eos_id         = self.tokenizer.eos_id,
        )

        new_ids  = out[0, len(ids):].tolist()
        response = self.tokenizer.decode(new_ids, skip_special=True)
        return response.strip()

    def chat(self):
        """Loop interativo no terminal."""
        print("\n" + "═" * 50)
        print("  🌙 Cephalon Luna — Pronta para conversar")
        print("  Digite 'sair' para encerrar")
        print("  Digite '/temp X' para ajustar temperature")
        print("  Digite '/topk X' para ajustar top-k")
        print("═" * 50 + "\n")

        while True:
            try:
                user_input = input("  👤 Operador > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  🌙 Luna > Até logo, Operador!")
                break

            if not user_input:
                continue

            if user_input.lower() == "sair":
                print("  🌙 Luna > Até logo, Operador!")
                break

            # Ajuste de temperatura
            if user_input.startswith("/temp"):
                try:
                    self.temperature = float(user_input.split()[1])
                    print(f"  ⚙️  Temperature → {self.temperature}")
                except (IndexError, ValueError):
                    print("  ⚠️  Use: /temp 0.7")
                continue

            # Ajuste de top-k
            if user_input.startswith("/topk"):
                try:
                    self.top_k = int(user_input.split()[1])
                    print(f"  ⚙️  Top-K → {self.top_k}")
                except (IndexError, ValueError):
                    print("  ⚠️  Use: /topk 40")
                continue

            response = self.ask(user_input)
            print(f"  🌙 Luna > {response}\n")