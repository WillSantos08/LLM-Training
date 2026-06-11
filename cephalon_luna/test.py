import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config.settings  import load_config
from core.model       import LunaModel
from core.tokenizer   import LunaTokenizer
from interface.repl   import LunaREPL


def main():
    print("""
╔══════════════════════════════════════════╗
     🌙  Cephalon Luna — Teste / Chat      
╚══════════════════════════════════════════╝
    """)

    cfg    = load_config("config.yaml")
    device = str(cfg.resolve_device())

    # ── Tokenizer ─────────────────────────────────────
    tok_path = cfg.paths.tokenizer
    if not os.path.exists(tok_path):
        print(f"  ❌ Tokenizer não encontrado : {tok_path}")
        print("  👉 Execute : python cephalon_luna/train.py")
        sys.exit(1)

    tokenizer = LunaTokenizer.load(tok_path)

    # ── Modelo ────────────────────────────────────────
    model_path = cfg.paths.latest_model
    if not os.path.exists(model_path):
        print(f"  ❌ Modelo não encontrado : {model_path}")
        print("  👉 Execute : python cephalon_luna/train.py")
        sys.exit(1)

    model = LunaModel.load(model_path, device=device)

    print(f"  🖥️  Device      : {device}")
    print(f"  🌡️  Temperature : {cfg.generation.temperature}")
    print(f"  🎯 Top-K       : {cfg.generation.top_k}")
    print(f"  📝 Max tokens  : {cfg.generation.max_new_tokens}")

    # ── Chat ──────────────────────────────────────────
    repl = LunaREPL(
        model       = model,
        tokenizer   = tokenizer,
        device      = device,
        temperature = cfg.generation.temperature,
        top_k       = cfg.generation.top_k,
        max_tokens  = cfg.generation.max_new_tokens,
    )

    repl.chat()


if __name__ == "__main__":
    main()