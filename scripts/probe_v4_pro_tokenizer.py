"""
Probe DeepSeek V4-Pro chat encoding + tokenization.

Renders three message lists with the canonical V4 encoder
(encoding_dsv4.py from the V4-Pro repo) and shows BOTH the rendered
prompt string and the resulting token IDs from the V4-Pro tokenizer.

Cases:
    1) Just a user question.
    2) User question + assistant turn with reasoning_content + content.
    3) Multi-turn: user → assistant (with thinking) → user.
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import hf_hub_download
import transformers


def main():
    # 1) Load canonical V4 encoder from the V4-Pro repo.
    enc_py = hf_hub_download("deepseek-ai/DeepSeek-V4-Pro", "encoding/encoding_dsv4.py")
    enc_dir = os.path.dirname(enc_py)
    if enc_dir not in sys.path:
        sys.path.insert(0, enc_dir)
    from encoding_dsv4 import encode_messages

    # 2) Load V4-Pro tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V4-Pro", trust_remote_code=True,
    )

    cases = [
        (
            "Case 1: just a user question",
            [{"role": "user", "content": "hello"}],
        ),
        (
            "Case 2: user question + assistant turn with reasoning_content + content",
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant",
                 "reasoning_content": "thinking...",
                 "content": "Hello! I am DeepSeek."},
            ],
        ),
        (
            "Case 3: multi-turn user → assistant (with thinking) → user",
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant",
                 "reasoning_content": "thinking...",
                 "content": "Hello! I am DeepSeek."},
                {"role": "user", "content": "1+1=?"},
            ],
        ),
    ]

    specials = set(tokenizer.get_added_vocab().values())

    for label, messages in cases:
        print("\n" + "=" * 80)
        print(label)
        print("=" * 80)
        print("messages:")
        for m in messages:
            print(f"  {m}")

        # messages -> string (canonical V4 encoder, thinking mode)
        prompt = encode_messages(messages, thinking_mode="thinking")
        print(f"\nrendered prompt (len={len(prompt)} chars):")
        print(f"  {prompt!r}")

        # string -> tokens
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        print(f"\ntoken count: {len(ids)}")
        print(f"token IDs:   {ids}")

        # token-by-token decode (annotate special tokens)
        print(f"\ntoken-by-token decode (★ = special token):")
        for i, tid in enumerate(ids):
            piece = tokenizer.decode([tid])
            star = "★" if tid in specials else " "
            print(f"  [{i:3d}] {star} id={tid:6d}  piece={piece!r}")


if __name__ == "__main__":
    main()
