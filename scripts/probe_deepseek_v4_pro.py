"""
Standalone probe for DeepSeek V4-Pro.

Goal: figure out what special tokens the model actually emits, by sending
deliberately INCOMPLETE prompts (raw partial templates / prefilled assistant
turns) and printing the raw, undecoded continuation.

This file does not import any project code. It uses only the openai SDK and
an environment variable DEEPSEEK_API_KEY. It does not modify any state.

Usage:
    python scripts/probe_deepseek_v4_pro.py           # run all probes
    python scripts/probe_deepseek_v4_pro.py --probe 3 # only run probe #3
    python scripts/probe_deepseek_v4_pro.py --max_tokens 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package not installed; pip install openai")

# Lazy import of the canonical V4 encoder (downloads encoding_dsv4.py from the
# V4-Flash repo on first call). Used by Probe 3.
def _load_canonical_encoder():
    from huggingface_hub import hf_hub_download
    enc_py = hf_hub_download("deepseek-ai/DeepSeek-V4-Pro", "encoding/encoding_dsv4.py")
    enc_dir = os.path.dirname(enc_py)
    if enc_dir not in sys.path:
        sys.path.insert(0, enc_dir)
    import encoding_dsv4
    return encoding_dsv4


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    sys.exit("DEEPSEEK_API_KEY env var not set")

# Beta endpoint is required for assistant-prefill (prefix=True) to work.
# It also serves /v1/completions for the same model.
BASE_URL = "https://api.deepseek.com/beta"
MODEL = "deepseek-v4-pro"


# --------------------------------------------------------------------------
# Probe definitions — each one is a (name, kind, payload) triple.
#   kind="completions" -> POST /v1/completions with `prompt`
#   kind="chat_plain"  -> POST /v1/chat/completions (no prefill)
# --------------------------------------------------------------------------

# Harder question — 12-ball balance puzzle requires multi-step decision-tree
# reasoning, not a single inclusion-exclusion formula. The optimal answer (3)
# can only be derived after working through 3 carefully-chosen weighings.
QUESTION = (
    "You have 12 identical-looking balls. One of them is either heavier OR "
    "lighter than the others (you don't know which). Using a balance scale, "
    "what is the minimum number of weighings needed to identify the odd ball "
    "AND determine whether it is heavier or lighter? Justify the answer with "
    "a concrete weighing strategy."
)

# The exact REASONING_EFFORT_MAX preamble from encoding_dsv4.py — applied to
# the /v1/completions prompt to mirror what the server would prepend when
# reasoning_effort="max" + thinking_mode="thinking" are requested via chat.
REASONING_EFFORT_MAX_PREAMBLE = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose "
    "the problem to resolve the root cause, rigorously stress-testing your "
    "logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every "
    "intermediate step, considered alternative, and rejected hypothesis to "
    "ensure absolutely no assumption is left unchecked."
)

PROBES = [
    # 1) /v1/completions: V3-style template + REASONING_EFFORT_MAX preamble +
    #    <｜Assistant｜><think>. This mirrors what the canonical encoder
    #    produces for thinking_mode="thinking" + reasoning_effort="max".
    {
        "name": "1. completions endpoint, V3 template + max-effort preamble + <think>",
        "kind": "completions",
        "prompt": (
            f"<｜begin▁of▁sentence｜>{REASONING_EFFORT_MAX_PREAMBLE}\n\n"
            f"<｜User｜>{QUESTION}<｜Assistant｜><think>"
        ),
    },

    # 2) /v1/chat/completions: plain user message, server applies its canonical
    #    template. We pass reasoning_effort="max" (top-level) so the server
    #    injects the same preamble as Probe 1 — useful side-by-side comparison.
    {
        "name": "2. chat endpoint, plain user message + reasoning_effort='max'",
        "kind": "chat_plain",
        "messages": [
            {"role": "user", "content": QUESTION},
        ],
    },

    # 3) /v1/completions: same as Probe 1 but the prompt is built via the
    #    canonical encoder (encoding_dsv4.py from the V4-Flash repo) — i.e. by
    #    the SAME tokenizer/encoder DeepSeek's server uses. Verifies the hand-
    #    assembled prompt in Probe 1 is byte-identical to encoder output and
    #    rules out any subtle whitespace/encoding drift as a cause of differences.
    {
        "name": "3. completions endpoint, prompt rendered by canonical encoder + max effort",
        "kind": "completions_canonical",
        "messages": [
            {"role": "user", "content": QUESTION},
        ],
    },
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def show_choice_repr(choice, indent="    "):
    """Print everything we can extract about a choice — including special tokens."""
    msg_or_text = getattr(choice, "message", None) or getattr(choice, "text", None)
    finish = getattr(choice, "finish_reason", None)
    stop = getattr(choice, "stop_reason", None)

    print(f"{indent}finish_reason: {finish!r}   stop_reason: {stop!r}")

    if hasattr(choice, "text") and choice.text is not None:
        # /v1/completions response
        print(f"{indent}text (repr to expose specials):")
        print(f"{indent}  {choice.text!r}")
    if hasattr(choice, "message") and choice.message is not None:
        msg = choice.message
        rc = getattr(msg, "reasoning_content", None)
        ct = getattr(msg, "content", None)
        print(f"{indent}message.role: {getattr(msg, 'role', None)!r}")
        print(f"{indent}message.reasoning_content (repr):")
        print(f"{indent}  {rc!r}" if rc is not None else f"{indent}  <None>")
        print(f"{indent}message.content (repr):")
        print(f"{indent}  {ct!r}" if ct is not None else f"{indent}  <None>")


def run_probe(client: OpenAI, probe: dict, max_tokens: int):
    print("\n" + "=" * 80)
    print(probe["name"])
    print("=" * 80)
    kind = probe["kind"]
    print(f"endpoint kind: {kind}")
    # Force thinking mode on every probe so we test apples-to-apples.
    # /v1/completions: SDK does NOT take reasoning_effort as a top-level kwarg,
    # so we fold both flags into extra_body.
    # /v1/chat/completions: reasoning_effort is a top-level kwarg.
    import pdb; pdb.set_trace()
    if kind in ("completions", "completions_canonical"):
        if kind == "completions_canonical":
            # Build the prompt via the canonical V4 encoder so it's bit-for-bit
            # what DeepSeek's server-side template would produce.
            enc = _load_canonical_encoder()
            prompt = enc.encode_messages(
                probe["messages"],
                thinking_mode="thinking",
                reasoning_effort="max",
            )
            print(f"prompt source:  canonical encoder (encoding_dsv4.py)")
        else:
            prompt = probe["prompt"]
            print(f"prompt source:  hand-assembled in script")
        print(f"prompt (repr):\n  {prompt!r}")
        print(f"thinking-mode flags: extra_body={{'thinking': {{'type': 'enabled'}}, 'reasoning_effort': 'max'}}")
        print(f"--- response ---")
        try:
            resp = client.completions.create(
                model=MODEL,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                n=1,
                extra_body={
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "max",
                },
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            return
        for c in resp.choices:
            show_choice_repr(c)
    elif kind == "chat_plain":
        print(f"messages:\n  {json.dumps(probe['messages'], ensure_ascii=False, indent=2)}")
        print(f"thinking-mode flags: reasoning_effort='max', extra_body={{'thinking': {{'type': 'enabled'}}}}")
        print(f"--- response ---")
        body = dict(
            model=MODEL,
            messages=probe["messages"],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            n=1,
        )
        try:
            resp = client.chat.completions.create(
                **body,
                reasoning_effort="max",
                extra_body={"thinking": {"type": "enabled"}},
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            return
        for c in resp.choices:
            show_choice_repr(c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_tokens", type=int, default=512,
                   help="max tokens in continuation (default 512 — keep small to see patterns)")
    p.add_argument("--probe", type=int, default=None,
                   help="1-indexed probe to run alone (otherwise all probes run)")
    args = p.parse_args()

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"DeepSeek base_url: {BASE_URL}")
    print(f"model:             {MODEL}")
    print(f"max_tokens:        {args.max_tokens}")
    print(f"# probes:          {len(PROBES)}")

    probes = PROBES if args.probe is None else [PROBES[args.probe - 1]]
    for probe in probes:
        run_probe(client, probe, args.max_tokens)

    print("\n" + "=" * 80)
    print("Done. Inspect each block above for special tokens (e.g. <｜...｜>, <think>,")
    print("</think>, EOS markers) that appear in text/content/reasoning_content.")
    print("=" * 80)


if __name__ == "__main__":
    main()
