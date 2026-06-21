#!/usr/bin/env python3
"""Phase 1: Tighten the feedback loop — diagnose tokenizer issue."""
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from transformers import AutoProcessor

MODEL_PATH = "outputs/stage3a_sft_box"

print("=== Tokenizer Special Token Diagnostic ===")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

# 1. Check if special tokens are registered
special_tokens = ["<|box|>", "<|/box|>", "<|point|>", "<|/point|>", "<|ref|>", "<|/ref|>"]
print(f"\nVocab size: {len(tokenizer)}")
print(f"Added tokens: {tokenizer.added_tokens_encoder}")

for tok in special_tokens:
    tid = tokenizer.convert_tokens_to_ids(tok)
    decoded = tokenizer.decode([tid])
    encoded = tokenizer.encode(tok, add_special_tokens=False)
    print(f"  '{tok}' -> id={tid}, decode='{decoded}', encoded_as={encoded}")

# 2. Check how a reasoning string is tokenized
reasoning = """Intent Analysis: I need to count all persons in the image.
Grounding: <|ref|>persons<|/ref|><|box|>[[352,190,622,825],[246,0,523,784]]<|/box|>
Summarization: There are 2 person(s) in total."""

print(f"\n=== Tokenizing a reasoning string ===")
tokens = tokenizer.encode(reasoning, add_special_tokens=False)
print(f"Total tokens: {len(tokens)}")

# Show tokens around special tokens
for tok in special_tokens:
    tid = tokenizer.convert_tokens_to_ids(tok)
    # Find occurrences
    positions = [i for i, t in enumerate(tokens) if t == tid]
    print(f"  '{tok}' (id={tid}): found at positions {positions}")
    if positions:
        for p in positions[:3]:
            ctx = tokens[max(0,p-2):p+3]
            decoded_ctx = [tokenizer.decode([t]) for t in ctx]
            print(f"    pos {p}: context={ctx} -> {decoded_ctx}")

# 3. Check: what happens if special tokens are NOT in vocab?
# Try decoding a token that produces Thai characters
print(f"\n=== Decoding suspicious tokens ===")
# The garbled output contains Thai chars like ส่งเสริม, ลักษณ์, ผู้ใช้, เท่านั้น
# Let's find what tokens these are
thai_texts = ["ส่งเสริม", "ลักษณ์", "ผู้ใช้", "เท่านั้น", "ทรัพ", "จะได้รับ"]
for text in thai_texts:
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"  '{text}' -> ids={ids}")

# 4. Check: what tokens are near the special token IDs?
print(f"\n=== Checking token neighborhood around special token IDs ===")
for tok in special_tokens:
    tid = tokenizer.convert_tokens_to_ids(tok)
    for offset in [-2, -1, 1, 2]:
        try:
            neighbor = tokenizer.decode([tid + offset])
            print(f"  id {tid + offset} (offset {offset:+d} from '{tok}'): '{neighbor}'")
        except:
            print(f"  id {tid + offset}: <decode error>")

# 5. Check the actual token IDs in the base model vs adapter
print(f"\n=== Base model tokenizer check ===")
base_processor = AutoProcessor.from_pretrained("models/Qwen3-VL-4B-Thinking", trust_remote_code=True)
base_tokenizer = base_processor.tokenizer
print(f"Base vocab size: {len(base_tokenizer)}")
for tok in special_tokens:
    base_tid = base_tokenizer.convert_tokens_to_ids(tok)
    base_decoded = base_tokenizer.decode([base_tid])
    print(f"  '{tok}' -> base_id={base_tid}, decode='{base_decoded}'")