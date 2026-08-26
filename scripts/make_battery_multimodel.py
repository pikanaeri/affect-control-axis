#!/usr/bin/env python
"""Make halli75/Algoverse's behavioral battery model-swappable — word-for-word except the model lock.

Every runner already resolves the model as `mid = os.environ.get("E2E_MODEL", PRIMARY)`, then hard-blocks
anything but Gemma-4 with a 2-line guard:  `if mid != PRIMARY:  die/raise("fallback ... forbidden")`.
This script neutralizes that guard (turns it into `if False:`) and un-pins the `env['E2E_MODEL'] = '...gemma...'`
assignments in the hub launchers so an outer E2E_MODEL wins. NOTHING else is touched — the experiment
science is byte-identical. After patching, select the model at runtime:

    E2E_MODEL="Qwen/Qwen2.5-VL-7B-Instruct" python scripts/battery_exp03_run.py     # or any HF VLM id

Usage:  python make_battery_multimodel.py /path/to/halli75_algoverse    (defaults to ./halli75_algoverse)
The nf4 loader path (AutoModelForImageTextToText + BitsAndBytesConfig) loads Qwen2.5-VL / Pixtral /
Llama-3.2-Vision as-is. Runtime deps unchanged: EMOTIC data (E2E_EMOTIC/E2E_SPLIT), a writable lock dir
(E2E_LOCK), and XAI_API_KEY only for the Grok caption-rewrite control in exp03/exp04.
"""
import os, re, sys

GEMMA = "google/gemma-4-E4B-it"

def patch_file(path):
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines(keepends=True)
    changed = 0
    # 1) neutralize the model-lock guard: a line whose strip == "if mid != PRIMARY:"
    for i, ln in enumerate(lines):
        if ln.strip() == "if mid != PRIMARY:":
            indent = ln[: len(ln) - len(ln.lstrip())]
            lines[i] = f"{indent}if False:  # model-lock removed (multimodel): E2E_MODEL selects the model\n"
            changed += 1
    out = "".join(lines)
    # 2) un-pin hard E2E_MODEL assignments in hub/boot launchers so an outer value wins
    def unpin(m):
        q = m.group("q")
        return f"env[{q}E2E_MODEL{q}] = os.environ.get({q}E2E_MODEL{q}, {q}{GEMMA}{q})"
    out, n2 = re.subn(
        r"env\[(?P<q>['\"])E2E_MODEL['\"]\]\s*=\s*['\"]" + re.escape(GEMMA) + r"['\"]",
        unpin, out,
    )
    changed += n2
    if changed and out != src:
        open(path, "w", encoding="utf-8").write(out)
    return changed

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "halli75_algoverse"
    sdir = os.path.join(root, "scripts")
    if not os.path.isdir(sdir):
        print("no scripts/ under", root); return 1
    total_files = total_edits = 0
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".py"):
            continue
        c = patch_file(os.path.join(sdir, fn))
        if c:
            total_files += 1; total_edits += c
            print(f"  patched {fn}: {c} edit(s)")
    print(f"done: {total_edits} edits across {total_files} files. "
          f"Run any experiment with E2E_MODEL set, e.g. Qwen/Qwen2.5-VL-7B-Instruct.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
