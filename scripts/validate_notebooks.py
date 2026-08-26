#!/usr/bin/env python
"""Static stress-test for the experiment notebooks: parse every code cell (skipping Jupyter
magics), report syntax errors, cell counts, and flag well-formedness of any hard-coded HF model
ids (org/name). Run before shipping a notebook to the group.

    python scripts/validate_notebooks.py            # all notebooks/
    python scripts/validate_notebooks.py nb.ipynb   # one notebook
"""
import ast, json, os, re, sys, glob

MODEL_RE = re.compile(r'["\'](?:google|Qwen|meta-llama|mistral[\w-]*|llava-hf)/[A-Za-z0-9._-]+["\']')
# A real Jupyter magic: line-leading `!shell`, or `%`/`%%` immediately followed by a letter
# (`%pip`, `%%capture`). This deliberately does NOT match `%(...)` — a %-format continuation.
MAGIC_RE = re.compile(r"^\s*(!|%%?[A-Za-z])")

def _demagic(src):
    """Blank magic lines to `pass` (preserving indentation + line numbers) so ast can parse the rest."""
    out = []
    for l in src.splitlines():
        if MAGIC_RE.match(l):
            out.append(" " * (len(l) - len(l.lstrip())) + "pass")
        else:
            out.append(l)
    return "\n".join(out)

def check(path):
    nb = json.load(open(path, encoding="utf-8"))
    cells = nb.get("cells", [])
    code = [c for c in cells if c.get("cell_type") == "code"]
    bad, models = [], set()
    for i, c in enumerate(code):
        src = "".join(c.get("source", []))
        try:
            ast.parse(_demagic(src))
        except SyntaxError as e:
            bad.append((i, f"{e.msg} (line {e.lineno})"))
        models.update(m.strip("\"'") for m in MODEL_RE.findall(src))
    return len(cells), len(code), bad, sorted(models)

def main():
    args = sys.argv[1:]
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = args or sorted(glob.glob(os.path.join(repo, "notebooks", "*.ipynb")))
    total_bad = 0
    for p in paths:
        n_all, n_code, bad, models = check(p)
        tag = "OK  " if not bad else "FAIL"
        print(f"[{tag}] {os.path.basename(p):38s} {n_all:2d} cells ({n_code} code)"
              + (f"  models: {', '.join(models)}" if models else ""))
        for i, msg in bad:
            print(f"        code cell {i}: {msg}")
        total_bad += len(bad)
    print(f"\n{len(paths)} notebook(s), {total_bad} bad cell(s).")
    return 1 if total_bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
