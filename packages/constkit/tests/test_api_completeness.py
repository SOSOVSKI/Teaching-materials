"""API completeness test: verify code ↔ API_REFERENCE.md are in sync.

Uses `ast` to extract public symbols from source files and `inspect`
to verify they are importable at runtime. Parses API_REFERENCE.md to
extract documented symbol names and asserts:
  1. Every public code symbol is documented.
  2. Every documented symbol exists and is importable.

This test runs as part of the normal pytest suite so API drift is
caught automatically.
"""

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

# Paths relative to the package root (src layout).
#   tests/ -> parent -> packages/constkit/
_PKG_ROOT = Path(__file__).parent.parent            # packages/constkit/
_SRC = _PKG_ROOT / "src" / "constkit"               # importable source modules
_API_REF = _PKG_ROOT / "docs" / "API_REFERENCE.md"  # API reference doc


# ---------------------------------------------------------------------------
# Helpers: extract symbols from source
# ---------------------------------------------------------------------------

def _code_symbols(pkg: Path) -> set[str]:
    """Return all public symbol names defined in constkit source files."""
    found: set[str] = set()
    for py in sorted(pkg.glob("*.py")):
        if py.name.startswith("_"):
            continue
        tree = ast.parse(py.read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                found.add(node.name)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not item.name.startswith("_"):
                            found.add(f"{node.name}.{item.name}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    found.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        found.add(target.id)
    return found


def _doc_symbols(ref: Path) -> set[str]:
    """Extract documented symbol names from API_REFERENCE.md headings."""
    text = ref.read_text()
    found: set[str] = set()
    heading_re = re.compile(r"^#{1,6}\s+`([^`]+)`", re.MULTILINE)
    for m in heading_re.finditer(text):
        raw = m.group(1).strip()
        if re.match(r"^constkit\.\w+(\s|$)", raw):
            continue
        raw = re.sub(r"^(class|def)\s+", "", raw)
        raw = re.sub(r"^constkit\.\w+\.", "", raw)
        name = raw.split("(")[0].strip()
        name = name.split("->")[0].strip()
        parts = name.split(".")
        if any(p.startswith("_") for p in parts):
            continue
        had_signature = "(" in raw
        if len(parts) == 1 and name.islower() and not had_signature:
            continue
        if name:
            found.add(name)
    return found


# ---------------------------------------------------------------------------
# Runtime existence check via inspect
# ---------------------------------------------------------------------------

def _module_for_symbol(simple_name: str) -> list[str]:
    """Return the constkit sub-modules likely to contain this symbol."""
    candidates = [
        "constkit",
        "constkit.tensor2",
        "constkit.calculus",
        "constkit.coordinates",
        "constkit.kinematics",
        "constkit.invariants",
        "constkit.stress",
        "constkit.rates",
        "constkit.transforms",
        "constkit.vector",
        "constkit.index_notation",
        "constkit.display",
    ]
    return candidates


def _symbol_exists_at_runtime(sym: str) -> bool:
    """Check that `sym` (possibly `Class.method`) is importable."""
    if "." in sym:
        class_name, method_name = sym.split(".", 1)
        sym = class_name
        attr = method_name
    else:
        attr = None

    for mod_name in _module_for_symbol(sym):
        try:
            mod = importlib.import_module(mod_name)
            obj = getattr(mod, sym, None)
            if obj is None:
                continue
            if attr is not None:
                return hasattr(obj, attr)
            return True
        except ImportError:
            continue
    return False


# ---------------------------------------------------------------------------
# Parameterised tests
# ---------------------------------------------------------------------------

_CODE_SYMS = _code_symbols(_SRC)
_DOC_SYMS = _doc_symbols(_API_REF)

_missing_from_docs = sorted(_CODE_SYMS - _DOC_SYMS)
_phantom_in_docs = sorted(_DOC_SYMS - _CODE_SYMS)


@pytest.mark.parametrize("symbol", _missing_from_docs or ["__no_gaps__"])
def test_code_symbol_is_documented(symbol):
    """Every public code symbol must appear in API_REFERENCE.md."""
    if symbol == "__no_gaps__":
        return  # no missing symbols — test passes vacuously
    pytest.fail(
        f"Symbol '{symbol}' is defined in constkit source but missing from "
        f"API_REFERENCE.md."
    )


@pytest.mark.parametrize("symbol", _phantom_in_docs or ["__no_phantoms__"])
def test_documented_symbol_exists_in_code(symbol):
    """Every documented symbol must exist in the constkit source."""
    if symbol == "__no_phantoms__":
        return  # no phantoms — test passes vacuously
    pytest.fail(
        f"Symbol '{symbol}' is documented in API_REFERENCE.md but not found "
        f"in constkit source."
    )


@pytest.mark.parametrize("symbol", sorted(_DOC_SYMS))
def test_documented_symbol_importable(symbol):
    """Every documented symbol must be importable at runtime."""
    assert _symbol_exists_at_runtime(symbol), (
        f"Symbol '{symbol}' is documented but cannot be found at runtime "
        f"via importlib/inspect."
    )
