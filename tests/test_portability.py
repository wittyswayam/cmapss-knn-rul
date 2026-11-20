import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def reused_quote_fstrings(path: Path) -> list[int]:
    """Lines where an f-string contains a string literal with the same quote character (valid only on Python 3.12+)."""
    bad, stack = [], []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            name = tokenize.tok_name[tok.type]
            if name == "FSTRING_START":
                stack.append(tok.string.lstrip("fFrRbBuU"))
            elif name == "FSTRING_END":
                stack.pop()
            elif name == "STRING" and stack and tok.string.lstrip("fFrRbBuU").startswith(stack[-1]):
                bad.append(tok.start[0])
    return bad


@unittest.skipUnless(hasattr(tokenize, "FSTRING_START"), "needs the Python 3.12 tokenizer to detect the pattern")
class PortabilityTests(unittest.TestCase):
    def test_no_python_312_only_fstring_syntax(self):
        """The project promises Python 3.10+, where an f-string cannot reuse its own quote character inside braces."""
        offenders = {str(p.relative_to(ROOT)): reused_quote_fstrings(p) for folder in ("src", "scripts", "tests") for p in (ROOT / folder).rglob("*.py")}
        self.assertEqual({k: v for k, v in offenders.items() if v}, {})


if __name__ == "__main__":
    unittest.main()
