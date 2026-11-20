import math
import re
import unittest
from pathlib import Path

from cmapss_knn.experiments import provenance as pv

ROOT = Path(__file__).resolve().parents[1]

REPORT = """# Title

## A section

### FD002

| model | x |
|---|---|
| m1 | 1.500 |

## Another

### FD002

| metric | y |
|---|---|
| r | 2 |
"""


class MarkdownParsingTests(unittest.TestCase):
    def test_section_isolates_the_right_duplicate_heading(self):
        sec = pv.md_section(REPORT, "## A section")
        tabs = pv.parse_md_tables(sec, "### FD002")
        self.assertEqual(len(tabs), 1)
        self.assertEqual(list(tabs[0].columns), ["model", "x"])
        self.assertEqual(tabs[0]["x"].iloc[0], "1.500")
        with self.assertRaises(ValueError):
            pv.md_section(REPORT, "## Missing")


class LiteralScanTests(unittest.TestCase):
    def test_report_and_figure_code_contains_only_reviewed_numeric_literals(self):
        self.assertEqual(pv.literal_scan(ROOT / "src" / "cmapss_knn"), [])


class DocumentationTests(unittest.TestCase):
    def test_nasa_table_in_the_protocol_matches_the_formula(self):
        rows = re.findall(r"^\| ([+-]\d+) \| (?:early|late)[^|]*\| ([0-9.]+) \|$", (ROOT / "docs" / "EVALUATION_PROTOCOL.md").read_text(), flags=re.M)
        self.assertEqual(len(rows), 6)
        for d, s in rows:
            d = int(d)
            expected = math.exp(-d / 13) - 1 if d < 0 else math.exp(d / 10) - 1
            self.assertAlmostEqual(float(s), expected, places=2)

    def test_protocol_embeds_no_experiment_results(self):
        text = (ROOT / "docs" / "EVALUATION_PROTOCOL.md").read_text()
        self.assertNotIn("42.6", text)
        self.assertNotIn("14.0 to 16.2", text)


if __name__ == "__main__":
    unittest.main()
