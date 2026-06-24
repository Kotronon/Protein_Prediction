from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from protein_prediction_io import (  # noqa: E402
    FastaRecord,
    extract_uniprot_accession,
    read_fasta,
    read_fasta_dict,
    safe_filename,
    write_caid_scores,
    write_fasta_record,
)


class ProteinPredictionIoTests(unittest.TestCase):
    def test_read_fasta_preserves_headers_and_can_uppercase_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.fasta"
            path.write_text(">sp|P12345|NAME description\nacD\nEf\n", encoding="utf-8")

            records = read_fasta(path, uppercase=True)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].header, "sp|P12345|NAME description")
        self.assertEqual(records[0].sequence, "ACDEF")
        self.assertEqual(records[0].uniprot_accession, "P12345")

    def test_read_fasta_dict_rejects_duplicate_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.fasta"
            path.write_text(">a\nAA\n>a\nBB\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate FASTA header"):
                read_fasta_dict(path)

    def test_safe_filename_and_uniprot_fallbacks(self) -> None:
        self.assertEqual(extract_uniprot_accession(">tr|Q9XYZ1|NAME text"), "Q9XYZ1")
        self.assertEqual(extract_uniprot_accession("plain_id details"), "plain_id")
        self.assertEqual(safe_filename("bad/id|x"), "bad_id_x")
        self.assertEqual(safe_filename("////", fallback="protein"), "protein")

    def test_write_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fasta_path = Path(tmp) / "out.fasta"
            caid_path = Path(tmp) / "out.caid"

            with fasta_path.open("w", encoding="utf-8") as handle:
                write_fasta_record(handle, FastaRecord("h", "ABCDEFG"), width=3)
            write_caid_scores(caid_path, "h", "AC", [0.1, 0.2], precision=2)

            self.assertEqual(fasta_path.read_text(encoding="utf-8"), ">h\nABC\nDEF\nG\n")
            self.assertEqual(
                caid_path.read_text(encoding="utf-8"),
                ">h\n1\tA\t0.10\n2\tC\t0.20\n",
            )


if __name__ == "__main__":
    unittest.main()
