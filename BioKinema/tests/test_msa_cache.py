import json
import multiprocessing
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from protenix.data.msa_cache import (
    get_or_create_msa,
    sequence_cache_key,
    validate_msa_cache,
)


def _write_msa_result(sequence: str, build_dir: Path, query_fasta: Path) -> None:
    del query_fasta
    (build_dir / "0.a3m").write_text(f">query_0\n{sequence}\n", encoding="utf-8")
    result_dir = build_dir / "0"
    result_dir.mkdir()
    for filename in ("non_pairing.a3m", "pairing.a3m"):
        (result_dir / filename).write_text(
            f">query\n{sequence}\n>homolog\n{sequence}\n", encoding="utf-8"
        )


def _concurrent_worker(sequence: str, cache_root: str, build_log: str) -> None:
    def build(seq: str, build_dir: Path, query_fasta: Path) -> None:
        with open(build_log, "a", encoding="utf-8") as handle:
            handle.write("build\n")
        time.sleep(0.2)
        _write_msa_result(seq, build_dir, query_fasta)

    get_or_create_msa(sequence, cache_root, build)


class MSACacheTest(unittest.TestCase):
    def test_same_sequence_is_built_once(self):
        with TemporaryDirectory() as temp_dir:
            builds = []

            def build(sequence, build_dir, query_fasta):
                builds.append(sequence)
                _write_msa_result(sequence, build_dir, query_fasta)

            first = get_or_create_msa(" acd e ", temp_dir, build)
            second = get_or_create_msa("ACDE", temp_dir, build)

            self.assertEqual(first, second)
            self.assertEqual(builds, ["ACDE"])
            self.assertTrue((first / "non_pairing.a3m").is_file())

    def test_two_proteins_are_cached_independently_and_reused(self):
        with TemporaryDirectory() as temp_dir:
            builds = []

            def build(sequence, build_dir, query_fasta):
                builds.append(sequence)
                _write_msa_result(sequence, build_dir, query_fasta)

            seq_a = "ACDEFGHIK"
            seq_b = "LMNPQRSTV"
            path_a = get_or_create_msa(seq_a, temp_dir, build)
            path_b = get_or_create_msa(seq_b, temp_dir, build)

            # Reversed protein order and a different ligand do not alter MSA identity.
            self.assertEqual(path_b, get_or_create_msa(seq_b, temp_dir, build))
            self.assertEqual(path_a, get_or_create_msa(seq_a, temp_dir, build))
            self.assertEqual(builds, [seq_a, seq_b])
            self.assertNotEqual(path_a, path_b)

    def test_corrupt_entry_is_rebuilt(self):
        with TemporaryDirectory() as temp_dir:
            build_count = 0

            def build(sequence, build_dir, query_fasta):
                nonlocal build_count
                build_count += 1
                _write_msa_result(sequence, build_dir, query_fasta)

            sequence = "ACDEFG"
            result_dir = get_or_create_msa(sequence, temp_dir, build)
            (result_dir / "non_pairing.a3m").write_text(
                ">query\nWRONG\n", encoding="utf-8"
            )

            rebuilt_dir = get_or_create_msa(sequence, temp_dir, build)
            valid, reason = validate_msa_cache(rebuilt_dir.parent, sequence)
            self.assertTrue(valid, reason)
            self.assertEqual(build_count, 2)

    def test_failed_build_is_not_committed(self):
        with TemporaryDirectory() as temp_dir:
            sequence = "ACDEFG"

            def fail_build(sequence, build_dir, query_fasta):
                del sequence, build_dir, query_fasta
                raise RuntimeError("server unavailable")

            with self.assertRaisesRegex(RuntimeError, "server unavailable"):
                get_or_create_msa(sequence, temp_dir, fail_build)

            final_dir = Path(temp_dir) / "v1" / sequence_cache_key(sequence)
            self.assertFalse(final_dir.exists())

            result_dir = get_or_create_msa(sequence, temp_dir, _write_msa_result)
            manifest = json.loads(
                (result_dir.parent / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")

    def test_query_only_fallback_without_raw_result_is_not_committed(self):
        with TemporaryDirectory() as temp_dir:
            sequence = "ACDEFG"

            def write_dummy_only(sequence, build_dir, query_fasta):
                del query_fasta
                result_dir = build_dir / "0"
                result_dir.mkdir()
                for filename in ("non_pairing.a3m", "pairing.a3m"):
                    (result_dir / filename).write_text(
                        f">query\n{sequence}\n", encoding="utf-8"
                    )

            with self.assertRaisesRegex(RuntimeError, "0.a3m"):
                get_or_create_msa(sequence, temp_dir, write_dummy_only)

            final_dir = Path(temp_dir) / "v1" / sequence_cache_key(sequence)
            self.assertFalse(final_dir.exists())

    def test_parallel_requests_build_once(self):
        with TemporaryDirectory() as temp_dir:
            sequence = "ACDEFGHIKLMNPQRSTVWY"
            build_log = str(Path(temp_dir) / "build.log")
            context = multiprocessing.get_context("fork")
            processes = [
                context.Process(
                    target=_concurrent_worker,
                    args=(sequence, temp_dir, build_log),
                )
                for _ in range(4)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)

            lines = Path(build_log).read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines, ["build"])


if __name__ == "__main__":
    unittest.main()
