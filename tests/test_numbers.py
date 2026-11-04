"""The public numbers are recomputed from the raw results, and the README and the pages must say the same."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_numbers", ROOT / "scripts/check_numbers.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def numbers():
    return {
        "main": check.main_test(),
        "throughput": check.throughput(),
        "practice": check.practice(),
        "people": check.people(),
        "learning": check.learning_round(),
        "product": check.rule_change(),
        "demo": check.demo(),
        "person": check.person(),
        "sheets": check.hashes(),
    }


def test_the_readme_and_the_pages_say_what_the_raw_results_say():
    assert check.public(numbers()) == []


def test_the_frozen_sheets_still_match_their_hashes():
    assert check.hashes()["not matching"] == 0


def test_the_loras_prompt_held_no_rules():
    manifest = (ROOT / "baselines/results/test24b/manifest.json").read_text()
    assert (
        '"p-checked": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"' in manifest
    )  # the sha256 of nothing
