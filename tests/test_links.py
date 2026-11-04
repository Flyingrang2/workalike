"""A judge's first clicks must work: the README's code anchors land on what they name, and every path it or the pages link to exists."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
GENERATED = {"LICENSE", "baselines/results/original-commits-per-day.txt"}  # added when the public copy is built


def test_the_code_anchors_land_on_what_they_name():
    anchors = re.findall(r"\[`(\w+)\(\)`\]\(([\w/.]+)#L(\d+)\)", README)
    assert len(anchors) >= 5
    for name, path, line in anchors:
        assert f"def {name}(" in (ROOT / path).read_text().splitlines()[int(line) - 1], (name, path, line)


@pytest.mark.parametrize(
    "doc",
    [
        "README.md",
        "CREDITS.md",
        "baselines/README.md",
        "baselines/results/README.md",
        "scripts/README.md",
        "experiments/README.md",
    ],
)
def test_relative_links_exist(doc):
    text = (ROOT / doc).read_text()
    for target in re.findall(r"\]\((?!https?:|#)([^)#\s]+)", text):
        path = (ROOT / doc).parent / target
        assert path.exists() or str(path.resolve().relative_to(ROOT)) in GENERATED, (doc, target)


@pytest.mark.parametrize("page", ["site/page.html", "site/evidence.html"])
def test_the_pages_link_to_paths_in_the_repo(page):
    links = re.findall(r"github\.com/Flyingrang2/workalike/(?:blob|tree)/main/([^\"#]+)", (ROOT / page).read_text())
    assert links
    for target in links:
        assert (ROOT / target).exists() or target in GENERATED, (page, target)
