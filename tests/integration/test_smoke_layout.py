from pathlib import Path


def test_src_layout_exists():
    root = Path(__file__).resolve().parents[2]
    expected = [
        root / "src" / "core",
        root / "src" / "subgraph_matching",
        root / "src" / "subgraph_mining",
        root / "src" / "analyze",
        root / "src" / "compare",
    ]
    for p in expected:
        assert p.exists()
