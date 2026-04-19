import importlib


def test_src_packages_importable():
    modules = [
        "src.core.utils",
        "src.core.models",
        "src.subgraph_matching.config",
        "src.subgraph_mining.config",
        "src.analyze.count_patterns",
        "src.compare.compare",
    ]
    for name in modules:
        importlib.import_module(name)
