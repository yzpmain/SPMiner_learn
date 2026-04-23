import networkx as nx
from orbitsi.search import OrbitSIEngine
from orbitsi.orbit import ORCAOrbitCounter

try:
    G = nx.path_graph(5)
    P = nx.path_graph(3)
    eng = OrbitSIEngine(G, ORCAOrbitCounter, graphlet_size=4)
    r1 = eng.run(P)
    r2 = eng.run(P)
    print(f'Type r1: {type(r1)} Value: {r1}')
    print(f'Type r2: {type(r2)}')
    if hasattr(r2, "__len__"):
        print(f'len: {len(r2)}')
    if isinstance(r2, list):
        print(f'sample: {r2[:3]}')
    else:
        print(f'r2: {r2}')
except Exception:
    import traceback
    traceback.print_exc()
