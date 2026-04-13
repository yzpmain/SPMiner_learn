import pickle
from pathlib import Path
from collections import Counter, defaultdict
import networkx as nx

path = Path('results/mining_as733_smoke.pkl')
data = pickle.load(path.open('rb'))
print('num_patterns', len(data))
size_counts = Counter(len(g) for g in data)
print('size_counts', dict(sorted(size_counts.items())))

wl = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash
by_size = defaultdict(list)
for g in data:
    by_size[len(g)].append(g)

for size in sorted(by_size):
    hashes = Counter(wl(g) for g in by_size[size])
    print(f'\nSIZE {size} total {len(by_size[size])} unique_wl {len(hashes)}')
    for h, c in hashes.most_common(2):
        g = next(g for g in by_size[size] if wl(g) == h)
        edges = sorted(tuple(sorted(e)) for e in g.edges())
        degrees = sorted((d for _, d in g.degree()), reverse=True)
        anchors = [n for n, d in g.nodes(data=True) if d.get('anchor') == 1]
        print('  REP', c, 'nodes', len(g), 'edges', g.number_of_edges(), 'deg', degrees, 'anchor', anchors)
        print('  EDGES', edges)
