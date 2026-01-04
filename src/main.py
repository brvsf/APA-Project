import time
import gc
import csv
import random
import statistics as stats

from algorithms.dijkstra import dijkstra
from algorithms.binary_heap import dijkstra_binary_heap
from algorithms.duan import duan_breaking_sorting_barrier_sssp

def benchmark(func, runs=5, warmup=1):
    for _ in range(warmup):
        func()

    times = []
    for _ in range(runs):
        gc.disable()
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        gc.enable()
        times.append(t1 - t0)

    return stats.mean(times), stats.pstdev(times)

def make_sparse_digraph(n, out_degree=4, wmin=1, wmax=10, seed=123):
    rng = random.Random(seed)
    nodes = [str(i) for i in range(n)]
    g = {u: [] for u in nodes}

    for i in range(1, n):
        u = nodes[i - 1]
        v = nodes[i]
        g[u].append((v, float(rng.randint(wmin, wmax))))

    for u in nodes:
        existing = {v for (v, _) in g[u]}
        need = max(0, out_degree - len(g[u]))

        while need > 0:
            v = rng.choice(nodes)
            if v == u or v in existing:
                continue
            g[u].append((v, float(rng.randint(wmin, wmax))))
            existing.add(v)
            need -= 1

    return g

def main():
    RUNS = 5
    WARMUP = 1
    OUT_DEGREE = 4

    N_LIST = list(range(1_000, 25_001, 1_000))

    out_path = "results.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "dijkstra_mean", "dijkstra_std",
            "heap_mean", "heap_std",
            "duan_mean", "duan_std",
        ])

        print("Running benchmarks and exporting CSV...")
        print(f"Output: {out_path}\n")

        for n in N_LIST:
            G = make_sparse_digraph(n, out_degree=OUT_DEGREE, seed=123)
            src = "0"

            d_mean, d_std = benchmark(lambda: dijkstra(G, src), RUNS, WARMUP)
            h_mean, h_std = benchmark(lambda: dijkstra_binary_heap(G, src), RUNS, WARMUP)
            b_mean, b_std = benchmark(lambda: duan_breaking_sorting_barrier_sssp(G, src), RUNS, WARMUP)

            writer.writerow([n, d_mean, d_std, h_mean, h_std, b_mean, b_std])

            print(
                f"n={n:6d} | "
                f"dij={d_mean:.6f}s | "
                f"heap={h_mean:.6f}s | "
                f"duan={b_mean:.6f}s"
            )

if __name__ == "__main__":
    main()
