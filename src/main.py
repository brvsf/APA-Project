import time
from algorithms.dijkstra import dijkstra
from algorithms.binary_heap import dijkstra_binary_heap, reconstruct_path
from algorithms.graphs import GRAPH_SMALL, GRAPH_MEDIUM, GRAPH_BIG

def main():
    print("===== DIJKSTRA =====")
    time_start = time.time()
    dijk_res_small_dist, dijk_res_small_path = dijkstra(GRAPH_SMALL, 'P', 'Q')
    time_end = time.time()
    time_sub_dijk_small = time_end - time_start
    print(f"Dijkstra with small graph time: {time_sub_dijk_small}")

    time_start = time.time()
    dijk_res_med_dist, dijk_res_med_path = dijkstra(GRAPH_MEDIUM, 'P', 'Q')
    time_end = time.time()
    time_sub_dijk_med = time_end - time_start
    print(f"Dijkstra with medium graph time: {time_sub_dijk_med}")

    time_start = time.time()
    dijk_res_big_dist, dijk_res_big_path = dijkstra(GRAPH_MEDIUM, 'P', 'Q')
    time_end = time.time()
    time_sub_dijk_big = time_end - time_start
    print(f"Dijkstra with big graph time: {time_sub_dijk_big}")
    print("="*20)


if __name__ == '__main__':
    main()
