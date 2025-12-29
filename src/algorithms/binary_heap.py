import heapq
from graphs import GRAPH_SMALL, GRAPH_MEDIUM, GRAPH_BIG
from graphs import EXPECTED_GRAPH_SMALL, EXPECTED_GRAPH_MEDIUM, EXPECTED_GRAPH_BIG

def dijkstra_binary_heap(graph, start):
    """
    Implementação clássica de Dijkstra usando binary heap (CLRS style)

    graph: dict {u: [(v, weight), ...]}
    start: nó fonte s
    """

    dist = {v: float('inf') for v in graph}
    prev = {v: None for v in graph}
    dist[start] = 0

    heap = [(0, start)]

    visited = set()

    while heap:
        current_dist, u = heapq.heappop(heap)

        if u in visited:
            continue

        visited.add(u)

        for v, weight in graph[u]:
            if dist[v] > current_dist + weight:
                dist[v] = current_dist + weight
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))

    return dist, prev

def reconstruct_path(prev, start, end):
    path = []
    v = end
    while v is not None:
        path.append(v)
        v = prev[v]
    path.reverse()

    if path[0] == start:
        return path
    return None

def test_dijkstra_binary_heap(graph, graph_name, expected_result):
    dist, prev = dijkstra_binary_heap(graph, 'P')
    path = reconstruct_path(prev, 'P', 'Q')
    distance = dist['Q']

    if not isinstance(expected_result, dict):
        return False

    expected_distance = expected_result.get('expected_distance')
    expected_path = expected_result.get('expected_path')

    if expected_distance is None or expected_path is None:
        return False

    path_distance = 0
    path_valid = True

    for i in range(len(path) - 1):
        current = path[i]
        next_node = path[i + 1]
        edge_found = False

        for neighbor, weight in graph[current]:
            if neighbor == next_node:
                path_distance += weight
                edge_found = True
                break

        if not edge_found:
            path_valid = False
            break

    print(f"\nTesting {graph_name} (Binary Heap)")
    print(f"Found: distance={distance}, path={' → '.join(path)}")
    print(f"Expected: distance={expected_distance}, path={' → '.join(expected_path)}")

    distance_match = abs(distance - expected_distance) < 0.0001
    path_exact_match = path == expected_path
    consistency_match = abs(distance - path_distance) < 0.0001

    if distance_match and path_exact_match and path_valid and consistency_match:
        print("Values match")
        return True
    else:
        print("TEST FAILED")
        if not distance_match:
            print(f"Distance mismatch: expected {expected_distance}, got {distance}")
        if not path_valid:
            print("Invalid path")
        if not consistency_match:
            print(f"Internal inconsistency: calculated {distance} ≠ verified {path_distance}")
        return False


def run_all_tests():
    print("DIJKSTRA BINARY HEAP TESTS WITH EXPECTED RESULTS")

    results = []

    test1_passed = test_dijkstra_binary_heap(
        GRAPH_SMALL, "GRAPH_SMALL", EXPECTED_GRAPH_SMALL
    )
    results.append(("GRAPH_SMALL", test1_passed))

    test2_passed = test_dijkstra_binary_heap(
        GRAPH_MEDIUM, "GRAPH_MEDIUM", EXPECTED_GRAPH_MEDIUM
    )
    results.append(("GRAPH_MEDIUM", test2_passed))

    test3_passed = test_dijkstra_binary_heap(
        GRAPH_BIG, "GRAPH_BIG", EXPECTED_GRAPH_BIG
    )
    results.append(("GRAPH_BIG", test3_passed))

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    if passed_count == total_count:
        print("\nAll tests passed!")
    else:
        print(f"\n{total_count - passed_count} test(s) failed")

    return passed_count == total_count

if __name__ == "__main__":
    try:
        all_passed = run_all_tests()
        exit(0 if all_passed else 1)
    except Exception as e:
        print(f"\nError during tests: {e}")
        exit(1)
