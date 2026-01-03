try:
    from .graphs import (
        GRAPH_SMALL, GRAPH_MEDIUM, GRAPH_BIG,
        EXPECTED_GRAPH_SMALL, EXPECTED_GRAPH_MEDIUM, EXPECTED_GRAPH_BIG
    )
except ImportError:
    from graphs import (
        GRAPH_SMALL, GRAPH_MEDIUM, GRAPH_BIG,
        EXPECTED_GRAPH_SMALL, EXPECTED_GRAPH_MEDIUM, EXPECTED_GRAPH_BIG
    )

def dijkstra(graph, start):
    """
    Complete SSSP
    graph: adjacency dictionary {node: [(neighbor, weight), ...]}
    start: starting node (e.g., 'P')

    Returns:
      dist: dict {node: shortest distance from start}
      prev: dict {node: predecessor on the shortest path}
    """
    A = set()
    B = set()
    C = set(graph.keys())

    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    dist[start] = 0

    A.add(start)
    C.remove(start)

    for neighbor, weight in graph[start]:
        if neighbor in C:
            dist[neighbor] = weight
            prev[neighbor] = start
            B.add(neighbor)
            C.remove(neighbor)

    while B:
        current = min(B, key=lambda x: dist[x])
        B.remove(current)
        A.add(current)

        for neighbor, weight in graph[current]:
            if neighbor in C:
                B.add(neighbor)
                C.remove(neighbor)
                dist[neighbor] = dist[current] + weight
                prev[neighbor] = current
            elif neighbor in B:
                new_dist = dist[current] + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = current

    return dist, prev

def build_path(prev, start, end):
    """
    Reconstructs the path from start to end using the prev dictionary.
    Returns a list of nodes or None if end is unreachable.
    """
    if start == end:
        return [start]
    if end not in prev or prev[end] is None:
        return None

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return path if path and path[0] == start else None

def test_dijkstra(graph, graph_name, expected_result):
    dist, prev = dijkstra(graph, 'P')

    if not isinstance(expected_result, dict):
        return False

    expected_distance = expected_result.get('expected_distance')
    expected_path = expected_result.get('expected_path')

    if expected_distance is None or expected_path is None:
        return False

    end = expected_path[-1]
    distance = dist.get(end, float('inf'))
    path = build_path(prev, 'P', end)

    path_valid = True
    path_distance = 0

    if path is None:
        path_valid = False
    else:
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

    found_path_str = "∅" if path is None else " → ".join(path)

    print(f"\nTesting {graph_name} (SSSP)")
    print(f"Found: distance={distance}, path={found_path_str}")
    print(f"Expected: distance={expected_distance}, path={' → '.join(expected_path)}")

    distance_match = abs(distance - expected_distance) < 0.0001
    path_exact_match = (path == expected_path)
    consistency_match = (path is not None) and (abs(distance - path_distance) < 0.0001)

    if distance_match and path_exact_match and path_valid and consistency_match:
        print("Values match")
        return True
    else:
        print("TEST FAILED")
        if not distance_match:
            print(f"Distance mismatch: expected {expected_distance}, got {distance}")
        if not path_exact_match:
            print(f"Path mismatch: expected {' → '.join(expected_path)}, got {found_path_str}")
        if not path_valid:
            print("Invalid or unreachable path")
        if path is not None and not consistency_match:
            print(f"Internal inconsistency: calculated {distance} ≠ verified {path_distance}")
        return False

def run_all_tests():
    print("DIJKSTRA ALGORITHM TESTS WITH EXPECTED RESULTS (SSSP)")

    results = []

    test1_passed = test_dijkstra(GRAPH_SMALL, "GRAPH_SMALL", EXPECTED_GRAPH_SMALL)
    results.append(("GRAPH_SMALL", test1_passed))

    test2_passed = test_dijkstra(GRAPH_MEDIUM, "GRAPH_MEDIUM", EXPECTED_GRAPH_MEDIUM)
    results.append(("GRAPH_MEDIUM", test2_passed))

    test3_passed = test_dijkstra(GRAPH_BIG, "GRAPH_BIG", EXPECTED_GRAPH_BIG)
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
