from graphs import GRAPH_SMALL, GRAPH_MEDIUM, GRAPH_BIG
from graphs import EXPECTED_GRAPH_SMALL, EXPECTED_GRAPH_MEDIUM, EXPECTED_GRAPH_BIG


def dijkstra(graph, start, end):
    """
    graph: dict de adjacências {nó: [(vizinho, peso), ...]}
    start: nó inicial (P)
    end: nó destino (Q)
    """

    # CONJUNTOS DE NÓS
    A = set()             # Conjunto A: nós com caminho mínimo conhecido
    B = set()             # Conjunsto B: nós candidatos (conectados a A)
    C = set(graph.keys()) # Conjunto C: nós ainda não visitados (restantes)

    # DISTÂNCIAS PROVISÓRIAS
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    prev = {node: None for node in graph}

    # INICIALIZAÇÃO (TRANSFERIR P DE C PARA A)
    A.add(start)
    C.remove(start)
    for neighbor, weight in graph[start]:
        if neighbor in C:
            dist[neighbor] = weight
            prev[neighbor] = start
            B.add(neighbor)
            C.remove(neighbor)

    # LOOP PRINCIPAL
    while end not in A:
        # PASSO 2: escolher nó de B com menor distância
        current = min(B, key=lambda x: dist[x])
        B.remove(current)
        A.add(current)

        # PASSO 1: examinar vizinhos do nó recém-adicionado
        for neighbor, weight in graph[current]:
            if neighbor in C:
                # Nó estava em C: mover para B
                B.add(neighbor)
                C.remove(neighbor)
                dist[neighbor] = dist[current] + weight
                prev[neighbor] = current
            elif neighbor in B:
                # Nó estava em B: verificar se o novo caminho é melhor
                if dist[current] + weight < dist[neighbor]:
                    dist[neighbor] = dist[current] + weight
                    prev[neighbor] = current

    # RECONSTRUÇÃO DO CAMINHO MÍNIMO
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return dist[end], path

def test_dijkstra(graph, graph_name, expected_result):
    distance, path = dijkstra(graph, 'P', 'Q')

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

    print(f"\nTesting {graph_name}")
    print(f"Found: distance={distance}, path={' → '.join(path)}")
    print(f"Expected: distance={expected_distance}, path={' → '.join(expected_path)}")

    distance_match = abs(distance - expected_distance) < 0.0001
    path_exact_match = path == expected_path
    consistency_match = abs(distance - path_distance) < 0.0001

    if distance_match and path_exact_match and path_valid and consistency_match:
        print(f"Values match")
        return True
    else:
        print(f"TEST FAILED")
        if not distance_match:
            print(f"Distance mismatch: expected {expected_distance}, got {distance}")
        if not path_valid:
            print(f"Invalid path")
        if not consistency_match:
            print(f"Internal inconsistency: calculated {distance} ≠ verified {path_distance}")
        return False

def run_all_tests():
    print("DIJKSTRA ALGORITHM TESTS WITH EXPECTED RESULTS")

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
