GRAPH_SMALL = {
    'P': [('A', 2), ('B', 5)],
    'A': [('B', 2), ('C', 4)],
    'B': [('C', 1), ('Q', 7)],
    'C': [('Q', 3)],
    'Q': []
}

GRAPH_MEDIUM = {
    'P': [('A', 4), ('B', 8)],
    'A': [('B', 2), ('C', 5)],
    'B': [('C', 1), ('D', 10)],
    'C': [('D', 3), ('E', 2)],
    'D': [('E', 4), ('F', 6)],
    'E': [('F', 1), ('G', 7)],
    'F': [('G', 2), ('Q', 3)],
    'G': [('Q', 1)],
    'H': [('P', 5)],
    'Q': []
}

GRAPH_BIG = {
    'P': [('A', 2), ('B', 4), ('C', 7)],
    'A': [('D', 3), ('E', 6)],
    'B': [('E', 2), ('F', 5)],
    'C': [('F', 1), ('G', 8)],
    'D': [('H', 4)],
    'E': [('H', 2), ('I', 7)],
    'F': [('I', 3)],
    'G': [('I', 2), ('J', 4)],
    'H': [('K', 1)],
    'I': [('K', 5), ('L', 2)],
    'J': [('L', 1)],
    'K': [('M', 3)],
    'L': [('M', 2)],
    'M': [('Q', 4)],
    'Q': []
}

EXPECTED_GRAPH_SMALL = {
    'expected_distance': 8,
    'expected_path': ['P', 'A', 'B', 'C', 'Q'],
}

EXPECTED_GRAPH_MEDIUM = {
    'expected_distance': 13,
    'expected_path': ['P', 'A', 'B', 'C', 'E', 'F', 'Q'],
}

EXPECTED_GRAPH_BIG = {
    'expected_distance': 16,
    'expected_path': ['P', 'B', 'E', 'H', 'K', 'M', 'Q'],
}
