import math
import heapq
from collections import deque
from bisect import bisect_left, insort

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

INF = float("inf")

def _tie_better(u, prev_u):
    """
    Deterministic tie-breaking when multiple relaxations produce the same db[v].
    Used to keep predecessor choices stable across runs.
    """
    if prev_u is None:
        return True
    return str(u) < str(prev_u)

class SSSPContext:
    """
    State maintained by the BMSSP procedure.

    - db[v]: tentative distance label updated by relaxations in Algorithms 1–3.
    - pred[v]: predecessor pointer consistent with db updates.
    - complete[v]: completeness flag used by the BMSSP contract:
      Algorithm 2 requires S={x} with x complete; BMSSP marks returned U as complete.
    """

    def __init__(self, graph, source):
        self.g = graph
        self.db = {v: INF for v in graph}
        self.pred = {v: None for v in graph}
        self.complete = {v: False for v in graph}

        self.db[source] = 0.0
        self.complete[source] = True

class Block:
    """
    Block container for Lemma 3.3 structure D.
    Tracks (minv, maxv) to support boundary/upper-bound reasoning across blocks.
    """
    __slots__ = ("items", "minv", "maxv")

    def __init__(self, items=None):
        self.items = []
        self.minv = INF
        self.maxv = -INF
        if items:
            self.items = list(items)
            self.items.sort()
            self._recalc()

    def _recalc(self):
        if not self.items:
            self.minv = INF
            self.maxv = -INF
            return
        self.minv = self.items[0][0]
        self.maxv = self.items[-1][0]

    def add_sorted(self, pair):
        insort(self.items, pair)
        v, _k = pair
        if len(self.items) == 1:
            self.minv = v
            self.maxv = v
        else:
            if v < self.minv:
                self.minv = v
            if v > self.maxv:
                self.maxv = v

    def remove_key(self, key):
        for i, (_v, k) in enumerate(self.items):
            if k == key:
                self.items.pop(i)
                self._recalc()
                return True
        return False

    def pop_at(self, idx):
        pair = self.items.pop(idx)
        self._recalc()
        return pair

    def __len__(self):
        return len(self.items)

class Lemma33_BlockD:
    """
    Data structure D implementing the API of Lemma 3.3.

      Initialize(M, B)
      Insert(<x, val>)
      BatchPrepend(list of <x, val>)   (values smaller than any current value)
      Pull() -> (x, S')               (x is next boundary; |S'| <= M)
      non_empty()

    Organization follows the lemma's two sequences of blocks:
      - D0: blocks created by BatchPrepend
      - D1: blocks used by Insert, located via upper bounds
    """

    def __init__(self):
        self.M = 1
        self.B = INF
        self.D0 = deque()
        self.D1 = []
        self.D1_ubs = []
        self.best = {}
        self.key_loc = {}

    def Initialize(self, M, B):
        self.M = max(1, int(M))
        self.B = B
        self.D0.clear()
        self.D1.clear()
        self.D1_ubs.clear()
        self.best.clear()
        self.key_loc.clear()

        empty = Block()
        self.D1.append(empty)
        self.D1_ubs.append(B)

    def non_empty(self):
        return bool(self.best)

    def _delete_key(self, key):
        if key not in self.best:
            return
        where = self.key_loc.get(key)
        del self.best[key]
        self.key_loc.pop(key, None)

        if not where:
            return

        seq, idx = where
        if seq == "D0":
            for blk in self.D0:
                if blk.remove_key(key):
                    break
        else:
            if idx is not None and 0 <= idx < len(self.D1):
                self.D1[idx].remove_key(key)

        self._cleanup()

    def Insert(self, pair):
        """
        Lemma 3.3 Insert:
          - stores only the smallest value per key
          - locates the first D1 block with upper bound >= val
          - splits a block if it exceeds size M
        """
        key, val = pair

        old = self.best.get(key)
        if old is not None and val >= old:
            return
        if old is not None:
            self._delete_key(key)

        self.best[key] = val

        j = bisect_left(self.D1_ubs, val)
        if j == len(self.D1):
            j = len(self.D1) - 1

        blk = self.D1[j]
        blk.add_sorted((val, key))
        self.key_loc[key] = ("D1", j)
        self.D1_ubs[j] = blk.maxv

        if len(blk) > self.M:
            items = blk.items
            mid = len(items) // 2

            left_blk = Block(items[:mid])
            right_blk = Block(items[mid:])

            self.D1[j] = left_blk
            self.D1_ubs[j] = left_blk.maxv

            self.D1.insert(j + 1, right_blk)
            self.D1_ubs.insert(j + 1, right_blk.maxv)

            for v, k in left_blk.items:
                if self.best.get(k) == v:
                    self.key_loc[k] = ("D1", j)
            for v, k in right_blk.items:
                if self.best.get(k) == v:
                    self.key_loc[k] = ("D1", j + 1)

        self._cleanup()

    def BatchPrepend(self, pairs):
        """
        Lemma 3.3 BatchPrepend:
          - precondition: all values in L are smaller than any current value in D
          - stores only the smallest value per key
          - prepends blocks into D0
        """
        if not pairs:
            return

        cur_min = min(self.best.values()) if self.best else INF

        local_best = {}
        for k, v in pairs:
            if (k not in local_best) or (v < local_best[k]):
                local_best[k] = v

        prepend = []
        to_insert = []

        for k, v in local_best.items():
            old = self.best.get(k)
            if old is not None and v >= old:
                continue
            if v <= cur_min:
                if old is not None:
                    self._delete_key(k)
                self.best[k] = v
                prepend.append((v, k))
            else:
                to_insert.append((k, v))

        for k, v in to_insert:
            self.Insert((k, v))

        if not prepend:
            return

        prepend.sort()

        lo = (self.M + 1) // 2
        blocks = []
        i = 0
        n = len(prepend)

        while i < n:
            take = min(self.M, n - i)
            blocks.append(Block(prepend[i:i + take]))
            i += take

        if len(blocks) >= 2 and len(blocks[-1]) < lo:
            need = lo - len(blocks[-1])
            donor = blocks[-2]
            if len(donor) - need >= lo:
                moved = donor.items[-need:]
                donor.items = donor.items[:-need]
                donor._recalc()

                blocks[-1].items = moved + blocks[-1].items
                blocks[-1].items.sort()
                blocks[-1]._recalc()

        for blk in reversed(blocks):
            self.D0.appendleft(blk)
            for v, k in blk.items:
                if self.best.get(k) == v:
                    self.key_loc[k] = ("D0", None)

        self._cleanup()

    def _cleanup(self):
        while self.D0 and len(self.D0[0]) == 0:
            self.D0.popleft()

        i = 0
        while i < len(self.D1):
            if len(self.D1[i]) == 0 and len(self.D1) > 1:
                self.D1.pop(i)
                self.D1_ubs.pop(i)
                for k2, loc in list(self.key_loc.items()):
                    if loc and loc[0] == "D1":
                        self.key_loc[k2] = ("D1", None)
                continue
            i += 1

        for i, blk in enumerate(self.D1):
            self.D1_ubs[i] = self.B if len(blk) == 0 else blk.maxv

        run = -INF
        for i in range(len(self.D1_ubs)):
            if self.D1_ubs[i] < run:
                self.D1_ubs[i] = run
            else:
                run = self.D1_ubs[i]

    def Pull(self):
        """
        Lemma 3.3 Pull:
          - extracts at most M keys corresponding to the smallest values
          - returns boundary x: B if empty, else the smallest remaining value
        """
        if not self.best:
            return self.B, []

        h = []
        for bi, blk in enumerate(self.D0):
            if len(blk) > 0:
                v, k = blk.items[0]
                heapq.heappush(h, (v, k, 0, bi, 0))

        for bi, blk in enumerate(self.D1):
            if len(blk) > 0:
                v, k = blk.items[0]
                heapq.heappush(h, (v, k, 1, bi, 0))

        chosen = []
        chosen_locs = []
        seen_keys = set()

        while h and len(chosen) < self.M:
            v, k, seq, bi, ii = heapq.heappop(h)

            blk = self.D0[bi] if seq == 0 else self.D1[bi]
            ni = ii + 1
            if ni < len(blk.items):
                nv, nk = blk.items[ni]
                heapq.heappush(h, (nv, nk, seq, bi, ni))

            cur = self.best.get(k)
            if cur is None or cur != v:
                continue
            if k in seen_keys:
                continue

            seen_keys.add(k)
            chosen.append(k)
            chosen_locs.append((seq, bi, ii, k))

        by_block = {}
        for seq, bi, ii, k in chosen_locs:
            by_block.setdefault((seq, bi), []).append((ii, k))

        for (seq, bi), lst in by_block.items():
            lst.sort(reverse=True)
            blk = self.D0[bi] if seq == 0 else self.D1[bi]
            for ii, k in lst:
                if ii < len(blk.items) and blk.items[ii][1] == k:
                    blk.pop_at(ii)
                else:
                    blk.remove_key(k)
                self.best.pop(k, None)
                self.key_loc.pop(k, None)

        self._cleanup()

        x = self.B if not self.best else min(self.best.values())
        return x, chosen

def FindPivots(ctx: SSSPContext, B: float, S: set, k: int):
    W = set(S)
    W_prev = set(S)

    for _i in range(1, k + 1):
        W_i = set()
        for u in W_prev:
            for v, w in ctx.g[u]:
                nd = ctx.db[u] + w
                if nd <= ctx.db[v]:
                    if nd < ctx.db[v] or _tie_better(u, ctx.pred[v]):
                        ctx.db[v] = nd
                        ctx.pred[v] = u
                    if nd < B:
                        W_i.add(v)

        W |= W_i
        if len(W) > k * len(S):
            return set(S), W
        W_prev = W_i

    parent = {v: None for v in W}
    children = {v: [] for v in W}

    for u in W:
        du = ctx.db[u]
        for v, w in ctx.g[u]:
            if v not in W:
                continue
            if ctx.db[v] == du + w:
                pu = parent[v]
                if pu is None or _tie_better(u, pu):
                    parent[v] = u

    for v in W:
        p = parent[v]
        if p is not None:
            children[p].append(v)

    def subtree_size(root):
        q = deque([root])
        seen = {root}
        while q:
            x = q.popleft()
            for y in children.get(x, []):
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        return len(seen)

    P = set()
    for u in S:
        if u in W and parent[u] is None:
            if subtree_size(u) >= k:
                P.add(u)

    return P, W

def BaseCase(ctx: SSSPContext, B: float, S: set, k: int):
    if len(S) != 1:
        return B, set()

    (x,) = tuple(S)
    if not ctx.complete.get(x, False):
        raise ValueError("BaseCase precondition violated: x must be complete.")

    U0 = set(S)
    H = [(ctx.db[x], x)]

    while H and len(U0) < k + 1:
        du, u = heapq.heappop(H)
        if du != ctx.db[u]:
            continue

        U0.add(u)
        for v, w in ctx.g[u]:
            nd = ctx.db[u] + w
            if nd <= ctx.db[v] and nd < B:
                if nd < ctx.db[v] or _tie_better(u, ctx.pred[v]):
                    ctx.db[v] = nd
                    ctx.pred[v] = u
                heapq.heappush(H, (ctx.db[v], v))

    if len(U0) <= k:
        Bp, U = B, U0
    else:
        Bp = max(ctx.db[v] for v in U0)
        U = {v for v in U0 if ctx.db[v] < Bp}

    for v in U:
        ctx.complete[v] = True

    return Bp, U

def BMSSP(ctx: SSSPContext, l: int, B: float, S: set, k: int, t: int, D: Lemma33_BlockD):
    if not S:
        return B, set()

    if l == 0:
        return BaseCase(ctx, B, S, k)

    P, W = FindPivots(ctx, B, S, k)

    M = 2 ** ((l - 1) * t)
    D.Initialize(M, B)

    for x in P:
        D.Insert((x, ctx.db[x]))

    Bp_last = min((ctx.db[x] for x in P), default=B)

    U = set()
    capU = (k * k) * (2 ** (l * t))

    while len(U) < capU and D.non_empty():
        Bi, Si_list = D.Pull()
        if not Si_list:
            break

        Si = set(Si_list)
        for x in Si:
            ctx.complete[x] = True

        Bp_i, Ui = BMSSP(ctx, l - 1, Bi, Si, k, t, D)
        U |= Ui
        Bp_last = Bp_i

        K = []
        for u in Ui:
            for v, w in ctx.g[u]:
                nd = ctx.db[u] + w
                if nd <= ctx.db[v]:
                    if nd < ctx.db[v] or _tie_better(u, ctx.pred[v]):
                        ctx.db[v] = nd
                        ctx.pred[v] = u

                    if Bi <= nd < B:
                        D.Insert((v, nd))
                    elif Bp_i <= nd < Bi:
                        K.append((v, nd))

        extras = [(x, ctx.db[x]) for x in Si if Bp_i <= ctx.db[x] < Bi]
        D.BatchPrepend(K + extras)

    Bfinal = min(Bp_last, B)
    U |= {x for x in W if ctx.db[x] < Bfinal}

    for v in U:
        ctx.complete[v] = True

    return Bfinal, U

def duan_breaking_sorting_barrier_sssp(graph, source):
    n = max(2, len(graph))
    logn = max(1.0, math.log2(n))

    k = max(1, int(logn ** (1.0 / 3.0)))
    t = max(1, int(logn ** (2.0 / 3.0)))
    l = int(math.ceil(logn / t))

    ctx = SSSPContext(graph, source)
    D = Lemma33_BlockD()

    BMSSP(ctx, l, INF, {source}, k, t, D)

    dist = {v: ctx.db[v] for v in graph}
    prev = {v: ctx.pred[v] for v in graph}

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

def test_duan(graph, graph_name, expected_result):
    dist, prev = duan_breaking_sorting_barrier_sssp(graph, "P")

    expected_distance = expected_result.get("expected_distance")
    expected_path = expected_result.get("expected_path")
    if expected_distance is None or expected_path is None:
        return False

    end = expected_path[-1]
    found_dist = dist.get(end, INF)
    found_path = build_path(prev, "P", end)

    path_ok = found_path is not None
    path_dist = 0.0
    if path_ok:
        for i in range(len(found_path) - 1):
            u, v = found_path[i], found_path[i + 1]
            w_uv = None
            for nei, w in graph[u]:
                if nei == v:
                    w_uv = w
                    break
            if w_uv is None:
                path_ok = False
                break
            path_dist += w_uv

    print(f"\nTesting {graph_name} (BMSSP)")
    print(f"Found: distance={found_dist}, path={'∅' if found_path is None else ' → '.join(found_path)}")
    print(f"Expected: distance={expected_distance}, path={' → '.join(expected_path)}")

    dist_ok = abs(found_dist - expected_distance) < 1e-9
    exact_path_ok = (found_path == expected_path)
    consistent = path_ok and abs(found_dist - path_dist) < 1e-9

    if dist_ok and exact_path_ok and consistent:
        print("Values match")
        return True

    print("TEST FAILED")
    if not dist_ok:
        print(f"Distance mismatch: expected {expected_distance}, got {found_dist}")
    if not exact_path_ok:
        print(f"Path mismatch: expected {' → '.join(expected_path)}, got {'∅' if found_path is None else ' → '.join(found_path)}")
    if not consistent:
        print(f"Inconsistency/path invalid: verified={path_dist}, reported={found_dist}, path_ok={path_ok}")
    return False

def run_all_tests():
    print("BMSSP TESTS WITH EXPECTED RESULTS")

    results = []
    results.append(("GRAPH_SMALL", test_duan(GRAPH_SMALL, "GRAPH_SMALL", EXPECTED_GRAPH_SMALL)))
    results.append(("GRAPH_MEDIUM", test_duan(GRAPH_MEDIUM, "GRAPH_MEDIUM", EXPECTED_GRAPH_MEDIUM)))
    results.append(("GRAPH_BIG", test_duan(GRAPH_BIG, "GRAPH_BIG", EXPECTED_GRAPH_BIG)))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    if passed == total:
        print("\nAll tests passed!")
    else:
        print(f"\n{total - passed} test(s) failed")

    return passed == total

if __name__ == "__main__":
    try:
        ok = run_all_tests()
        exit(0 if ok else 1)
    except Exception as e:
        print(f"\nError during tests: {e}")
        exit(1)
