# Save as perfect_simulation_gallo.py and run with Python 3.8+
import random
from collections import deque

# -------------------------
# Generic Algorithm 1 (paper pseudo-code)
# -------------------------
def algorithm1(m, n, update_fun, seed=None, max_backward=10000):
    """
    Implements Algorithm 1 skeleton from Gallo & Garcia (2010).
    - m, n: integer window indices (m <= n); we'll construct [X]_m^n.
    - update_fun(u, known_suffix) -> symbol or None (None == '?')
      known_suffix should be provided as a string/list of symbols corresponding
      to the most recently known suffix (from older to newer).
    - Seeds the RNG if provided.
    Returns (theta, constructed_list) where constructed_list is list of symbols for indices theta..n.
    """

    if seed is not None:
        random.seed(seed)

    # Initialize U_m ... U_n
    U = {}              # dictionary index -> uniform realization
    X = {}              # constructed symbols (index -> symbol) or None if unknown
    for i in range(m, n+1):
        U[i] = random.random()
        X[i] = None

    B = set(range(m, n+1))   # indices still to construct
    i = m

    # forward attempt using available U
    while i <= n and update_fun(U[i], suffix_from_X(X, i-1)):
        X[i] = update_fun(U[i], suffix_from_X(X, i-1))
        B.discard(i)
        i += 1

    # now backward loop until B is empty
    backward_steps = 0
    while B:
        i -= 1
        if backward_steps > max_backward:
            raise RuntimeError("Exceeded max backward steps (potential non-termination in practice).")
        backward_steps += 1

        B.add(i)
        U[i] = random.random()
        # The paper has an inner loop "while Ui in [#E_epsilon,1[" which corresponds
        # to forced extra moves. For simplicity we skip special epsilon logic and just
        # attempt to set X[i] = F(Ui, empty_suffix). If unknown, we keep expanding backwards.
        sym = update_fun(U[i], [])
        if sym is not None:
            X[i] = sym
            B.discard(i)

            # forward fill as far as possible
            t = min(B) if B else None
            while B and t is not None and update_fun(U[t], suffix_from_X(X, t-1)):
                X[t] = update_fun(U[t], suffix_from_X(X, t-1))
                B.discard(t)
                if B:
                    t = min(B)
                else:
                    t = None
        else:
            # couldn't set X[i] yet; will generate another U[i-1] in next loop iteration
            pass

    theta = min(k for k in X if X[k] is not None)
    # Return theta and the list of symbols from theta..n
    return theta, [X[idx] for idx in range(theta, n+1)]

def suffix_from_X(Xdict, i):
    """
    Return the known suffix up to position i (i is the index of most recent past).
    We'll return as a list from older->newer [X_{i-L}, ..., X_{i}] for the available
    contiguous known block ending at i. If none is known return empty list.
    """
    if i not in Xdict:
        return []
    # gather contiguous known block ending at i (search backward)
    res = deque()
    j = i
    while j in Xdict and Xdict[j] is not None:
        res.appendleft(Xdict[j])
        j -= 1
    return list(res)


# -------------------------
# Example update_fun 1: toy binary kernel with decaying dependence (CFF-like)
# This is only illustrative: it builds a small partition for finite depths.
# -------------------------
def make_binary_cff_update(p_mem_func, alphabet=(0,1), max_depth=10):
    """
    Create an update_fun for a binary kernel defined by p_mem_func(symbol, past_suffix)
    which returns P(symbol | past_suffix) for finite suffixes, and where suffix is a
    list of bits (older->newer).
    We approximate the paper's partition by using cumulative prob on min over completions
    up to depth `max_depth` (practical approximation).
    """

    def update_fun(u, known_suffix):
        # known_suffix: list older->newer
        # Try to decide symbol using the best info we have.
        # We'll attempt increasing lookback depth up to max_depth.
        # For each depth d, build conditional prob for each symbol approximating
        # inf over completions by taking minimum over sample completions (here we
        # just evaluate p_mem_func on padded suffixes).
        for d in range(len(known_suffix), max_depth+1):
            # construct candidate suffix of length d (right-aligned)
            # if known_suffix shorter than d, pad left with a wildcard that p_mem_func must handle
            suffix = known_suffix[-d:] if len(known_suffix) >= d else ['*']*(d - len(known_suffix)) + known_suffix
            # compute probabilities for symbols
            probs = [p_mem_func(a, suffix) for a in alphabet]
            total = sum(probs)
            if total <= 0:
                continue
            # normalize
            cum = 0.0
            for a, p in zip(alphabet, probs):
                cum += p/total
                if u < cum:
                    return a
            # if u >= cum (numerical issues) continue to next d
        # cannot decide with available depth -> return None (paper: '?')
        return None

    return update_fun

# -------------------------
# Example update_fun 2: RCT example (user supplies a dict context->probabilities)
# -------------------------
def make_rct_update(context_tree_probs, alphabet):
    """
    context_tree_probs: dict mapping context tuple (older->newer) to dict symbol->probability.
      contexts are stored with newest symbol at the end, e.g. ('1','0') for suffix "...10".
    update_fun tries to find the shortest suffix (from 0..max_len) present in the tree
    and uses its probs to map u->symbol. Always returns symbol (since RCT leaves cover all pasts).
    """
    max_ctx = max((len(c) for c in context_tree_probs), default=0)

    def update_fun(u, known_suffix):
        # we might not know enough of suffix (known_suffix could be shorter than needed)
        # attempt with what we have (right-aligned)
        for L in range(0, max_ctx+1):
            ctx = tuple(known_suffix[-L:]) if L>0 else tuple()
            if ctx in context_tree_probs:
                probs = context_tree_probs[ctx]
                cum = 0.0
                for a in alphabet:
                    cum += probs.get(a, 0.0)
                    if u < cum:
                        return a
                # if none matched due to rounding, return last alphabet
                return alphabet[-1]
        return None
    return update_fun

# -------------------------
# Toy demo: binary kernel with short finite dependence (user example)
# -------------------------
if __name__ == "__main__":
    # Toy kernel: P(1 | last bit b) = 0.3 if b==0 else 0.8, decays when unknown
    def p_mem(symbol, suffix):
        # suffix is list older->newer, last element is most recent
        if len(suffix) == 0 or suffix[-1] == '*':
            # no info: small bias
            return 0.5 if symbol == 1 else 0.5
        last = suffix[-1]
        if last == 0:
            return 0.3 if symbol == 1 else 0.7
        else:
            return 0.8 if symbol == 1 else 0.2

    update = make_binary_cff_update(p_mem_func=p_mem, alphabet=(0,1), max_depth=3)

    # simulate window m..n = 0..4
    theta, sample = algorithm1(0, 4, update_fun=update, seed=1234, max_backward=2000)
    print("theta:", theta, "sample:", sample)
