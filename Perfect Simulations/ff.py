import itertools
import random

class ReferenceContextTree:
    def __init__(self, alphabet, reference_string, lag_function, max_depth=8):
        self.alphabet = alphabet
        self.reference_string = tuple(reference_string)
        self.lag_function = lag_function
        self.max_depth = max_depth
        self.contexts = set()
        self.generate_contexts()

    def has_reference_within_threshold(self, v):
        w = self.reference_string
        l = self.lag_function
        len_w = len(w)
        indices = [j for j in range(len(v) - len_w + 1) if tuple(v[j:j + len_w]) == w]
        if not indices:
            return False, None
        m_w_v = len(v) - 1 - indices[-1]
        max_allowed = m_w_v + len_w + l(m_w_v)
        if len(v) <= max_allowed:
            return True, indices[-1]
        else:
            return False, None

    def generate_contexts(self):
        all_seqs = itertools.product(self.alphabet, repeat=self.max_depth)
        for seq in all_seqs:
            v = list(seq)
            if self.has_reference_within_threshold(v)[0]:
                self.contexts.add(tuple(v))

    def find_context(self, past):
        for l in range(len(past), 0, -1):
            suffix = tuple(past[-l:])
            if suffix in self.contexts:
                return suffix
        # Empty past can also be considered if you allow renewal from empty
        if tuple() in self.contexts:
            return tuple()
        return None

def F_reference(u, past, context_tree_obj, transition_dict, alphabet, marginal_probs=None):
    context = context_tree_obj.find_context(past)
    left = 0.0
    # Use context-dependent transitions if context found
    if context is not None and (context in context_tree_obj.contexts):
        for symbol in alphabet:
            prob = transition_dict.get((context, symbol), 0)
            if left <= u < left + prob:
                return symbol
            left += prob
        return alphabet[-1]  # fallback
    # No context: use marginals
    if marginal_probs is None:
        marginal_probs = {a: 1.0 / len(alphabet) for a in alphabet}
    left = 0.0
    for symbol in alphabet:
        prob = marginal_probs[symbol]
        if left <= u < left + prob:
            return symbol
        left += prob
    return alphabet[-1]

def explicit_perfect_simulation(m, n, F, context_tree_obj, transition_dict, alphabet, marginal_probs=None):
    U = {i: random.uniform(0, 1) for i in range(m, n + 1)}
    X = {}
    B = set(range(m, n + 1))
    i = m
    eta = m

    while i <= n and F(U[i], [X[j] for j in range(m, i)], context_tree_obj, transition_dict, alphabet, marginal_probs) in alphabet and B:
        X[i] = F(U[i], [X[j] for j in range(m, i)], context_tree_obj, transition_dict, alphabet, marginal_probs)
        B.discard(i)
        i += 1
    if not B:
        return eta, [X[j] for j in range(m, n + 1)], X[n]
    eta = i
    while B:
        i -= 1
        if i not in U:
            U[i] = random.uniform(0, 1)
        while F(U[i], [], context_tree_obj, transition_dict, alphabet, marginal_probs) not in alphabet:
            U[i] = random.uniform(0, 1)
        X[i] = F(U[i], [], context_tree_obj, transition_dict, alphabet, marginal_probs)
        B.add(i)
        t = min(B)
        while t <= n and F(U[t], [X[j] for j in range(i, t)], context_tree_obj, transition_dict, alphabet, marginal_probs) in alphabet and B:
            X[t] = F(U[t], [X[j] for j in range(i, t)], context_tree_obj, transition_dict, alphabet, marginal_probs)
            B.discard(t)
            t = min(B) if B else t
        eta = i
    return eta, [X[j] for j in range(eta, n + 1)], X[n]

# --- EXAMPLE USAGE ---
alphabet = [+1, -1]
reference_string = [+1, -1]
lag_func = lambda k: 2   # Context only exists if reference string w seen within 2 steps
max_depth = 6
marginal_probs = {+1: 0.01, -1: 0.99}  # can be nonuniform

tree = ReferenceContextTree(alphabet, reference_string, lag_func, max_depth=max_depth)
# Assign transitions (e.g., uniform for demo)
transition_dict = {}
for context in tree.contexts:
    for s in alphabet:
        transition_dict[(context, s)] = 0.5
# Add transition for empty context (optional, for renewal)
transition_dict[(tuple(), +1)] = 0.5
transition_dict[(tuple(), -1)] = 0.5

# Run the simulation!
m, n = 0, 10
eta, sim_chain, final_value = explicit_perfect_simulation(m, n, F_reference, tree, transition_dict, alphabet, marginal_probs)
print("Regeneration time eta:", eta)
print("Simulated chain:", sim_chain)
print("Final state X_n:", final_value)
