class ContextTreeNode:
    def __init__(self, context, parent=None):
        self.context = context  # list of +1, -1
        self.parent = parent
        self.children = []
        self.is_context = False  # True if node is a context (leaf of the tree)
    
    def add_child(self, child_node):
        self.children.append(child_node)

def generate_context_tree(alphabet,reference_string, max_depth):
    """
    Recursively generates a context tree for binary state space {+1, -1},
    up to max_depth, with the property that a context is determined by
    the last occurrence of a reference string.
    """

    # Root (empty context)
    root = ContextTreeNode(context=[])
    nodes = [root]
    
    def extend(node, depth):
        if depth == max_depth:
            node.is_context = True
            return
        for symbol in alphabet:
            new_context = node.context + [symbol]
            # Suffix property: context ends only if the reference string appears at the end
            if len(new_context) >= len(reference_string) and new_context[-len(reference_string):] == reference_string:
                child = ContextTreeNode(context=new_context, parent=node)
                child.is_context = True
                node.add_child(child)
                nodes.append(child)
            else:
                child = ContextTreeNode(context=new_context, parent=node)
                node.add_child(child)
                nodes.append(child)
                extend(child, depth + 1)
    
    extend(root, 0)
    return root, nodes



def print_contexts(root):
    # Print all contexts (leaves)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_context:
            print('Context:', node.context)
        stack.extend(node.children)


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
        lag_func = self.lag_function
        len_w = len(w)
        indices = [j for j in range(len(v) - len_w + 1) if tuple(v[j:j + len_w]) == w]
        if not indices:
            return False, None
        m_w_v = len(v) - 1 - indices[-1]
        max_allowed = m_w_v + len_w + lag_func(m_w_v)
        if len(v) <= max_allowed:
            return True, indices[-1]
        else:
             return False, None

    # def generate_contexts(self):
    #     all_seqs = itertools.product(self.alphabet, repeat=self.max_depth)
    #     for seq in all_seqs:
    #         v = list(seq)
    #         if self.has_reference_within_threshold(v)[0]:
    #             self.contexts.add(tuple(v))
    def generate_contexts(self):
        root, nodes = generate_context_tree(self.alphabet, self.reference_string, self.max_depth)
        for node in nodes:
            if node.is_context:
                if self.has_reference_within_threshold(node.context)[0]:
                    self.contexts.add(tuple(node.context))


    def find_context(self, past):
        for past_index in range(len(past), 0, -1):
            suffix = tuple(past[-past_index:])
            if suffix in self.contexts:
                return suffix
        # Empty past can also be considered if you allow renewal from empty
        if tuple() in self.contexts:
            return tuple()
        return None
# # Example usage
# reference_string = [+1, -1]  # Example reference string
# max_depth = 50  # Max recursion depth
# root, nodes = generate_context_tree(reference_string, max_depth)
# print_contexts(root)
