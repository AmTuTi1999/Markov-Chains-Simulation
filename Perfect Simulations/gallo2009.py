import itertools

class ContextTreeNode:
    def __init__(self, context, parent=None):
        self.context = context  # list of +1, -1
        self.parent = parent
        self.children = []
        self.is_context = False  # True if node is a context (leaf of the tree)
    
    def add_child(self, child_node):
        self.children.append(child_node)

def generate_context_tree(reference_string, max_depth):
    """
    Recursively generates a context tree for binary state space {+1, -1},
    up to max_depth, with the property that a context is determined by
    the last occurrence of a reference string.
    """
    alphabet = [+1, -1]

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

# Example usage
reference_string = [+1, -1]  # Example reference string
max_depth = 4  # Max recursion depth
root, nodes = generate_context_tree(reference_string, max_depth)
print_contexts(root)
