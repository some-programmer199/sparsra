from dataclasses import dataclass, field
from typing import Any, List
from S_ops import S_op

@dataclass
class TreeNode:
    s_op: S_op
    parent: 'TreeNode' = None
    children: List['TreeNode'] = field(default_factory=list)
    state: Any = None # The state of the world after applying this s_op
    visits: int = 0
    value: float = 0.0

def add_child(parent: TreeNode, child_s_op: S_op, state: Any) -> TreeNode:
    child_node = TreeNode(s_op=child_s_op, parent=parent, state=state)
    parent.children.append(child_node)
    return child_node