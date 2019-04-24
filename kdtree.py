from collections import namedtuple
from operator import itemgetter
from pprint import pformat


class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))


def kdtree(point_list, depth=0):
    if not point_list:
        return None

    k = len(point_list[0])      # assumes all points have the same dimension
    # Select axis based on depth so that axis cycles through all valid values
    axis = depth % k

    # Sort point list by axis and choose median as pivot element
    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2

    # Create node and construct subtrees
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1)
    )


def main():
    """Example usage"""
    point_list = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = kdtree(point_list)
    print(tree)


if __name__ == '__main__':
    main()