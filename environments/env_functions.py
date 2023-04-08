from globals import *
from environments.env_objects import *


def get_dims_from_pic(map_dir, path='maps'):
    with open(f'{path}/{map_dir}') as f:
        lines = f.readlines()
        height = int(re.search(r'\d+', lines[1]).group())
        width = int(re.search(r'\d+', lines[2]).group())
    return height, width


def get_np_from_dot_map(map_dir, path='maps'):
    with open(f'{path}/{map_dir}') as f:
        lines = f.readlines()
        height, width = get_dims_from_pic(map_dir, path)
        img_np = np.zeros((height, width))
        for height_index, line in enumerate(lines[4:]):
            for width_index, curr_str in enumerate(line):
                if curr_str == '.':
                    img_np[height_index, width_index] = 1
        return img_np, (height, width)


def distance_nodes(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


def set_nei(name_1, name_2, nodes_dict):
    if name_1 in nodes_dict and name_2 in nodes_dict and name_1 != name_2:
        node1 = nodes_dict[name_1]
        node2 = nodes_dict[name_2]
        dist = distance_nodes(node1, node2)
        if dist == 1:
            node1.neighbours.append(node2.xy_name)
            node2.neighbours.append(node1.xy_name)


def make_self_neighbour(nodes):
    for node_1 in nodes:
        node_1.neighbours.append(node_1.xy_name)


def build_graph_from_np(img_np, show_map=False):
    # 0 - wall, 1 - free space
    nodes = []
    nodes_dict = {}

    x_size, y_size = img_np.shape
    # CREATE NODES
    for i_x in range(x_size):
        for i_y in range(y_size):
            if img_np[i_x, i_y] == 1:
                node = Node(i_x, i_y)
                # node = Node(i_y, i_x)
                nodes.append(node)
                nodes_dict[node.xy_name] = node

    # CREATE NEIGHBOURS
    # make_neighbours(nodes)

    name_1, name_2 = '', ''
    for i_x in range(x_size):
        for i_y in range(y_size):
            name_2 = f'{i_x}_{i_y}'
            set_nei(name_1, name_2, nodes_dict)
            name_1 = name_2

    print('finished rows')

    for i_y in range(y_size):
        for i_x in range(x_size):
            name_2 = f'{i_x}_{i_y}'
            set_nei(name_1, name_2, nodes_dict)
            name_1 = name_2
    make_self_neighbour(nodes)
    print('finished columns')

    if show_map:
        plt.imshow(img_np, cmap='gray', origin='lower')
        plt.show()
        # plt.pause(1)
        # plt.close()

    return nodes, nodes_dict