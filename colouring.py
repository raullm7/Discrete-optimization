#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from random import shuffle
import copy
import random


def reassign_colour(node_count, node, colours, graph):
    neighbours = [col for col in range(node_count) if graph[node, col] == 1]
    neighbor_colour_set = set([colours[n] for n in neighbours])

    colour = 0
    while colour in neighbor_colour_set:
        colour += 1
    colours[node] = colour


def rearrange(colours):
    node_order = []
    groups_of_colours = [[] for i in range(max(colours) + 1)]
    for i in range(len(colours)):
        groups_of_colours[colours[i]].append((i, colours[i]))
    for i in range(len(groups_of_colours)):
        shuffle(groups_of_colours[i])

    rand = random.uniform(0.0, 1.6)
    if  rand >= 1.5:
        groups_of_colours.sort(key = lambda x: len(x))
    elif  rand >= 1.2:
        shuffle(groups_of_colours)
    elif rand >= 0.7:
        groups_of_colours.sort(key = lambda x: -x[0][1])
    else:
        groups_of_colours.sort(key = lambda x: -len(x))
    for i in range(len(groups_of_colours)):
        node_order += groups_of_colours[i]
    return node_order


def greedy(node_count, colours, graph):
    # Initial colouring
    for node in range(node_count):
        reassign_colour(node_count, node, colours, graph)

    # Shuffle the node order and sort
    iter_count = int(1 / float(node_count) * 1000000)
    for i in range(iter_count):
        old_colours = copy.deepcopy(colours)
        node_order = rearrange(colours)
        colours = [0] * node_count
        for node in range(node_count):
            reassign_colour(node_count, node_order[node][0], colours, graph)
        if len(set(old_colours)) < len(set(colours)):
            colours = old_colours

    return colours


def build_graph(node_count, edge_count, edges):
    adjacency_matrix = np.zeros((node_count, node_count))
    for edge in edges:
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1
    return adjacency_matrix


def solve_it(input_data):
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    graph = build_graph(node_count, edge_count, edges)

    colours = [0] * node_count
    colours = greedy(node_count, colours, graph)

    output_data = str(len(set(colours))) + ' 1\n'
    output_data += ' '.join(map(str, colours))

    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
