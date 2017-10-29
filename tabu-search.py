#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import numpy as np
import copy

Point = namedtuple("Point", ['x', 'y'])
distances = 0
node_count = 0


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def random_solution():
    perm = np.arange(node_count)
    np.random.shuffle(perm)
    return perm


def generate_new_solution(solution):
    new_solution = copy.deepcopy(solution)
    first_city, second_city = np.random.randint(new_solution.size), np.random.randint(new_solution.size)

    exclude = [copy.copy(first_city)]
    exclude.append((len(new_solution) if first_city == 0 else first_city - 1))
    exclude.append((0 if first_city == new_solution.size - 1 else first_city + 1))

    while second_city in exclude:
        second_city = np.random.randint(len(new_solution))

    if second_city < first_city:
        first_city, second_city = second_city, first_city

    new_solution[first_city:second_city] = np.flip(new_solution[first_city:second_city], 0)
    return new_solution, [[solution[first_city - 1], solution[first_city]], [solution[second_city - 1], solution[second_city]]]


def is_tabu(permutation, tabu_list):
    for i in range(len(permutation)):
        second_city = permutation[0] if i == (len(permutation) - 1) else permutation[i + 1]
        for tabu in tabu_list:
            return tabu == [permutation[i], second_city]
    return False


def generate_candidate(current_solution, tabu_list):
    new_solution, edges = [], None
    old_cost = 0
    new_cost = 1

    while not len(new_solution) or (is_tabu(new_solution, tabu_list)):
        old_cost = new_cost
        new_solution, edges = generate_new_solution(current_solution)
        new_cost = get_cost(new_solution)
        if old_cost * 0.9 > new_cost:
            break

    candidate_cost = new_cost
    candidate = copy.copy(new_solution)
    return [candidate, edges, candidate_cost]


def search(tabu_list_size, candidate_list_size, max_iter):
    current_solution = random_solution()
    current_cost = get_cost(current_solution)
    best_solution = copy.deepcopy(current_solution)
    best_solution_cost = current_cost

    tabu_list = [0] * tabu_list_size
    candidates = [0] * candidate_list_size

    for iteration in range(max_iter):
        for i in range(candidate_list_size):
            candidates[i] = generate_candidate(current_solution, tabu_list)

        candidates.sort(key=lambda x: x[2])
        first_candidate = candidates[0]

        best_candidate_solution = first_candidate[0]
        best_candidate_edges = first_candidate[1]
        best_candidate_cost = first_candidate[2]

        if best_candidate_cost < current_cost:
            current_solution = best_candidate_solution
            if best_candidate_cost < best_solution_cost:
                best_solution = best_candidate_solution
                best_solution_cost = best_candidate_cost

            for edge in best_candidate_edges:
                np.insert(tabu_list, 0, edge)

            while len(tabu_list) > tabu_list_size:
                tabu_list = np.delete(tabu_list, 0)
    return best_solution


def get_cost(solution):
    cost = distances[solution[len(solution) - 1], solution[0]]
    for i in range(len(solution) - 1):
        cost += distances[solution[i], solution[i + 1]]
    return cost


def solve_it(input_data):
    global distances
    global node_count
    # parse the input
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = []
    for i in range(1, node_count+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    distances = np.zeros((node_count, node_count))
    for i in range(node_count):
        for j in range(node_count):
            if distances[i][j] == 0:
                d = length(points[i], points[j])
                distances[i][j] = d
                distances[j][i] = d

    max_iter = 1000
    tabu_list_size = round(node_count / 4)
    max_candidates = node_count * 3
    best_solution = search(tabu_list_size, max_candidates, max_iter)
    best_cost = get_cost(best_solution)

    # prepare the solution in the specified output format
    output_data = '%.2f' % best_cost + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, best_solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

