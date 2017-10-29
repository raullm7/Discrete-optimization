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


def greedy():
    sol = [0] * (len(distances))
    total_distance = 0
    for city_index in range(len(distances)):
        distances_from_city = [0] * len(distances[city_index])
        for i in range(len(distances[city_index])):
            distances_from_city[i] = (i, distances[city_index][i])

        distances_from_city.sort(key = lambda x: x[1])
        for next_city_distance in distances_from_city:
            if next_city_distance[1] == 0:
                continue
            next_city_index = next_city_distance[0]
            if next_city_index not in sol:
                total_distance += next_city_distance[1]
                sol[city_index] = next_city_index
                break
    return sol, total_distance


def swap_cities(city1, city2, solution):
    this_solution = copy.deepcopy(solution)
    temp = this_solution[city1]
    this_solution[city1] = this_solution[city2]
    this_solution[city2] = temp
    return this_solution


def decrement_tabu(tabu_list):
    for i in range(len(tabu_list)):
        for j in range(len(tabu_list)):
            tabu_list[i][j] -= (0 if tabu_list[i][j] <= 0 else 1)
    return tabu_list


def tabu_move(tabu_list, city1, city2):
    tabu_list[city1][city2] += 10
    tabu_list[city2][city1] += 10
    return  tabu_list


def get_best_neighbour(tabu_list, init_solution):
    best_solution = copy.deepcopy(init_solution)
    best_cost = get_cost(init_solution)
    city1 = 0
    city2 = 0
    first_neighbour = True
    current_cost = best_cost
    for i in range(len(best_solution)):
        for j in range(len(best_solution)):
            if i == j:
                continue

            cost_before_swap = current_cost
            current_solution = swap_cities(i, j, init_solution)
            current_cost = get_cost(current_solution)

            if current_cost < best_cost * 0.8:
                # If current solution gives a result a 20% better, relax the tabu constraint
                tabu_list[i][j] = 0

            if (current_cost < best_cost or first_neighbour) and tabu_list[i][j] == 0:
                first_neighbour = False
                city1 = i
                city2 = j
                best_solution = copy.deepcopy(current_solution)
                best_cost = current_cost

    if city1 != 0:
        tabu_list = decrement_tabu(tabu_list)
        tabu_list = tabu_move(tabu_list, city1, city2)

    return best_solution, tabu_list


def get_cost(solution):
    cost = distances[solution[len(solution) - 1], solution[0]]
    for i in range(len(solution) - 1):
        cost += distances[solution[i], solution[i + 1]]
    return cost


def get_cost_to_print(solution):
    cost = distances[0, solution[0]]
    for i in range(len(solution) - 1):
        cost += distances[solution[i], solution[i + 1]]
    cost += distances[solution[len(solution) - 1], 0]
    return cost


def solve_it(input_data):
    global distances
    global node_count
    # Modify this code to run your optimization algorithm

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

    (current_solution, cost) = greedy()

    number_iterations = 1000
    tabu_list = np.zeros((node_count, node_count))

    best_solution = copy.deepcopy(current_solution)
    best_cost = get_cost(best_solution)

    for i in range(number_iterations):
        current_solution, tabu_list = get_best_neighbour(tabu_list, current_solution)
        current_cost = get_cost(current_solution)

        if current_cost < best_cost:
            print('Current best cost: ' + str(current_cost))
            best_solution = copy.deepcopy(current_solution)
            best_cost = current_cost


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

