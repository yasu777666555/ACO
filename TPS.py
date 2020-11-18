#coding:utf-8

import random
import math
import numpy as np

class City:
    city_co = []

    def __init__(self, num_city):
        self.num_city = num_city

    def generate_city(self):
        """Generate City Coordinate
        ex: city_co[0] has [city0's x_coordinate, city0's y_coordinate]
        """
        for i in range(self.num_city):
           self.city_co.append([random.randint(0,500),random.randint(0,500)])

    def cal_distance(self, p_city, d_city):
        distance = math.sqrt(math.pow(p_city[0] - d_city[0], 2) + math.pow(p_city[1] - d_city[1], 2))
        return distance

class ACO():
    def __init__(self, ants, cities, iteration):
       self.num_city = cities.num_city
       self.num_ants = ants
       self.ants = []
       self.cities = cities
       self.alpha = 1
       self.beta = 5
       self.evaporation = 0.5
       self.trails = [0] * cities.num_city
       self.distance = [0] * cities.num_city
       self.cost = [ants]
       self.graph = {}
       self.iteration = iteration
       self.worst = 500 * cities.num_city
       self.best_cost = 700 * cities.num_city
       self.best_route = [0]

       for i in range(cities.num_city):
           self.trails[i] = [0] * cities.num_city

    def generate_ants(self):
        for i in range(self.num_ants):
            self.ants.append(self.Ant(self.num_city))

    def initiate_ants(self):
        """Initiate Ants
        select first visited city
        """
        for ant in self.ants:
            ant.ant_clear()
            ant.set_visited_cities(random.randrange(0, self.num_city),0)

    def start_ant(self):
        self.generate_ants()
        self.initiate_ants()

        for i in range(self.iteration):
            self.move_ant(i)
            self.update_trails(self.ants)
            self.update_best(self.ants)
            self.initiate_ants()

    def move_ant(self, iteration):
        """Move Ants until visiting all cities
        """
        for i in range(self.num_city - 1):
            for ant in self.ants:
               ant.set_visited_cities(self.select_city(iteration, ant, i), i + 1)

    def select_city(self, iteration, ant, current_city):
        best = -1
        next_city = None
        visited_cities = np.array(ant.visited_cities)
        unvisited_cities = np.where(visited_cities == False)
        unvisited_index = unvisited_cities[0]

        if iteration == 0:
           return unvisited_index[random.randrange(0,len(unvisited_index))]

        probability = self.cal_probability(ant, current_city)
        for i in range(len(probability)):
            if best < probability[i]:
                best = probability[i]
                next_city = unvisited_index[i]
            elif best < probability[i] and random.uniform(0.0,1.0) > 0.5:
                best = probability[i]
                next_city = unvisited_index[i]

        return next_city

    def cal_probability(self, ant, current_city):
        pheromone = 0.0
        coefficient_list = []
        unvisited_cities_coefficient = []
        visited_cities = np.array(ant.visited_cities)
        unvisited_cities = np.where(visited_cities == False)
        unvisited_index = unvisited_cities[0]

        for city in unvisited_index:
            coefficient = math.pow(self.trails[ant.trail[current_city]][city], self.alpha) * math.pow(1 / self.cities.cal_distance(self.cities.city_co[ant.trail[current_city]], self.cities.city_co[city]), self.beta)
            coefficient_list.append(coefficient)
            pheromone += coefficient

        for i in range(len(unvisited_index)):
            if coefficient_list[i] == 0:
               unvisited_cities_coefficient.append(0)
            else:
               unvisited_cities_coefficient.append(coefficient_list[i]/pheromone)

        return unvisited_cities_coefficient

    def update_trails(self, ants):
        for n in range(len(self.trails)):
            for m in range(len(self.trails[n])):
                self.trails[n][m] = self.trails[n][m] * self.evaporation
        for ant in ants:
            contribution =  self.worst / ant.visiting_cost(self.cities)
            for i in range(self.num_city - 1):
                #contribution = self.worst / self.cities.cal_distance(self.cities.city_co[ant.trail[i]], self.cities.city_co[ant.trail[i + 1]])
                self.trails[ant.trail[i]][ant.trail[i + 1]] += contribution
            #contribution = self.cities.cal_distance(self.cities.city_co[ant.trail[self.num_city - 1]], self.cities.city_co[ant.trail[0]])
            self.trails[ant.trail[self.num_city - 1]][ant.trail[0]] += contribution

    def update_best(self, ants):
        for ant in ants:
            if ant.visiting_cost(self.cities) < self.best_cost:
                self.best_cost = ant.visiting_cost(self.cities)
                self.best_route = ant.trail

    class Ant:
        def __init__(self, num_city):
            self.num_cities = num_city
            self.visited_cities = [0] * num_city
            self.trail = [0] * num_city

        def set_visited_cities(self, city_index, trail_index):
            self.visited_cities[city_index] = True
            self.trail[trail_index] = city_index

        def clear_visited_cities(self):
            for i in range(self.num_cities):
                self.visited_cities[i] = False

        def visiting_cost(self, cities):
            sum_cost = 0
            for i in range(len(self.trail) - 1):
                sum_cost += cities.cal_distance(cities.city_co[self.trail[i]], cities.city_co[self.trail[i + 1]])
            sum_cost += cities.cal_distance(cities.city_co[self.trail[len(self.trail) - 1]], cities.city_co[self.trail[0]])
            return sum_cost

        def ant_clear(self):
            self.visited_cities = [0] * self.num_cities
            self.trail = [0] * self.num_cities



import matplotlib.pyplot as plt

cities = City(50)
cities.generate_city()
print("cities: "cities.city_co)
print("distance: ",cities.cal_distance(cities.city_co[0], cities.city_co[1]))
print("number of cities:",cities.num_city)

ant = ACO(10,cities,1000)
ant1 = ant.start_ant()
print("best route:", ant.best_route)

data = []

for co in ant.best_route:
    cities.city_co[co]
    data.append(cities.city_co[co])
data.append(cities.city_co[ant.best_route[0]])

print(data)

x, y = zip(*data)
plt.scatter(x, y)
plt.plot(x,y)
plt.show()
