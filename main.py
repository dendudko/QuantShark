import networkx
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


# 0 - свободная ячейка: 'honeydew'
# 1 - недоступная ячейка: 'lightcoral'
# 2 - ячейка в зоне покрытия вышки: 'lime'
# 3 - вышка: 'midnightblue'

def save_fig(file_name):
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)


class CityGrid:
    def __init__(self, n, m, radius, tower_cost=0, budget=np.inf, coverage=0.35):
        self.n = n
        self.m = m
        self.coverage = coverage
        self.grid = np.zeros((self.n, self.m))
        self.tower_count = 0
        self.radius = radius
        self.graph = nx.Graph()
        self.paths = []
        self.tower_cost = tower_cost
        self.budget = budget
        self.place_obstructed_blocks()

    def place_obstructed_blocks(self):
        generated_count = 0
        while generated_count < round(self.coverage * self.n * self.m):
            random_i, random_j = random.randint(0, self.n - 1), random.randint(0, self.m - 1)
            if self.grid[random_i, random_j] == 0:
                self.grid[random_i, random_j] = 1
                generated_count += 1

    def place_tower(self, row, col):
        if self.grid[row, col] != 1 and self.grid[row, col] != 3:
            for i in range(max(0, row - self.radius), min(self.n, row + self.radius + 1)):
                for j in range(max(0, col - self.radius), min(self.m, col + self.radius + 1)):
                    if self.grid[i, j] == 0:
                        self.grid[i, j] = 2
            self.grid[row, col] = 3
            self.tower_count += 1
        else:
            print(f'Can`t place tower: ({row}, {col})')

    def place_optimal_towers(self):
        # Первый проход, вычисляем веса, ставим одну вышку
        weights = np.zeros((self.n, self.m))
        all_unobstructed = list(zip(*np.where(self.grid == 0)))
        for u in all_unobstructed:
            for i in range(max(0, u[0] - self.radius), min(self.n, u[0] + self.radius + 1)):
                for j in range(max(0, u[1] - self.radius), min(self.m, u[1] + self.radius + 1)):
                    if self.grid[i, j] == 0:
                        weights[u[0], u[1]] += 1
            # Даем приоритет объектам на расстоянии радиуса от края
            if u[0] == self.radius or self.n - u[0] - 1 == self.radius:
                weights[u[0], u[1]] += self.radius
            if u[1] == self.radius or self.m - u[1] - 1 == self.radius:
                weights[u[0], u[1]] += self.radius
        max_index = np.unravel_index(np.argmax(weights), np.array(weights).shape)
        if self.budget >= self.tower_cost:
            self.place_tower(max_index[0], max_index[1])
            self.budget -= self.tower_cost
        # Следующие проходы, пересчитываем веса для затронутых вышкой объектов, ставим новые вышки
        while self.grid.min() == 0 and self.budget >= self.tower_cost:
            for i in range(max(0, max_index[0] - self.radius * 2), min(self.n, max_index[0] + self.radius * 2 + 1)):
                for j in range(max(0, max_index[1] - self.radius * 2), min(self.m, max_index[1] + self.radius * 2 + 1)):
                    if self.grid[i, j] != 1:
                        weights[i, j] = 0
                        if self.grid[i, j] != 3:
                            for i1 in range(max(0, i - self.radius), min(self.n, i + self.radius + 1)):
                                for j1 in range(max(0, j - self.radius), min(self.m, j + self.radius + 1)):
                                    if self.grid[i1, j1] == 0:
                                        weights[i, j] += 1
            max_index = np.unravel_index(np.argmax(weights), np.array(weights).shape)
            self.place_tower(max_index[0], max_index[1])
            self.budget -= self.tower_cost

        self.build_graph()

    def build_graph(self):
        towers = list(zip(*np.where(self.grid == 3)))
        for i in range(len(towers) - 1):
            for j in range(i + 1, len(towers)):
                # Не совсем понял задание - если пытаться разместить как можно меньше вышек,
                # то большинство находится вне зоны досягаемости других,
                # поэтому будем полагать, что радиус связи вышки с другой вышкой = радиус * 2
                if abs(towers[i][0] - towers[j][0]) <= self.radius * 2 and \
                        abs(towers[i][1] - towers[j][1]) <= self.radius * 2:
                    self.graph.add_edge((towers[i][1], towers[i][0]), (towers[j][1], towers[j][0]))

    def find_shortest_path(self, a, b):
        if len(self.graph) > 1:
            try:
                self.paths.append(nx.bidirectional_dijkstra(self.graph, list(self.graph.nodes)[a],
                                                            list(self.graph.nodes)[b])[1])
            except networkx.NetworkXNoPath:
                print(f'No path between {list(self.graph.nodes)[a]} and {list(self.graph.nodes)[b]}')
            except IndexError:
                print(f'List index {a} or {b} out of range')

    def create_heatmap(self):
        fig, ax = plt.subplots(figsize=(self.m / 3, self.n / 3))
        sns.heatmap(self.grid, square=True, cbar=False, vmin=0, vmax=3, xticklabels=[], yticklabels=[],
                    linewidths=.1, cmap=['honeydew', 'lightcoral', 'lime', 'midnightblue'])
        return fig, ax

    def show_grid(self, file_name):
        self.create_heatmap()
        save_fig(file_name)

    def show_graph(self, file_name):
        fig, ax = self.create_heatmap()
        a = np.array([[edge[0][0] + .5, edge[0][1] + .5] for edge in self.graph.edges])
        b = np.array([[edge[1][0] + .5, edge[1][1] + .5] for edge in self.graph.edges])
        ab_pairs = np.c_[a, b]
        ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)
        ax.plot(*ab_args, c='midnightblue', linewidth='2')
        save_fig(file_name)

    def show_shortest_paths(self, file_name):
        if len(self.paths) > 0:
            fig, ax = self.create_heatmap()
            for path in self.paths:
                arr = []
                for i in range(len(path) - 1):
                    arr.append([path[i][0] + .5, path[i][1] + .5,
                                path[i + 1][0] + .5, path[i + 1][1] + .5])
                ab_pairs = np.c_[arr]
                ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)
                ax.plot(*ab_args, c='midnightblue', linewidth='4', marker='o')
            save_fig(file_name)


n, m = 90, 160
city_grid = CityGrid(n, m, tower_cost=1, budget=170, radius=5)
city_grid.show_grid('1 - city_grid.png')
city_grid.place_optimal_towers()
city_grid.show_graph('2 - tower_links.png')
for i in range(3):
    city_grid.find_shortest_path(random.randint(0, len(city_grid.graph.nodes)),
                                 random.randint(0, len(city_grid.graph.nodes)))
city_grid.show_shortest_paths('3 - shortest_paths.png')
print('Number of towers:', city_grid.tower_count)
