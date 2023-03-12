import numpy as np
import numpy.random as nr
import networkx as nx
import cv2
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import sqlite3
import time
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from sklearn.cluster import KMeans, SpectralClustering


class VoronoiMap:

    def __init__(self, scale=0.5):
        self.scale = scale
        self.img_original = None
        self.img_marked = None
        self.img_thresh = None
        self.marker_colour = None
        self.VoronoiGraph = None
        self.G = None
        self.towns_x = None
        self.towns_y = None
        self.towns_xy = None
        self.towns_nodes = None
        self.no_towns = None
        self.routes = None
        self.route_distances = None
        self.no_agents = None
        self.starting_town = None
        self.calculation_walltime = None



    def load_original_image(self, image_path):
        """
        load in original unmarked image
        this image is used as background in final plot
        """

        self.img_original = cv2.imread(image_path)


    def load_marked_image(self, image_path, safe_colour=np.array([36, 28, 236])):
        """
        load marked image
        this image is the one where the non-blocked "empty" areas are filled in with the safe_colour. 
        """

        self.img_marked = cv2.imread(image_path)
        self.marker_colour = safe_colour


    def _generate_threshed_image(self, scaled_img_marked):
        """
        generate thresheld image from marked image
        """

        mask = cv2.inRange(scaled_img_marked, self.marker_colour, self.marker_colour)
        imask = mask>0
        red = np.zeros_like(scaled_img_marked, np.uint8)
        red[imask] = scaled_img_marked[imask]
    
        img_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
        lower_threshold = 0
        top_threshold = 10
        _, img_thresh = cv2.threshold(img_gray, top_threshold, 255, cv2.THRESH_BINARY)

        return img_thresh


    def generate_voronoi_paths(self):
        """
        generate legal voronoi paths
        legal here means that the voronoi edges do not pass into/through obstacles
        """

        width = int(self.img_original.shape[1] * self.scale)
        height = int(self.img_original.shape[0] * self.scale)
        dim = (width, height)

        scaled_img_original = cv2.resize(self.img_original, dim, interpolation=cv2.INTER_AREA)
        scaled_img_marked = cv2.resize(self.img_marked, dim, interpolation=cv2.INTER_AREA)
        scaled_img_thresh = self._generate_threshed_image(scaled_img_marked)


        Y, X = np.where(scaled_img_thresh == 0)
        points = np.array([X, Y]).T

        vor = Voronoi(points)
        polygon_matrix = scaled_img_thresh == 0

        finite_ridges = []
        for ridge in vor.ridge_vertices:
            if ridge[0] == -1 or ridge[1] == -1:
                pass
            else: 
                finite_ridges.append(ridge)

                
        filtered_ridges = []

        for ridge in finite_ridges:
            v1 = vor.vertices[ridge[0]]
            v2 = vor.vertices[ridge[1]]

            try:

                if (polygon_matrix[int(v1[1])][int(v1[0])] == False) and (polygon_matrix[int(v2[1])][int(v2[0])] == False):
                    filtered_ridges.append(ridge)
                else:
                    pass  
            except:
                pass

        filtered_vertices = []
        nogood_vertices = []
        fucked_vertices = []

        for vertex in vor.vertices:
            x = vertex[0]
            y = vertex[1]

            try:
                if (polygon_matrix[int(y)][int(x)] == False):
                    filtered_vertices.append(vertex)
                else:
                    nogood_vertices.append(vertex)
            except Exception as e:
                fucked_vertices.append(vertex)

        z = np.array(filtered_vertices)
        vor.ridge_vertices = np.array(filtered_ridges)

        self.VoronoiGraph = vor


    def generate_networkx_graph(self):
        """
        create a networkx graph based on the voronoi graph
        """

        def distance(v1, v2):
            return ((v1[1] - v2[1])**2 + (v1[0] - v2[0])**2)**0.5
        
        self.G = nx.Graph()

        # convert voronoi nodes/edges to networkx graph

        for ridge in self.VoronoiGraph.ridge_vertices:
            
            v1 = self.VoronoiGraph.vertices[ridge[0]]
            v2 = self.VoronoiGraph.vertices[ridge[1]]
            # print(tuple(v1.tolist()), '==>', tuple(v2.tolist()), 'distance =', distance(v1, v2))
            self.G.add_edge(tuple(v1.tolist()), tuple(v2.tolist()), weight=distance(v1, v2))


    def load_waypoints(self, waypoints_filename, start_x, start_y):
        """
        load the waypoints that the agents are to pass through
        """

        self.waypoints_filename = waypoints_filename

        with open(waypoints_filename, 'r') as f:
            df = pd.read_csv(f, delimiter=',', )
        
        self.towns_x = df.iloc[:, 0].to_numpy()
        self.towns_x = np.insert(self.towns_x, 0, start_x) * self.scale

        self.towns_y = df.iloc[:, 1].to_numpy()
        self.towns_y = np.insert(self.towns_y, 0, start_y) * self.scale

        self.towns_xy = np.stack([self.towns_x, self.towns_y]).transpose()

        self.no_towns = self.towns_x.shape[0]

        # find node closest to each point
        vertices = np.array(self.G.nodes())
        tree = KDTree(vertices)

        towns_nodes = []

        for town in self.towns_xy:

            dd, ii = tree.query(town, k=1, workers=-1)
            towns_nodes.append(vertices[ii])
        
        self.towns_nodes = towns_nodes

        return self.towns_x, self.towns_y, self.towns_xy, self.towns_nodes


    def generate_distance_matrix(self):
        """
        create pairwise distance matrix between all nodes that are assigned to a waypoint
        """

        from os.path import exists
        filename = 'distance_matrix_' + self.waypoints_filename.split('.')[0] + '.csv'
        file_exists = exists(filename)

        if not file_exists:
            create_dist_start = time.time()
            # get distance between town_nodes via nx
            print('No distance matrix found. Calculating new one.')
            distance_matrix = np.zeros(shape=[len(self.towns_nodes), len(self.towns_nodes)])

            for i in range(len(self.towns_nodes)):
                for j in range(len(self.towns_nodes)):
                    
                    if i == j:
                        continue
                    
                    else:
                        try:
                            distance_matrix[i][j] = nx.shortest_path_length(self.G, tuple(self.towns_nodes[i]), tuple(self.towns_nodes[j]), weight='weight')         
                        except:
                            print('No path from', tuple(self.towns_nodes[i]), '->', tuple(self.towns_nodes[j]))
                            distance_matrix[i][j] = 999999999999 # set huge distance

            distance_matrix = np.round_(distance_matrix).astype(int)
            print(distance_matrix)

            np.savetxt(filename, distance_matrix)
            print(time.time() - create_dist_start, 'seconds to create data matrix')
        else:
            print('Existing distance matrix file found')
            with open(filename) as f:
                distance_matrix = np.genfromtxt(f, delimiter=' ')
                distance_matrix = np.round_(distance_matrix).astype(int)

            print(distance_matrix)
        
        return distance_matrix


    def solve(self, no_agents=1, starting_town=0):
        
        self.no_agents, self.starting_town = no_agents, starting_town

        dist_matrix = self.generate_distance_matrix()

        def create_data_model():
            """Stores the data for the problem."""
            data = {}
            data['dist_matrix'] = dist_matrix
            data['num_vehicles'] = self.no_agents # NO_AGENTS
            data['depot'] = self.starting_town
            
            return data


        def print_solution(data, manager, routing, solution):
            """Prints solution on console."""
            print(f'Objective: {solution.ObjectiveValue()}\n')
            

            max_route_distance = 0
            max_waiting_time = 0
            routes = []
            route_distances = []
            
            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)
                plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
                
                route = []
                route_distance = 0
                route_waiting_time = 0
                
                while not routing.IsEnd(index):
                    plan_output += ' {} -> '.format(manager.IndexToNode(index))
                    route.append(manager.IndexToNode(index))
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id)
                
                routes.append(route)
                route_distances.append(route_distance)
                
                plan_output += '{}\n'.format(manager.IndexToNode(index))
                plan_output += 'Distance of the route: {}m\n'.format(route_distance)
                print(plan_output)
                max_route_distance = max(route_distance, max_route_distance)
                
            print('Maximum of the route distances: {}m'.format(max_route_distance))
            
            return routes, route_distances

        # Creating callback function for travelling distance cost

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            return data['dist_matrix'][from_node][to_node]
        

        print('Cost Type: Distance')
        walltime = time.time()
        cputime = time.process_time()

        # Instantiate the data problem.
        data = create_data_model()

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['dist_matrix']),
                                            data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)


        # Create and register a transit callback.
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


        # Add Distance constraint.
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            5000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            'Distance')


        distance_dimension = routing.GetDimensionOrDie('Distance')
        distance_dimension.SetGlobalSpanCostCoefficient(100)


        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

        # Setting time limit on solver
        search_parameters.time_limit.seconds = 60

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        self_walltime = time.time()-walltime
        self_cputime = time.process_time()-cputime
        self.calculation_walltime = self_walltime

        print('Wall time', self_walltime)
        print('CPU time', self_cputime)

        # Print solution on console.
        if solution:
            self.routes, self.route_distances = print_solution(data, manager, routing, solution)
            return self.routes, self.route_distances
        
        else:
            print('No solution found !')


    def plot_routes(self):
        """plot the route order for each vehicle"""

        # loop through vehicles
        colour_code = 0
        colours = ['blue', 'orange', 'red', 'yellow', 'purple']
        for route in self.routes:
            colour = colours[colour_code]
            route.append(self.starting_town)
            plt.plot(self.towns_x[route], self.towns_y[route], 'o', ms=5, alpha=0.4, c=colour)
            # plt.plot(self.towns_x[self.starting_town], self.towns_y[self.starting_town], 'or', markerfacecolor='w',  ms=0.1, alpha=0.1) #indicate start position with X
            plt.title(f'OR-Tools VRP Solver\n{self.no_towns} Waypoints, {self.no_agents} Robots\nTime taken: {self.calculation_walltime:.3}s')
            colour_code += 1

    

    def plot_astar_routes(self):
        """plot the astar routes for each vehicle"""

        # loop through vehicles
        
        colour_code = 0
        colours = ['blue', 'orange', 'red', 'yellow', 'purple']
        for route in self.routes:
            colour = colours[colour_code]
            route.append(self.starting_town)
            for i in range(len(route)-1):
                # print(route)
                shortest_path = nx.shortest_path(self.G, tuple(self.towns_nodes[route[i]]), tuple(self.towns_nodes[route[i+1]]))
                for j in range(len(shortest_path)-1):
                    plt.plot([shortest_path[j][0], shortest_path[j+1][0]], [shortest_path[j][1], shortest_path[j+1][1]], '-', c=colour)
                    
                    if j % 30 == 0:

                        plt.arrow(shortest_path[j][0], shortest_path[j][1], 
                                shortest_path[j+1][0]-shortest_path[j][0], shortest_path[j+1][1]-shortest_path[j][1],
                                width=1.5, color=colour)
                # plt.text(x=towns_x[route[i]], y=towns_y[route[i]], s=i, fontsize=6)

            plt.title(f'OR-Tools VRP Solver\n{self.no_towns} Waypoints, {self.no_agents} Robots\nTime taken: {self.calculation_walltime:.3}s')
            colour_code += 1


    def plot(self):
        """
        combination of plot_routes and plot_astar_routes
        """

        self.plot_routes()
        self.plot_astar_routes()
        width = int(self.img_original.shape[1] * self.scale)
        height = int(self.img_original.shape[0] * self.scale)
        dim = (width, height)
        scaled_img_original = cv2.resize(self.img_original, dim, interpolation=cv2.INTER_AREA)

        plt.plot(self.towns_x[self.starting_town], self.towns_y[self.starting_town], 'o', markerfacecolor='w',  ms=10, alpha=1, color='black') #indicate start position with X
        plt.text(self.towns_x[self.starting_town]+10, self.towns_y[self.starting_town]+10, 'Start/Dropoff Point', fontsize=10, weight='bold') #indicate start position with X

        plt.imshow(scaled_img_original)
        plt.show()
        plt.close()


    def full_solve(self, original_image_path, marked_image_path, waypoints_path, no_agents, start_x, start_y, plot=True):
        """
        consolidated method for all the above methods
        """

        self.load_original_image(original_image_path)
        self.load_marked_image(marked_image_path)
        self.generate_voronoi_paths()
        self.generate_networkx_graph()

        self.load_waypoints(waypoints_path, start_x=start_x, start_y=start_y)

        self.generate_distance_matrix()
        self.solve(no_agents=no_agents)

        if plot:
            self.plot()

        return self.routes, self.route_distances

    def plot_animation(self):

        width = int(self.img_original.shape[1] * self.scale)
        height = int(self.img_original.shape[0] * self.scale)
        dim = (width, height)
        scaled_img_original = cv2.resize(self.img_original, dim, interpolation=cv2.INTER_AREA)

        # plt.plot(self.towns_x[self.starting_town], self.towns_y[self.starting_town], 'o', markerfacecolor='w',  ms=10, alpha=1, color='black') #indicate start position with X
        # plt.text(self.towns_x[self.starting_town]+10, self.towns_y[self.starting_town]+10, 'Start/Dropoff Point', fontsize=10, weight='bold') #indicate start position with X

        # plt.imshow(scaled_img_original)

        colour_code = 0
        colours = ['blue', 'red', 'green', 'orange', 'purple']

        fig = plt.figure()
        ax = fig.add_subplot(111)

        paths_x = []
        paths_y = []

        for route in self.routes:

            path_x, path_y = [], []

            route.append(self.starting_town)
            for i in range(len(route)-1):
                # print(route)
                shortest_path = nx.shortest_path(self.G, tuple(self.towns_nodes[route[i]]), tuple(self.towns_nodes[route[i+1]]))
                for j in range(len(shortest_path)):
                    path_x.append(shortest_path[j][0])
                    path_y.append(shortest_path[j][1])

            paths_x.append(path_x)
            paths_y.append(path_y)

        ax.imshow(scaled_img_original)
        ax.plot(self.towns_x[self.starting_town], self.towns_y[self.starting_town], 'o', markerfacecolor='w',  ms=8, alpha=1, color='black') #indicate start position with X
        ax.text(self.towns_x[self.starting_town]+5, self.towns_y[self.starting_town]-10, 'Start/Dropoff Point', fontsize=10, weight='bold') #indicate start position with X
        ax.axis('off')

        colour_code = 0
        for route in self.routes:
            colour = colours[colour_code]
            route.append(self.starting_town)
            ax.plot(self.towns_x[route], self.towns_y[route], 'o', ms=5, alpha=0.5, c=colour)
            colour_code += 1

        line1, = ax.plot(paths_x[0], paths_y[0], colours[0], alpha=0.7)
        line2, = ax.plot(paths_x[1], paths_y[1], colours[1], alpha=0.5)
        line3, = ax.plot(paths_x[2], paths_y[2], colours[2], alpha=0.5)
        # line4, = ax.plot(paths_x[3], paths_y[3], colours[3], alpha=0.5)
        

        def updateline(num, paths_x, paths_y, line1, line2, line3):
            # print(data[0][..., :num])
            line1.set_data(paths_x[0][:num], paths_y[0][:num])
            line2.set_data(paths_x[1][:num], paths_y[1][:num])
            line3.set_data(paths_x[2][:num], paths_y[2][:num])
            # line4.set_data(paths_x[3][:num], paths_y[3][:num])
            
            return line1, line2, line3
        
        line_animation = animation.FuncAnimation(
        fig, updateline, interval=10, fargs=(paths_x, paths_y, line1, line2, line3), blit=True, repeat=True)

        line_animation.save('test.gif')
        plt.show()
