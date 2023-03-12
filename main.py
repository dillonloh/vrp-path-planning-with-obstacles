from VoronoiClass import VoronoiMap
import matplotlib.pyplot as plt


original_image_path = './maps/warehouse.png'
marked_image_path = './maps/warehouse-inverse.png'
waypoints_path = 'test.csv'

vm = VoronoiMap(scale=0.25)
vm.load_original_image(original_image_path)
vm.load_marked_image(marked_image_path)
vm.generate_voronoi_paths()
vm.generate_networkx_graph()

vm.load_waypoints(waypoints_path, start_x=700, start_y=385)

vm.generate_distance_matrix()
vm.solve(no_agents=3)

vm.plot()

## Above can be completely replaced with vm.full_solve()