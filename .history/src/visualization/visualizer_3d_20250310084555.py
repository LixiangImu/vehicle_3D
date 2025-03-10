import open3d as o3d
import numpy as np
import copy

class Visualizer3D:
    def __init__(self, config):
        self.config = config
        self.car_model = o3d.io.read_triangle_mesh(config['model']['path'])
        
    def visualize(self, detections, poses):
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=self.config['visualization']['window']['width'],
            height=self.config['visualization']['window']['height']
        )
        
        self._add_coordinate_frame(vis)
        self._add_ground_grid(vis)
        self._add_vehicles(vis, poses)
        self._setup_camera(vis)
        
        vis.run()
        vis.destroy_window() 