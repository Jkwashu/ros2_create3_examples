import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime
import json
from typing import List, Tuple
import math
import threading
from rclpy.node import Node

class EnhancedPathTracker(Node):
    def __init__(self, namespace=''):
        super().__init__('enhanced_path_tracker')
        self.path_points: List[Tuple[float, float]] = []
        self.timestamps: List[float] = []
        self.start_time = None
        self.lock = threading.Lock()
        self.child_nodes = []  # Added this for recursive node structure
        
        # Configure for 4x4 foot grid
        FOOT_TO_METER = 0.3048
        self.grid_size = (4 * FOOT_TO_METER, 4 * FOOT_TO_METER)
        self.grid_cells = (4, 4)
        
        # Subscribe to odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            f'{namespace}/odom',
            self.odom_callback,
            qos_profile_sensor_data
        )
        
        # Statistics tracking
        self.total_distance = 0.0
        self.last_position = None
        self.episode_count = 0
        self.current_state = None
        
        # Initialize plot
        self.setup_plot()
    
    def add_self_recursive(self, executor):
        """Add this node and all child nodes recursively to the executor"""
        executor.add_node(self)
        for child in self.child_nodes:
            if hasattr(child, 'add_self_recursive'):
                child.add_self_recursive(executor)
            else:
                executor.add_node(child)

    def add_child_nodes(self, *children):
        """Add child nodes to this node"""
        self.child_nodes.extend(children)
    
    def setup_plot(self):
        with self.lock:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.line, = self.ax.plot([], [], 'b-', linewidth=2, label='Robot Path')
            self.setup_grid()
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_xlabel('X Position (feet)', fontsize=12)
            self.ax.set_ylabel('Y Position (feet)', fontsize=12)
            self.ax.set_title('Robot Learning Path\n4x4 Grid (1 foot per square)', fontsize=14, pad=20)
            self.ax.set_aspect('equal')
            self.ax.set_xlim(-0.5, 4.5)
            self.ax.set_ylim(-0.5, 4.5)
            
            # Add grid cell numbers
            for i in range(4):
                for j in range(4):
                    self.ax.text(j + 0.5, i + 0.5, f'{i*4 + j}', 
                               ha='center', va='center', fontsize=10)
            
            self.ax.legend(loc='upper right')
            plt.tight_layout()
    
    def setup_grid(self):
        # Draw grid lines
        for i in range(5):
            self.ax.axvline(x=i, color='gray', linestyle='-', alpha=0.5)
            self.ax.axhline(y=i, color='gray', linestyle='-', alpha=0.5)
    
    def odom_callback(self, msg):
        try:
            METER_TO_FOOT = 3.28084
            x_meters = msg.pose.pose.position.x
            y_meters = msg.pose.pose.position.y
            x_feet = x_meters * METER_TO_FOOT
            y_feet = y_meters * METER_TO_FOOT
            
            current_pos = (x_feet, y_feet)
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.start_time is None:
                self.start_time = current_time
            
            # Update path and timestamps
            self.path_points.append(current_pos)
            self.timestamps.append(current_time - self.start_time)
            
            # Update total distance
            if self.last_position:
                dx = x_meters - self.last_position[0]
                dy = y_meters - self.last_position[1]
                self.total_distance += math.sqrt(dx*dx + dy*dy)
            
            self.last_position = (x_meters, y_meters)
            
            # Update plot
            self.update_plot()
        except Exception as e:
            self.get_logger().error(f'Error in odom_callback: {str(e)}')
    
    def update_plot(self):
        if not self.path_points:
            return
        
        try:
            with self.lock:
                x_coords, y_coords = zip(*self.path_points)
                self.line.set_data(x_coords, y_coords)
                
                # Update start and end points
                if len(self.path_points) > 1:
                    self.ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
                    self.ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='Current')
                
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        except Exception as e:
            self.get_logger().error(f'Error in update_plot: {str(e)}')
    
    def save_to_file(self, prefix="qlearning_path"):
        if not self.path_points:
            print("No path data to save!")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}"
            
            # Save path data to CSV first (this is less likely to fail)
            with open(f"{filename}.csv", 'w') as f:
                f.write("Timestamp,X_Position_Feet,Y_Position_Feet\n")
                for (x, y), t in zip(self.path_points, self.timestamps):
                    f.write(f"{t:.6f},{x:.6f},{y:.6f}\n")
            
            # Then save the plot
            with self.lock:
                plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
            
            print(f"Saved:\n- Path data: {filename}.csv\n- Plot: {filename}.png")
            
        except Exception as e:
            print(f"Error saving files: {str(e)}")
            # Try to at least save the CSV data
            try:
                with open(f"backup_path_data_{timestamp}.csv", 'w') as f:
                    f.write("Timestamp,X_Position_Feet,Y_Position_Feet\n")
                    for (x, y), t in zip(self.path_points, self.timestamps):
                        f.write(f"{t:.6f},{x:.6f},{y:.6f}\n")
                print("Saved backup data to backup_path_data.csv")
            except:
                print("Failed to save backup data")

def create_path_tracker(namespace=''):
    return EnhancedPathTracker(namespace)