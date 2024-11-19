import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime
import json
from typing import List, Tuple
import math
import threading

class EnhancedPathTracker:
    def __init__(self, namespace=''):
        rclpy.init()
        self.node = rclpy.create_node('enhanced_path_tracker')
        self.path_points: List[Tuple[float, float]] = []
        self.timestamps: List[float] = []
        self.start_time = None
        
        # Configure for 4x4 foot grid (converting to meters)
        FOOT_TO_METER = 0.3048  # 1 foot = 0.3048 meters
        self.grid_size = (4 * FOOT_TO_METER, 4 * FOOT_TO_METER)  # 4x4 feet in meters
        self.grid_cells = (4, 4)  # 4x4 grid
        
        # Subscribe to odometry
        self.odom_sub = self.node.create_subscription(
            Odometry,
            f'{namespace}/odom',
            self.odom_callback,
            qos_profile_sensor_data
        )
        
        # Initialize plot
        plt.style.use('seaborn')
        self.setup_plot()
        
        # Statistics tracking
        self.total_distance = 0.0
        self.last_position = None
        self.episode_count = 0
        self.current_state = None
        
    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Main path line
        self.line, = self.ax.plot([], [], 'b-', linewidth=2, label='Robot Path')
        
        # Grid setup
        self.setup_grid()
        
        # Plot formatting
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_xlabel('X Position (feet)', fontsize=12)
        self.ax.set_ylabel('Y Position (feet)', fontsize=12)
        self.ax.set_title('Robot Learning Path\n4x4 Grid (1 foot per square)', fontsize=14, pad=20)
        
        # Set fixed aspect ratio
        self.ax.set_aspect('equal')
        
        # Set axis limits (in feet)
        self.ax.set_xlim(-0.5, 4.5)
        self.ax.set_ylim(-0.5, 4.5)
        
        # Add grid cell numbers
        for i in range(4):
            for j in range(4):
                self.ax.text(j + 0.5, i + 0.5, f'{i*4 + j}', 
                           ha='center', va='center', fontsize=10)
        
        # Legend
        self.ax.legend(loc='upper right')
        
        plt.tight_layout()
    
    def setup_grid(self):
        """Draw the 4x4 grid overlay"""
        # Convert feet to meters for internal calculations
        FOOT_TO_METER = 0.3048
        
        # Draw vertical grid lines
        for i in range(5):  # 0 to 4 for 4x4 grid
            x = i
            self.ax.axvline(x=x, color='gray', linestyle='-', alpha=0.5)
            
        # Draw horizontal grid lines
        for i in range(5):
            y = i
            self.ax.axhline(y=y, color='gray', linestyle='-', alpha=0.5)
    
    def calculate_statistics(self) -> dict:
        """Calculate various path statistics"""
        if not self.path_points:
            return {}
        
        x_coords, y_coords = zip(*self.path_points)
        
        # Convert meters to feet for statistics
        METER_TO_FOOT = 3.28084
        x_coords_ft = [x * METER_TO_FOOT for x in x_coords]
        y_coords_ft = [y * METER_TO_FOOT for y in y_coords]
        
        stats = {
            "total_distance_feet": self.total_distance * METER_TO_FOOT,
            "duration_seconds": self.timestamps[-1] if self.timestamps else 0,
            "average_speed_fps": (self.total_distance * METER_TO_FOOT) / self.timestamps[-1] if self.timestamps else 0,
            "bounding_box_feet": {
                "min_x": min(x_coords_ft),
                "max_x": max(x_coords_ft),
                "min_y": min(y_coords_ft),
                "max_y": max(y_coords_ft)
            },
            "num_points": len(self.path_points),
            "start_point_feet": (x_coords_ft[0], y_coords_ft[0]),
            "end_point_feet": (x_coords_ft[-1], y_coords_ft[-1]),
            "episodes_completed": self.episode_count
        }
        
        return stats
    
    def odom_callback(self, msg):
        # Extract position and convert to feet for display
        METER_TO_FOOT = 3.28084
        x_meters = msg.pose.pose.position.x
        y_meters = msg.pose.pose.position.y
        x_feet = x_meters * METER_TO_FOOT
        y_feet = y_meters * METER_TO_FOOT
        
        current_pos = (x_feet, y_feet)
        current_time = self.node.get_clock().now().nanoseconds / 1e9
        
        if self.start_time is None:
            self.start_time = current_time
        
        # Update path and timestamps
        self.path_points.append(current_pos)
        self.timestamps.append(current_time - self.start_time)
        
        # Update total distance (keep internal calculations in meters)
        if self.last_position:
            dx = x_meters - self.last_position[0]
            dy = y_meters - self.last_position[1]
            self.total_distance += math.sqrt(dx*dx + dy*dy)
        
        self.last_position = (x_meters, y_meters)
        
        # Update visualization
        self.update_plot()
    
    def update_plot(self):
        if not self.path_points:
            return
            
        x_coords, y_coords = zip(*self.path_points)
        self.line.set_data(x_coords, y_coords)
        
        # Update start and end points
        if len(self.path_points) > 1:
            for artist in self.ax.artists[:]:
                if isinstance(artist, plt.Circle):
                    artist.remove()
            
            self.ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
            self.ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='Current')
        
        # Update plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save_to_file(self, prefix="qlearning_path"):
        if not self.path_points:
            print("No path data to save!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}"
        
        # Save path data to CSV
        with open(f"{filename}.csv", 'w') as f:
            f.write("Timestamp,X_Position_Feet,Y_Position_Feet\n")
            for (x, y), t in zip(self.path_points, self.timestamps):
                f.write(f"{t:.6f},{x:.6f},{y:.6f}\n")
        
        # Save statistics to JSON
        stats = self.calculate_statistics()
        with open(f"{filename}_stats.json", 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Save high-quality plots
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight')
        
        print(f"Saved:\n- Path data: {filename}.csv\n- Statistics: {filename}_stats.json")
        print(f"- Plots: {filename}.png and {filename}.pdf")

def run_tracking_thread(namespace=''):
    tracker = EnhancedPathTracker(namespace)
    try:
        while rclpy.ok():
            rclpy.spin_once(tracker.node)
    except KeyboardInterrupt:
        print("\nSaving path data before exit...")
        tracker.save_to_file()
    finally:
        tracker.node.destroy_node()
        rclpy.shutdown()
        plt.close()

if __name__ == '__main__':
    import sys
    namespace = f'{sys.argv[1]}' if len(sys.argv) >= 2 else ''
    
    # If running standalone
    tracker = EnhancedPathTracker(namespace)
    tracker.run()