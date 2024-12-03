import rclpy
from qlearning import QBot, QParameters, QTable, QNodeTemplate
import runner
import math
import sys
from queue import Queue
import threading

from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist

from irobot_create_msgs.msg import HazardDetectionVector, IrIntensityVector


def forward_turning(forward_velocity, turn_velocity):
    t = Twist()
    t.linear.x = forward_velocity
    t.angular.z = turn_velocity
    return t

class QDemoNode(QNodeTemplate):
    def __init__(self, namespace, max_ir=50):
        super().__init__('q_demo_node', namespace, runner.straight_twist(0.5), runner.turn_twist(math.pi / 4))
        self.max_ir = max_ir
        self.bumps = self.create_subscription(HazardDetectionVector, f"{namespace}/hazard_detection", self.bump_callback, qos_profile_sensor_data)
        self.irs = self.create_subscription(IrIntensityVector, f"{namespace}/ir_intensity", self.ir_callback, qos_profile_sensor_data)

    def num_states(self):
        return 3

    def set_reward(self, state):
        if state == 2:
            return -100
        elif state == 1:
            return -10
        elif self.last_action == 0:
            return 1
        else:
            return 0

    def bump_callback(self, msg):
        bump = runner.find_bump_from(msg.detections)
        if bump is not None:
            self.state = 2

    def ir_callback(self, msg):
        self.record_first_callback()
        if self.state != 2:
            ir_values = [reading.value for reading in msg.readings]
            if max(ir_values) > self.max_ir:
                self.state = 1
            else:
                self.state = 0


if __name__ == '__main__':
    import sys
    from path_tracker import create_path_tracker
    
    rclpy.init()
    
    namespace = f'{sys.argv[1]}' if len(sys.argv) >= 2 else ''
    params = QParameters()
    params.epsilon = 0.05

    # Create nodes
    tracker = create_path_tracker(namespace)
    demo_node = QDemoNode(namespace)
    main_node = QBot(demo_node, params)
    
    # Create executor
    executor = rclpy.executors.MultiThreadedExecutor()
    #executor.add_node(tracker)
    executor.add_node(main_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Save data first
        try:
            tracker.save_to_file()
        except Exception as e:
            print(f"Error saving path data: {e}")
        
        # Clean up
        try:
            main_node.destroy_node()
            tracker.destroy_node()
            plt.close('all')
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"Error during rclpy shutdown: {e}")
