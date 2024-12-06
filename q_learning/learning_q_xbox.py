import rclpy
from qlearning import QBot, QParameters, QTable, QNodeTemplate
import runner
import math
import sys
from queue import Queue
import threading
from evdev import InputDevice

from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from path_tracker import create_path_tracker

def forward_turning(forward_velocity, turn_velocity):
    t = Twist()
    t.linear.x = forward_velocity
    t.angular.z = turn_velocity
    return t

class QDemoNode(QNodeTemplate):
    def __init__(self, namespace, msg_queue, x_meters, x_squares, y_meters, y_squares):
        super().__init__('learning_q_xbox', namespace, 
                        runner.turn_twist(-math.pi/8),
                        runner.straight_twist(0.5), 
                        runner.turn_twist(math.pi/8))
        self.odometry = self.create_subscription(Odometry, namespace + '/odom', self.odom_callback, qos_profile_sensor_data)        
        self.x_meters = x_meters
        self.y_meters = y_meters
        self.x_squares = x_squares
        self.y_squares = y_squares
        self.msg_queue = msg_queue
        self.last_cmd = None

    def num_squares(self):
        return self.x_squares * self.y_squares

    def num_states(self):
        return self.num_squares() + 1

    def out_of_bounds_state(self):
        return self.num_states() - 1

    def set_reward(self, state):
        if not self.msg_queue.empty():
            self.last_cmd = self.msg_queue.get()
        if state == self.out_of_bounds_state():
            return -100
        elif self.last_cmd == 'A':
            return 10
        elif self.last_cmd == 'B':
            return -10
        else:
            return 0

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        x = int(p.x * self.x_squares / self.x_meters)
        y = int(p.y * self.y_squares / self.y_meters)
        self.state = y * self.x_squares + x
        if not (0 <= self.state < self.num_squares()):
            self.state = self.out_of_bounds_state()

class XBoxReader:
    def __init__(self, msg_queue, incoming):
        self.msg_queue = msg_queue
        self.incoming = incoming

    def loop(self):
        dev = InputDevice('/dev/input/event0')
        for event in dev.read_loop():
            print(f"event: {event}")
            if event.value == 1:
                if event.code == 304:
                    self.msg_queue.put("A")
                elif event.code == 305:
                    self.msg_queue.put("B")
            if not self.incoming.empty():
                print("got message")
                break

class LearningQXboxNode(runner.HdxNode):
    def __init__(self, namespace):
        super().__init__('learning_q_xbox_node', namespace)
        self.from_x = Queue()
        self.to_x = Queue()
        self.xboxer = XBoxReader(self.from_x, self.to_x)
        
        # Create nodes
        self.tracker = create_path_tracker(namespace)
        self.demo_node = QDemoNode(namespace, self.from_x, 
                                 x_meters=1.2192, y_meters=1.2192,  # 4 feet = 1.2192 meters
                                 x_squares=4, y_squares=4)
        
        params = QParameters()
        params.epsilon = 0.05
        self.main_node = QBot(self.demo_node, params)
        
        # Add child nodes
        self.add_child_nodes(self.main_node, self.tracker)
        
    def quit(self):
        print("\nSaving path data before exit...")
        try:
            self.tracker.save_to_file()
        except Exception as e:
            print(f"Error saving path data: {e}")
        
        try:
            self.to_x.put("QUIT")
        except Exception as e:
            print(f"Error sending quit signal: {e}")
        
        super().quit()


def run_recursive_node(recursive_node):
    executor = rclpy.executors.MultiThreadedExecutor()
    recursive_node.add_self_recursive(executor)
    xbox_thread = threading.Thread(target=lambda x: x.loop(), args=(recursive_node.xboxer,))
    xbox_thread.start()
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    user = input("Type anything to exit")
    recursive_node.quit()
    rclpy.shutdown()
    executor_thread.join()
    xbox_thread.join()



if __name__ == '__main__':
    rclpy.init()
    namespace = f'{sys.argv[1]}' if len(sys.argv) >= 2 else ''
    
    bot = LearningQXboxNode(namespace)
    run_recursive_node(bot)
