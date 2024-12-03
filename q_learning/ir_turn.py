import sys
import math
import runner
import rclpy
from irobot_create_msgs.msg import IrIntensityVector
from typing import List


class IrTurnNode(runner.HdxNode):
    def __init__(self, namespace: str = "", ir_too_close=50):
        super().__init__('ir_turn_node', namespace)
        self.ir_too_close = ir_too_close
        self.subscribe_ir(self.ir_callback)
        self.avoid_direction = None

    def ir_clear(self):
        return self.avoid_direction is None

    def is_turning(self):
        return not self.ir_clear()

    def ir_callback(self, msg: IrIntensityVector):
        self.record_first_callback()
        ir_values = [reading.value for reading in msg.readings]
        max_ir = max(ir_values[1:-1]) if self.ir_clear() else max(ir_values)
        if max_ir > self.ir_too_close:
            if self.ir_clear():
                mid = len(ir_values) // 2
                self.avoid_direction = math.pi / 4
                if sum(ir_values[:mid]) > sum(ir_values[-mid:]):
                    self.avoid_direction *= -1.0
            self.publish_twist(runner.turn_twist(self.avoid_direction))
        else:
            self.avoid_direction = None


class IrTurnBot(runner.HdxNode):
    def __init__(self, namespace: str = "", ir_limit=50):
        super().__init__('ir_turn_bot', namespace)
        self.ir_turn = IrTurnNode(namespace, ir_limit)
        self.add_child_nodes(self.ir_turn)
        self.create_timer(0.10, self.timer_callback)

    def timer_callback(self):
        self.record_first_callback()
        if self.ir_turn.ir_clear():
            self.publish_twist(runner.straight_twist(0.5))


if __name__ == '__main__':
    rclpy.init()
    bot = IrTurnBot(f'/{sys.argv[1]}')
    runner.run_recursive_node(bot)
