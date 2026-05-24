#!/usr/bin/env python3


import time


import rclpy


from rclpy.node import Node


from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from std_msgs.msg import String




# =========================================================
# PARAMETERS
# =========================================================


LINEAR_SPEED = 0.12


LINEAR_SPEED_YELLOW = 0.06


MAX_ANGULAR = 2.0


# =========================================================
# PID GAINS
# =========================================================


KP = 0.0025


KI = 0.00001


KD = 0.0025




# =========================================================
# CONTROLLER NODE
# =========================================================


class LineFollowerController(Node):


   def __init__(self):


       super().__init__('line_follower_controller')


       # =====================================================
       # VARIABLES
       # =====================================================


       self.lane_error = 0.0


       self.traffic_state = "RED"


       # =====================================================
       # PID VARIABLES
       # =====================================================


       self.prev_error = 0.0


       self.integral = 0.0


       self.prev_time = time.time()


       # =====================================================
       # SUBSCRIBERS
       # =====================================================


       self.create_subscription(
           Float32,
           '/lane_error',
           self.lane_error_callback,
           10
       )


       self.create_subscription(
           String,
           '/traffic_light_state',
           self.traffic_callback,
           10
       )


       # =====================================================
       # PUBLISHER
       # =====================================================


       self.cmd_pub = self.create_publisher(
           Twist,
           '/cmd_vel',
           10
       )


       # =====================================================
       # TIMER
       # =====================================================


       self.timer = self.create_timer(
           0.05,
           self.control_loop
       )


       self.get_logger().info(
           'PID Line Follower Controller Started'
       )


   # =====================================================
   # CALLBACKS
   # =====================================================


   def lane_error_callback(self, msg):


       self.lane_error = msg.data


   # =====================================================
   # TRAFFIC FSM
   # =====================================================


   def traffic_callback(self, msg):


       detected = msg.data.upper()


       # RED -> bloqueo total
       if detected == "RED":


           self.traffic_state = "RED"


       # GREEN desbloquea
       elif detected == "GREEN":


           self.traffic_state = "GREEN"


       # YELLOW solo después de GREEN
       elif detected == "YELLOW":


           if self.traffic_state == "GREEN":


               self.traffic_state = "YELLOW"


   # =====================================================
   # PID CONTROLLER
   # =====================================================


   def compute_pid(self):


       current_time = time.time()


       dt = current_time - self.prev_time


       if dt <= 0.0:
           dt = 0.0001


       error = self.lane_error


       # =================================================
       # PROPORTIONAL
       # =================================================


       p = KP * error


       # =================================================
       # INTEGRAL
       # =================================================


       self.integral += error * dt


       # Anti windup
       self.integral = max(
           min(self.integral, 300),
           -300
       )


       i = KI * self.integral


       # =================================================
       # DERIVATIVE
       # =================================================


       derivative = (error - self.prev_error) / dt


       d = KD * derivative


       # =================================================
       # PID OUTPUT
       # =================================================


       output = p + i + d


       # =================================================
       # UPDATE
       # =================================================


       self.prev_error = error


       self.prev_time = current_time


       return output


   # =====================================================
   # CONTROL LOOP
   # =====================================================


   def control_loop(self):


       cmd = Twist()


       # =====================================================
       # CURVE SPEED REDUCTION
       # =====================================================


       error_abs = abs(self.lane_error)


       speed_factor = 1.0 - min(
           error_abs / 250.0,
           0.7
       )


       # =====================================================
       # PID ANGULAR CONTROL
       # =====================================================


       angular_control = -self.compute_pid()


       # =====================================================
       # RED LIGHT
       # =====================================================


       if self.traffic_state == "RED":


           cmd.linear.x = 0.0
           cmd.angular.z = 0.0


       # =====================================================
       # YELLOW LIGHT
       # =====================================================


       elif self.traffic_state == "YELLOW":


           cmd.linear.x = (
               LINEAR_SPEED_YELLOW
               * speed_factor
           )


           cmd.angular.z = (
               angular_control
               * speed_factor
           )


       # =====================================================
       # GREEN LIGHT
       # =====================================================


       elif self.traffic_state == "GREEN":


           cmd.linear.x = (
               LINEAR_SPEED
               * speed_factor
           )


           cmd.angular.z = (
               angular_control
               * speed_factor
           )


       # =====================================================
       # DEFAULT
       # =====================================================


       else:


           cmd.linear.x = 0.0


           cmd.angular.z = 0.0


       # =====================================================
       # LIMIT ANGULAR
       # =====================================================


       cmd.angular.z = max(
           min(cmd.angular.z, MAX_ANGULAR),
           -MAX_ANGULAR
       )


       # =====================================================
       # PUBLISH
       # =====================================================


       self.cmd_pub.publish(cmd)


       # =====================================================
       # DEBUG
       # =====================================================


       self.get_logger().info(


           f'Light={self.traffic_state} | '
           f'Error={self.lane_error:.2f} | '
           f'V={cmd.linear.x:.2f} | '
           f'W={cmd.angular.z:.2f}'


       )


   # =====================================================
   # STOP ROBOT
   # =====================================================


   def stop_robot(self):


       cmd = Twist()


       cmd.linear.x = 0.0


       cmd.angular.z = 0.0


       self.cmd_pub.publish(cmd)




# =========================================================
# MAIN
# =========================================================


def main(args=None):


   rclpy.init(args=args)


   node = LineFollowerController()


   try:


       rclpy.spin(node)


   except KeyboardInterrupt:


       pass


   finally:


       node.stop_robot()


       node.destroy_node()


       rclpy.shutdown()




if __name__ == '__main__':


   main()

