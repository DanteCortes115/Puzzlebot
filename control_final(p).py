#!/usr/bin/env python3
"""
controller_node.py — Controlador PID para seguimiento de línea
Compatible con vision_node v2

Cambios v2:
- Maneja "STOP_PRIORITY" en /traffic_light_state:
    Llega cuando vision_node detecta STOP de señal activo
    simultáneo con semáforo verde.  El controlador lo trata
    como detención completa (igual que RED), garantizando que
    STOP de señal siempre gana sobre el semáforo verde.

- Maneja "GIVE_WAY" en /sign_command:
    Comportamiento idéntico a SLOW (reducir velocidad mientras
    la señal sea visible; vision_node manda SLOW_END al salir).

- Cruce horizontal:
    vision_node ya publica error=0 durante CRUCE_RECTO.
    El controlador simplemente sigue el PID normal; no hay
    comando de señal asociado al cruce, así que no requiere
    cambio de FSM.  Se documenta explícitamente para claridad.

- Logging reducido a DEBUG para no saturar la consola.
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String

# =========================================================
# PARAMETERS
# =========================================================

LINEAR_SPEED        = 0.10
LINEAR_SPEED_YELLOW = 0.05
LINEAR_SPEED_CURVE  = 0.055
MAX_ANGULAR         = 2.0

ANGULAR_TURN        = 0.8    # rad/s — ajustar según la pista

ERROR_CENTRADO_PX   = 25.0

# =========================================================
# PID GAINS
# =========================================================

KP = 0.0010
KI = 0.0
KD = 0.0006

KP_CURVE = 0.0045
KI_CURVE = 0.0
KD_CURVE = 0.0008

ALPHA_GAINS     = 0.04
MAX_ERROR_DELTA = 18.0

# =========================================================
# CONTROLLER NODE
# =========================================================

class LineFollowerController(Node):

    def __init__(self):
        super().__init__('line_follower_controller')

        # ── Variables de estado ───────────────────────────
        self.lane_error       = 0.0
        self.traffic_state    = "RED"   # RED | GREEN | YELLOW | STOP_PRIORITY
        self.curve_mode       = False
        self.dash_state       = "SOLIDA"
        self.speed_multiplier = 1.0

        # ── PID ───────────────────────────────────────────
        self.prev_error    = 0.0
        self.integral      = 0.0
        self.prev_time     = time.time()
        self.kp_actual     = KP
        self.kd_actual     = KD
        self.error_suave   = 0.0
        self.linear_actual = LINEAR_SPEED

        # ── FSM de señales ────────────────────────────────
        # Estados: IDLE | TURNING_LEFT | TURNING_RIGHT |
        #          REINCORP | STOPPING | SLOWING
        self.sign_fsm = "IDLE"
        self.turn_dir = None   # "LEFT" | "RIGHT"

        # ── Subscribers ───────────────────────────────────
        self.create_subscription(Float32, '/lane_error',          self.lane_error_cb,   10)
        self.create_subscription(String,  '/traffic_light_state', self.traffic_cb,      10)
        self.create_subscription(String,  '/curve_mode',          self.curve_cb,        10)
        self.create_subscription(String,  '/dash_state',          self.dash_cb,         10)
        self.create_subscription(String,  '/sign_command',        self.sign_command_cb, 10)
        self.create_subscription(Float32, '/speed_multiplier',    self.speed_mult_cb,   10)

        # ── Publisher ─────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Timer ─────────────────────────────────────────
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('PID Line Follower Controller Started (v2)')

    # =====================================================
    # CALLBACKS
    # =====================================================

    def lane_error_cb(self, msg):
        raw   = msg.data
        delta = raw - self.error_suave
        delta = max(min(delta, MAX_ERROR_DELTA), -MAX_ERROR_DELTA)
        self.error_suave += delta
        self.lane_error   = self.error_suave

    def curve_cb(self, msg):
        self.curve_mode = (msg.data == "CURVA")

    def traffic_cb(self, msg):
        """
        Estados posibles desde vision_node v2:
          RED, GREEN, YELLOW, NONE  — semáforo normal
          STOP_PRIORITY             — STOP de señal activo suprimiendo GREEN
        """
        detected = msg.data.upper()

        if detected == "STOP_PRIORITY":
            # STOP de señal ganó al semáforo verde: tratar como detención
            self.traffic_state = "STOP_PRIORITY"

        elif detected == "RED":
            self.traffic_state = "RED"

        elif detected == "GREEN":
            # Solo actualizar a GREEN si no hay un STOP de señal activo
            # (doble seguridad: vision_node ya no debería enviar GREEN
            # cuando hay STOP activo, pero por si acaso)
            if self.sign_fsm != "STOPPING":
                self.traffic_state = "GREEN"

        elif detected == "YELLOW":
            if self.traffic_state == "GREEN":
                self.traffic_state = "YELLOW"

        elif detected == "NONE":
            # Semáforo no visible — si estábamos en STOP_PRIORITY y
            # el sign_fsm ya terminó, volver a NONE
            if self.traffic_state == "STOP_PRIORITY" and self.sign_fsm == "IDLE":
                self.traffic_state = "NONE"
            elif self.traffic_state not in ("RED", "STOP_PRIORITY"):
                self.traffic_state = "NONE"

    def dash_cb(self, msg):
        self.dash_state = msg.data

    def speed_mult_cb(self, msg):
        self.speed_multiplier = float(msg.data)

    def sign_command_cb(self, msg):
        """
        Comandos desde vision_node (timers ya gestionados allá):
          TURN_LEFT / TURN_RIGHT  — iniciar giro
          TURN_END                — fin giro → reincorporación
          REINCORP_END            — fin ventana reincorporación
          STOP / STOP_END         — detención completa 4 s
          AHEAD_ONLY              — solo informativo
          GIVE_WAY                — ceder paso = reducir velocidad
          SLOW / SLOW_END         — zona de obras
          ROUNDABOUT              — solo informativo
        """
        cmd = msg.data
        self.get_logger().info(f"sign_command: {cmd}")

        if cmd == "TURN_LEFT":
            self.sign_fsm = "TURNING_LEFT"
            self.turn_dir = "LEFT"

        elif cmd == "TURN_RIGHT":
            self.sign_fsm = "TURNING_RIGHT"
            self.turn_dir = "RIGHT"

        elif cmd == "TURN_END":
            self.sign_fsm = "REINCORP"

        elif cmd == "REINCORP_END":
            self.sign_fsm = "IDLE"
            self.turn_dir = None
            self.get_logger().info("Reincorporación terminada — modo normal")

        elif cmd == "STOP":
            self.sign_fsm = "STOPPING"
            # Asegurar que traffic_state refleje la detención
            self.traffic_state = "STOP_PRIORITY"

        elif cmd == "STOP_END":
            self.sign_fsm = "IDLE"
            # Limpiar STOP_PRIORITY para que el semáforo retome control
            if self.traffic_state == "STOP_PRIORITY":
                self.traffic_state = "NONE"
            self.get_logger().info("Stop completado — reanudando")

        elif cmd == "AHEAD_ONLY":
            self.get_logger().info("Ahead Only — continuando normal")

        elif cmd in ("GIVE_WAY", "SLOW"):
            # Ambos reducen velocidad; speed_multiplier llega por /speed_multiplier
            self.sign_fsm = "SLOWING"

        elif cmd == "SLOW_END":
            self.sign_fsm = "IDLE"
            self.get_logger().info("Velocidad reducida terminada — normal")

        elif cmd == "ROUNDABOUT":
            self.get_logger().info("Roundabout — continuando normal")

    # =====================================================
    # PID
    # =====================================================

    def compute_pid(self):
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0.0:
            dt = 0.0001

        error = self.lane_error

        kp_target = KP_CURVE if self.curve_mode else KP
        kd_target = KD_CURVE if self.curve_mode else KD
        self.kp_actual = ALPHA_GAINS * kp_target + (1 - ALPHA_GAINS) * self.kp_actual
        self.kd_actual = ALPHA_GAINS * kd_target + (1 - ALPHA_GAINS) * self.kd_actual

        kp = self.kp_actual
        ki = KI_CURVE if self.curve_mode else KI
        kd = self.kd_actual

        p = kp * error
        self.integral += error * dt
        self.integral  = max(min(self.integral, 300), -300)
        i = ki * self.integral
        derivative = (error - self.prev_error) / dt
        d = kd * derivative

        output = p + i + d
        self.prev_error = error
        self.prev_time  = current_time
        return output

    # =====================================================
    # HELPERS
    # =====================================================

    def _debe_detenerse(self) -> bool:
        """
        True si alguna condición exige detención completa.
        Orden de prioridad:
          1. Señal STOP activa (sign_fsm == STOPPING)
          2. STOP_PRIORITY en semáforo (STOP señal suprimió verde)
          3. Semáforo en RED
          4. Línea punteada en PAUSA
        """
        if self.sign_fsm == "STOPPING":
            return True
        if self.traffic_state in ("RED", "STOP_PRIORITY"):
            return True
        if self.dash_state == "PAUSA":
            return True
        return False

    # =====================================================
    # CONTROL LOOP
    # =====================================================

    def control_loop(self):
        cmd = Twist()

        # ── Detención absoluta ────────────────────────────
        if self._debe_detenerse():
            self.cmd_pub.publish(cmd)
            self.get_logger().debug(
                f"DETENIDO — light={self.traffic_state} fsm={self.sign_fsm} "
                f"dash={self.dash_state}"
            )
            return

        # ── PID base ──────────────────────────────────────
        angular_control = -self.compute_pid()

        # ── Velocidad lineal base ─────────────────────────
        if self.sign_fsm in ("TURNING_LEFT", "TURNING_RIGHT"):
            linear_target = LINEAR_SPEED_CURVE
        elif self.curve_mode:
            linear_target = LINEAR_SPEED_CURVE
        else:
            error_abs     = abs(self.lane_error)
            speed_factor  = 1.0 - min(error_abs / 180.0, 0.80)
            linear_target = LINEAR_SPEED * speed_factor

        linear_target *= self.speed_multiplier

        self.linear_actual = (ALPHA_GAINS * linear_target +
                              (1 - ALPHA_GAINS) * self.linear_actual)

        # ── Semáforo amarillo ─────────────────────────────
        if self.traffic_state == "YELLOW":
            cmd.linear.x  = LINEAR_SPEED_YELLOW * (0.5 if self.curve_mode else 1.0)
            cmd.angular.z = angular_control
        else:
            cmd.linear.x  = self.linear_actual
            cmd.angular.z = angular_control

        # ── FSM de señales (sobreescribe si aplica) ───────
        if self.sign_fsm == "TURNING_LEFT":
            cmd.linear.x  = LINEAR_SPEED_CURVE * self.speed_multiplier
            cmd.angular.z = ANGULAR_TURN         # positivo = izquierda

        elif self.sign_fsm == "TURNING_RIGHT":
            cmd.linear.x  = LINEAR_SPEED_CURVE * self.speed_multiplier
            cmd.angular.z = -ANGULAR_TURN        # negativo = derecha

        elif self.sign_fsm == "REINCORP":
            # PID activo — vision_node ya manda error cercano a 0
            # si la línea fue encontrada; salir antes si ya está centrado
            if abs(self.lane_error) <= ERROR_CENTRADO_PX:
                self.sign_fsm = "IDLE"
                self.turn_dir = None
                self.get_logger().info("Reincorporado por centrado de línea")
            # cmd ya tiene PID calculado — no sobreescribir

        # SLOWING: linear_target ya usa speed_multiplier — nada más

        # ── Limitar angular ───────────────────────────────
        cmd.angular.z = max(min(cmd.angular.z, MAX_ANGULAR), -MAX_ANGULAR)

        self.cmd_pub.publish(cmd)

        self.get_logger().debug(
            f'Light={self.traffic_state} | FSM={self.sign_fsm} | '
            f'Dash={self.dash_state} | Curva={self.curve_mode} | '
            f'SpeedX={self.speed_multiplier:.1f} | '
            f'Err={self.lane_error:.1f} | '
            f'V={cmd.linear.x:.3f} | W={cmd.angular.z:.3f}'
        )

    # =====================================================
    # STOP ROBOT
    # =====================================================

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


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

