#!/usr/bin/env python3
"""
lidar_node.py — Nodo ROS2 para RPLidar A1
Publica:
  /lidar_obstacle  (std_msgs/String)  → "DANGER" | "CLEAR"
  /lidar_distance  (std_msgs/Float32) → distancia en mm al objeto más cercano (frente 180°)

Parámetros ajustables al inicio del archivo:
  ZONA_PELIGRO   — distancia en mm para publicar DANGER (default: 100 mm = 10 cm)
  OFFSET_ANGULO  — corrección de rotación del LiDAR físico
  PORT / BAUD    — puerto serie
"""

import threading
import time
import numpy as np
import serial

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32

# ── Configuración ──────────────────────────────────────────
PORT            = '/dev/ttyUSB1'
BAUD            = 115200

ZONA_PELIGRO    = 100    # mm — < 10 cm → DANGER
DIST_MIN        = 10     # mm — ignorar lecturas casi nulas
DIST_MAX        = 6000   # mm — ignorar > 6 m

OFFSET_ANGULO   = 24     # offset de rotación física del sensor

# Zona ciega (cable/conector del LiDAR)
IGNORAR_ANG_MIN  = 340
IGNORAR_ANG_MAX  = 360
IGNORAR_ANG_MIN2 = 0
IGNORAR_ANG_MAX2 = 20

# Frente: 180° frontales → 270°-360° y 0°-90°
FRENTE_MIN = 270
FRENTE_MAX = 90

PUBLISH_RATE_HZ = 1    # publicar cada segundo
# ──────────────────────────────────────────────────────────


def aplicar_offset(ang: float) -> float:
    return (ang - OFFSET_ANGULO) % 360


def angulo_en_frente(ang: float) -> bool:
    return ang >= FRENTE_MIN or ang <= FRENTE_MAX


def es_zona_ignorada(ang: float) -> bool:
    return (IGNORAR_ANG_MIN <= ang <= IGNORAR_ANG_MAX) or \
           (IGNORAR_ANG_MIN2 <= ang <= IGNORAR_ANG_MAX2)


class LidarNode(Node):

    def __init__(self):
        super().__init__('lidar_node')

        # ── Publishers ────────────────────────────────────
        self.pub_obstacle = self.create_publisher(String,  '/lidar_obstacle', 10)
        self.pub_distance = self.create_publisher(Float32, '/lidar_distance', 10)

        # ── Estado compartido ─────────────────────────────
        self._lock       = threading.Lock()
        self._scan       = {'angles': np.array([]), 'dists': np.array([])}
        self._running    = True

        # ── Hilo lector del LiDAR ─────────────────────────
        self._reader_thread = threading.Thread(
            target=self._leer_lidar, daemon=True
        )
        self._reader_thread.start()
        self.get_logger().info(f'LiDAR Node iniciado — puerto {PORT}, peligro < {ZONA_PELIGRO} mm')

        # ── Timer de publicación ──────────────────────────
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_loop)

    # =====================================================
    # HILO LECTOR SERIE
    # =====================================================

    def _leer_lidar(self):
        try:
            ser = serial.Serial(PORT, BAUD, timeout=2, rtscts=False, dsrdtr=False)
            ser.dtr = False
            time.sleep(0.1)

            # Secuencia de arranque RPLidar A1
            ser.write(b'\xA5\x40')   # reset
            time.sleep(2)
            ser.reset_input_buffer()
            ser.write(b'\xA5\x50')   # get device info
            time.sleep(0.2)
            ser.read(27)
            ser.reset_input_buffer()
            ser.write(b'\xA5\x20')   # start scan
            time.sleep(0.1)
            ser.read(7)              # leer descriptor de respuesta

            self.get_logger().info('RPLidar: escaneo activo.')

            angles_tmp = []
            dists_tmp  = []

            while self._running:
                raw = ser.read(5)
                if len(raw) < 5:
                    continue

                b0, b1, b2, b3, b4 = raw
                if (b0 & 0x03) not in (0x01, 0x02):
                    ser.reset_input_buffer()
                    continue

                quality  = b0 >> 2
                angle    = ((b1 >> 1) | (b2 << 7)) / 64.0
                distance = (b3 | (b4 << 8)) / 4.0
                is_start = (b0 & 0x01) == 1

                # Fin de barrido completo → publicar
                if is_start and len(angles_tmp) > 10:
                    with self._lock:
                        self._scan['angles'] = np.array(angles_tmp.copy())
                        self._scan['dists']  = np.array(dists_tmp.copy())
                    angles_tmp.clear()
                    dists_tmp.clear()

                if quality > 0 and DIST_MIN < distance < DIST_MAX:
                    ang_corr = aplicar_offset(angle)
                    if not es_zona_ignorada(ang_corr) and angulo_en_frente(ang_corr):
                        angles_tmp.append(ang_corr)
                        dists_tmp.append(distance)

            # Detener motor al salir
            ser.write(b'\xA5\x25')
            ser.dtr = True
            ser.close()

        except serial.SerialException as e:
            self.get_logger().error(f'Error serie: {e}')

    # =====================================================
    # TIMER DE PUBLICACIÓN
    # =====================================================

    def _publish_loop(self):
        with self._lock:
            dists = self._scan['dists'].copy()

        if len(dists) == 0:
            # Sin datos aún — publicar CLEAR para no bloquear el robot
            self.pub_obstacle.publish(String(data='CLEAR'))
            self.pub_distance.publish(Float32(data=float(DIST_MAX)))
            return

        dist_min = float(np.min(dists))

        # ── Publicar distancia mínima ─────────────────────
        self.pub_distance.publish(Float32(data=dist_min))

        # ── Publicar estado de obstáculo ──────────────────
        if dist_min < ZONA_PELIGRO:
            state = 'DANGER'
            self.get_logger().warn(
                f'PELIGRO — objeto a {dist_min:.0f} mm ({dist_min/10:.1f} cm)'
            )
        else:
            state = 'CLEAR'

        self.pub_obstacle.publish(String(data=state))

    # =====================================================
    # DESTRUCTOR
    # =====================================================

    def destroy_node(self):
        self._running = False
        super().destroy_node()


# =========================================================
# MAIN
# =========================================================

def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

