#!/usr/bin/env python3
"""
vision_node.py v8 — Jetson CSI + YOLOv8
═══════════════════════════════════════════════════════════════
PISTA REAL (foto):
  • 2 líneas sólidas negras verticales = bordes del carril
  • 1 línea sólida negra central = la que sigue el robot
  • En los CRUCES la línea central se INTERRUMPE y aparecen
    líneas punteadas HORIZONTALES negras a cada lado

ESTRATEGIA v8 — SIN HOUGH:
───────────────────────────────────────────────────────────────
DETECCIÓN DE LÍNEA (centroide por columnas):
  1. ROI inferior → gris → GaussianBlur(3,3) → Otsu invertido
  2. Franja de SCAN_BAND_PX filas al FONDO → perfil de columnas
  3. Dentro de la ventana de búsqueda → centroide ponderado = X
  4. Primera detección: zona fija INIT_ZONE (solo línea central)
     Trackeada: ventana ±WINDOW_HALF alrededor del último X
     Perdida >5 frames: ampliar a ±WINDOW_HALF_LOST
     Perdida >MAX_LOST_FRAMES: reset al setpoint

CRUCE (línea horizontal punteada):
  • Se escanea una franja HORIZONTAL en la parte MEDIA de la ROI
  • Si el perfil de FILAS (px negros por fila) supera CRUCE_RATIO
    durante CRUCE_CONFIRM_N frames → CRUCE detectado
  • Durante CRUCE_DURACION_S: publicar error=0 (robot recto)
  • Al terminar: _tracking=False → vuelve a buscar desde zona init

CURVA (polyfit Y→X sobre últimos N_CURVE_PTS centroides):
  • |slope| > CURVE_SLOPE_THRESH por CURVE_CONFIRM_N → CURVA
  • Vuelve a NORMAL cuando slope pequeño por CURVE_RELEASE_N

YOLO: hilo daemon, cada YOLO_EVERY_N frames, imgsz=320

DETECCIÓN DE SEÑALES — CALIBRACIÓN DE DISTANCIA:
  • SIGN_MAX_DIST_CM es una RAZÓN (SIGN_REAL_HEIGHT_CM / bbox_h),
    NO una distancia física en centímetros.
  • Para calibrar: pon la señal a ~7 cm de la cámara, ejecuta el
    nodo con SIGN_CALIBRATE=True, y anota el valor "ratio" en los
    logs. Ese número es tu SIGN_MAX_DIST_CM definitivo.
  • Fórmula: cuanto MÁS PEQUEÑO el valor, MÁS CERCA debe estar
    la señal para ser detectada (bbox más grande = ratio menor).
═══════════════════════════════════════════════════════════════
"""

from typing import Optional, List, Tuple
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
import cv2
import numpy as np
import threading
import time
import os
from collections import deque
from ultralytics import YOLO

Gst.init(None)

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
ROI_SPLIT    = 0.55      # fracción Y donde empieza la ROI de carril

# ── Setpoint ────────────────────────────────────────────────
# Ajustar: poner robot sobre la línea central y ver X en HUD
SETPOINT_RATIO = 0.50    # fracción del ancho (centro del frame)

# ── Franja de escaneo (fondo de la ROI) ─────────────────────
SCAN_BAND_PX = 25        # filas al fondo de la ROI para el perfil

# ── Zona inicial (fracción) sin tracking ────────────────────
# La línea central aparece cerca del centro; excluir bordes
INIT_ZONE_L = 0.30
INIT_ZONE_R = 0.70

# ── Ventana de tracking ──────────────────────────────────────
WINDOW_HALF_PX      = 75   # px — seguimiento normal
WINDOW_HALF_LOST_PX = 160  # px — perdida >5 frames

# Mínimo de píxeles blancos en ventana para detección válida
MIN_WHITE_PX = 12

# ── Curva ────────────────────────────────────────────────────
N_CURVE_PTS        = 14
CURVE_SLOPE_THRESH = 0.15
CURVE_CONFIRM_N    = 3
CURVE_RELEASE_N    = 8
MAX_LOST_FRAMES    = 22

# ── Suavizado ────────────────────────────────────────────────
ALPHA_SMOOTH = 0.25
BUFFER_SIZE  = 5

# ── Cruce ────────────────────────────────────────────────────
# Se detecta con el perfil de FILAS en zona media de la ROI:
# si alguna fila tiene >CRUCE_ROW_RATIO del ancho en negro → cruce
CRUCE_SCAN_START  = 0.35   # fracción Y de la ROI donde empieza zona cruce
CRUCE_SCAN_END    = 0.65   # fracción Y de la ROI donde termina
CRUCE_ROW_RATIO   = 0.50   # fracción del ancho en negro por fila → cruce
CRUCE_CONFIRM_N   = 8      # frames consecutivos con cruce para confirmar
CRUCE_DURACION_S  = 3.0    # segundos de error=0 tras confirmar cruce

# ── Semáforo ─────────────────────────────────────────────────
MIN_AREA_LIGHT = 300
CONFIRM_FRAMES = 5
RED_CORRECTION = 0.75

# ── YOLO ─────────────────────────────────────────────────────
YOLO_MODEL   = '/home/puzzlebot/models/completo.onnx'
YOLO_EVERY_N = 10
YOLO_CONF    = 0.35

SIGN_NAMES = {
    0: "Ahead Only", 1: "Construction", 2: "Give Way",
    3: "Turn left ahead", 4: "Turn right ahead",
    5: "Roundabout", 6: "Stop",
}
SIGN_REAL_HEIGHT_CM = 30.0
SIGN_CONFIRM_FRAMES = 3      # reducido de 4 → confirma más rápido cuando ya está cerca

# ── CALIBRACIÓN DE DISTANCIA ─────────────────────────────────
# SIGN_MAX_DIST_CM es la razón (SIGN_REAL_HEIGHT_CM / bbox_h_px).
# NO es distancia física en cm — es un umbral de tamaño aparente.
#
# Cuanto más PEQUEÑO, más CERCA debe estar la señal para activarse
# (exige un bbox más grande en pantalla).
#
# CÓMO CALIBRAR:
#   1. Pon SIGN_CALIBRATE = True
#   2. Coloca la señal a ~7 cm de la cámara
#   3. Ejecuta el nodo y anota el valor "ratio" en los logs
#   4. Pon ese valor en SIGN_MAX_DIST_CM y pon SIGN_CALIBRATE = False
#
# Valor inicial conservador: 0.08
# (equivale a exigir bbox_h > 375 px — señal muy cercana y grande)
# Si tu cámara y señal dan ratio ~0.05 a 7 cm, usa 0.05.
# Si el ratio a 7 cm resulta ser ~0.12, usa 0.12.
SIGN_MAX_DIST_CM  = 0.08   # ← AJUSTAR tras calibrar (ver instrucciones arriba)
SIGN_CALIBRATE    = True   # ← poner False una vez calibrado

SIGN_COOLDOWN_S     = 2.0
TURN_DURATION_S     = 2.0
TURN_REINCORP_S     = 2.0
STOP_DURATION_S     = 4.0
SLOW_SPEED_RATIO    = 0.50


# ═══════════════════════════════════════════════════════════════
# NODO
# ═══════════════════════════════════════════════════════════════

class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')
        self.has_display = os.environ.get('DISPLAY') is not None

        # Publishers
        self.lane_pub  = self.create_publisher(Float32, '/lane_error',          10)
        self.light_pub = self.create_publisher(String,  '/traffic_light_state', 10)
        self.curve_pub = self.create_publisher(String,  '/curve_mode',          10)
        self.sign_pub  = self.create_publisher(String,  '/traffic_sign',        10)
        self.dash_pub  = self.create_publisher(String,  '/dash_state',          10)
        self.cmd_pub   = self.create_publisher(String,  '/sign_command',        10)
        self.speed_pub = self.create_publisher(Float32, '/speed_multiplier',    10)

        # Cámara GStreamer
        # sensor-mode=0: full sensor IMX219 (3264x2464) → máximo FOV, sin crop.
        # nvvidconv escala a 1280x720 antes del appsink.
        self.gst_pipeline = Gst.parse_launch(
            "nvarguscamerasrc sensor-mode=0 wbmode=1 ! "
            "video/x-raw(memory:NVMM), format=NV12, width=3264, height=2464, framerate=21/1 ! "
            "nvvidconv ! video/x-raw, format=I420, width=1280, height=720 ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        self.sink = self.gst_pipeline.get_by_name('sink')
        self.gst_pipeline.set_state(Gst.State.PLAYING)
        self.gst_pipeline.get_state(timeout=5 * Gst.SECOND)

        # YOLO (hilo separado)
        self.yolo             = YOLO(YOLO_MODEL, task='detect')
        self.last_sign        = "NONE"
        self.last_sign_bbox_h = 0.0
        self.last_boxes       = []
        self.yolo_frame       = None
        self.yolo_lock        = threading.Lock()
        self.yolo_running     = True
        self.yolo_thread      = threading.Thread(
            target=self._yolo_worker, daemon=True)
        self.yolo_thread.start()

        # Setpoint
        self._setpoint = int(FRAME_WIDTH * SETPOINT_RATIO)

        # Tracker de curva
        self._curve_pts: deque = deque(maxlen=N_CURVE_PTS)

        # Estado carril
        self.modo           = "NORMAL"
        self.lado_curva     = None
        self.lost_frames    = 0
        self.last_x         = float(self._setpoint)
        self._tracking      = False
        self.last_error     = 0.0
        self.error_buffer: List[float] = []
        self._curve_confirm = 0
        self._curve_release = 0

        # Cruce
        self._cruce_frames = 0
        self._en_cruce     = False
        self._cruce_end    = 0.0

        # Semáforo
        self.fsm_state       = "NONE"
        self.candidate       = "NONE"
        self.candidate_count = 0

        # Señales
        self._sign_candidate        = "NONE"
        self._sign_candidate_count  = 0
        self._sign_candidate_bbox_h = 0.0
        self.sign_confirmado        = "NONE"
        self._sign_cooldown_end     = 0.0
        self._sign_action_state     = "IDLE"
        self._sign_action_end       = 0.0
        self._sign_turn_dir         = None
        self._speed_multiplier      = 1.0

        self.frame_n = 0
        self.timer = self.create_timer(0.033, self.process_frame)
        self.get_logger().info(
            f"VisionNode v8 | setpoint={self._setpoint}px "
            f"| sin Hough | centroide por columnas")
        if SIGN_CALIBRATE:
            self.get_logger().warn(
                "⚠ MODO CALIBRACIÓN ACTIVO — coloca la señal a ~7 cm y anota "
                "el valor 'ratio' en los logs. Luego ajusta SIGN_MAX_DIST_CM "
                "y pon SIGN_CALIBRATE=False.")

    # ─────────────────────────────────────────────────────────
    # YOLO WORKER
    # ─────────────────────────────────────────────────────────

    def _yolo_worker(self):
        while self.yolo_running:
            frame = None
            with self.yolo_lock:
                if self.yolo_frame is not None:
                    frame = self.yolo_frame
                    self.yolo_frame = None
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                results = self.yolo.predict(
                    frame, imgsz=320, conf=YOLO_CONF, verbose=False)
                boxes = results[0].boxes
                if len(boxes) > 0:
                    def bh(b):
                        x1,y1,x2,y2 = b.xyxy[0]; return float(y2-y1)
                    best = max(boxes, key=bh)
                    self.last_sign        = SIGN_NAMES.get(int(best.cls[0]), "?")
                    self.last_sign_bbox_h = bh(best)
                    self.last_boxes = [
                        (tuple(map(int, b.xyxy[0])),
                         SIGN_NAMES.get(int(b.cls[0]), "?"),
                         float(b.conf[0]),
                         SIGN_REAL_HEIGHT_CM / bh(b) if bh(b) > 0 else 999)
                        for b in boxes]
                else:
                    self.last_sign        = "NONE"
                    self.last_sign_bbox_h = 0.0
                    self.last_boxes       = []
            except Exception as e:
                self.get_logger().error(f"YOLO: {e}")
                time.sleep(0.05)

    # ─────────────────────────────────────────────────────────
    # FSM SEÑALES
    # ─────────────────────────────────────────────────────────

    def _actualizar_sign_fsm(self):
        now = time.time()
        det = self.last_sign
        dh  = self.last_sign_bbox_h

        if self._sign_action_state == "STOPPING":
            if now >= self._sign_action_end:
                self._sign_action_state = "IDLE"
                self._sign_cooldown_end = now + SIGN_COOLDOWN_S
                self.sign_confirmado    = "NONE"
                self.cmd_pub.publish(String(data="STOP_END"))
            return self.sign_confirmado

        if self._sign_action_state == "TURNING":
            if now >= self._sign_action_end:
                self._sign_action_state = "REINCORP"
                self._sign_action_end   = now + TURN_REINCORP_S
                self.cmd_pub.publish(String(data="TURN_END"))
            return self.sign_confirmado

        if self._sign_action_state == "REINCORP":
            if now >= self._sign_action_end:
                self._sign_action_state = "IDLE"
                self._sign_cooldown_end = now + SIGN_COOLDOWN_S
                self.sign_confirmado    = "NONE"
                self.cmd_pub.publish(String(data="REINCORP_END"))
            return self.sign_confirmado

        if self._sign_action_state == "SLOWING":
            if det == "NONE":
                if not hasattr(self, '_slow_t'):
                    self._slow_t = now
                elif now - self._slow_t > 3.0:
                    self._sign_action_state = "IDLE"
                    self._speed_multiplier  = 1.0
                    self._sign_cooldown_end = now + SIGN_COOLDOWN_S
                    self.sign_confirmado    = "NONE"
                    self.speed_pub.publish(Float32(data=1.0))
                    self.cmd_pub.publish(String(data="SLOW_END"))
                    if hasattr(self, '_slow_t'): del self._slow_t
            else:
                if hasattr(self, '_slow_t'): del self._slow_t
            return self.sign_confirmado

        if now < self._sign_cooldown_end:
            return self.sign_confirmado

        if det != "NONE":
            # ── CALIBRACIÓN / FILTRO DE DISTANCIA ──────────────
            # ratio = SIGN_REAL_HEIGHT_CM / bbox_h_px
            # Cuanto más pequeño el ratio, más cerca está la señal
            # (bbox más grande = señal más cercana a la cámara).
            # SIGN_MAX_DIST_CM es el umbral máximo de ratio permitido.
            ratio = SIGN_REAL_HEIGHT_CM / dh if dh > 0 else 999.0

            if SIGN_CALIBRATE:
                # Log de calibración: anota "ratio" cuando la señal
                # esté a ~7 cm y usa ese valor como SIGN_MAX_DIST_CM.
                self.get_logger().info(
                    f"[CAL] sign={det:15s} bbox_h={dh:6.1f}px  "
                    f"ratio={ratio:.4f}  (SIGN_MAX_DIST_CM={SIGN_MAX_DIST_CM})")

            if ratio > SIGN_MAX_DIST_CM:
                # Señal demasiado lejos (bbox pequeño) → ignorar
                self._sign_candidate        = "NONE"
                self._sign_candidate_count  = 0
                self._sign_candidate_bbox_h = 0.0
                return self.sign_confirmado

            # ── CONFIRMACIÓN ────────────────────────────────────
            if det == self._sign_candidate:
                self._sign_candidate_count  += 1
                self._sign_candidate_bbox_h  = dh
                if self._sign_candidate_count >= SIGN_CONFIRM_FRAMES:
                    self.sign_confirmado       = det
                    self._sign_candidate       = "NONE"
                    self._sign_candidate_count = 0
                    self._ejecutar_accion(det, now)
            else:
                if dh > self._sign_candidate_bbox_h:
                    self._sign_candidate        = det
                    self._sign_candidate_count  = 1
                    self._sign_candidate_bbox_h = dh
        else:
            self._sign_candidate        = "NONE"
            self._sign_candidate_count  = 0
            self._sign_candidate_bbox_h = 0.0

        return self.sign_confirmado

    def _ejecutar_accion(self, sign, now):
        self.get_logger().info(f"→ SEÑAL: {sign}")
        if "Turn" in sign:
            self._sign_turn_dir     = "LEFT" if "left" in sign else "RIGHT"
            self._sign_action_state = "TURNING"
            self._sign_action_end   = now + TURN_DURATION_S
            self.cmd_pub.publish(String(data=f"TURN_{self._sign_turn_dir}"))
        elif sign == "Stop":
            self._sign_action_state = "STOPPING"
            self._sign_action_end   = now + STOP_DURATION_S
            self.cmd_pub.publish(String(data="STOP"))
        elif sign == "Ahead Only":
            self._sign_cooldown_end = now + SIGN_COOLDOWN_S
            self.sign_confirmado    = "NONE"
            self.cmd_pub.publish(String(data="AHEAD_ONLY"))
        elif sign in ("Give Way", "Construction"):
            self._sign_action_state = "SLOWING"
            self._speed_multiplier  = SLOW_SPEED_RATIO
            self.speed_pub.publish(Float32(data=SLOW_SPEED_RATIO))
            self.cmd_pub.publish(
                String(data="GIVE_WAY" if sign == "Give Way" else "SLOW"))
        elif sign == "Roundabout":
            self._sign_cooldown_end = now + SIGN_COOLDOWN_S
            self.sign_confirmado    = "NONE"
            self.cmd_pub.publish(String(data="ROUNDABOUT"))

    def _effective_light(self):
        if self._sign_action_state == "STOPPING" and self.fsm_state == "GREEN":
            return "STOP_PRIORITY"
        return self.fsm_state

    # ─────────────────────────────────────────────────────────
    # LEER FRAME
    # ─────────────────────────────────────────────────────────

    def _leer_frame(self):
        sample = self.sink.emit('try-pull-sample', 100 * Gst.MSECOND)
        if sample is None: return None
        buf  = sample.get_buffer()
        caps = sample.get_caps()
        w    = caps.get_structure(0).get_value('width')
        h    = caps.get_structure(0).get_value('height')
        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok: return None
        yuv  = np.frombuffer(mi.data, dtype=np.uint8).reshape((h * 3 // 2, w)).copy()
        buf.unmap(mi)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        frame[:, :, 2] = (frame[:, :, 2] * RED_CORRECTION).astype(np.uint8)
        return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # ─────────────────────────────────────────────────────────
    # MÁSCARA BINARIA (líneas negras → blancas)
    # ─────────────────────────────────────────────────────────

    def _binary_mask(self, gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        k = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # ─────────────────────────────────────────────────────────
    # CENTROIDE DENTRO DE VENTANA X
    # Perfil de columnas en la franja inferior de la ROI.
    # ─────────────────────────────────────────────────────────

    def _centroide_en_ventana(self, mask: np.ndarray,
                               x_lo: int, x_hi: int) -> Optional[float]:
        h       = mask.shape[0]
        y_start = max(0, h - SCAN_BAND_PX)
        band    = mask[y_start:, x_lo:x_hi]                  # franja inferior
        perfil  = band.sum(axis=0).astype(np.float32) / 255.0  # px por columna

        total = perfil.sum()
        if total < MIN_WHITE_PX:
            return None

        cols = np.arange(len(perfil), dtype=np.float32)
        return float((cols * perfil).sum() / total) + x_lo

    # ─────────────────────────────────────────────────────────
    # DETECCIÓN DE CRUCE
    #
    # Pista real: en el cruce la línea central DESAPARECE y hay
    # líneas punteadas HORIZONTALES.  Se detecta escaneando la
    # zona media de la ROI por filas: si alguna fila tiene
    # >CRUCE_ROW_RATIO del ancho en negro → hay línea horizontal.
    #
    # Para evitar falsos positivos con las líneas sólidas
    # VERTICALES (bordes), se ignoran las columnas extremas
    # (10 % a cada lado) que es donde van los bordes.
    # ─────────────────────────────────────────────────────────

    def _detectar_cruce(self, mask: np.ndarray) -> bool:
        h, w      = mask.shape
        y0        = int(h * CRUCE_SCAN_START)
        y1        = int(h * CRUCE_SCAN_END)
        # Excluir bordes laterales donde están las líneas sólidas
        margen    = int(w * 0.12)
        zona      = mask[y0:y1, margen: w - margen]
        # Píxeles blancos por fila
        por_fila  = zona.sum(axis=1).astype(np.float32) / 255.0
        ancho_util = w - 2 * margen
        umbral    = ancho_util * CRUCE_ROW_RATIO
        return bool(np.any(por_fila > umbral))

    # ─────────────────────────────────────────────────────────
    # DETECT LANE — núcleo principal
    # ─────────────────────────────────────────────────────────

    def detect_lane(self, frame: np.ndarray) -> float:
        h_f, w_f = frame.shape[:2]
        split_y  = int(h_f * ROI_SPLIT)
        roi_bgr  = frame[split_y:, :]
        gray     = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        mask     = self._binary_mask(gray)
        h_roi, w = mask.shape
        now      = time.time()
        setpoint = self._setpoint

        # ── 1. CRUCE ────────────────────────────────────
        if self._detectar_cruce(mask):
            self._cruce_frames += 1
        else:
            self._cruce_frames = max(0, self._cruce_frames - 1)

        if self._en_cruce and now >= self._cruce_end:
            self._en_cruce     = False
            self._cruce_frames = 0
            self._tracking     = False          # volver a buscar desde zona init
            self.last_x        = float(setpoint)
            self._curve_pts.clear()
            self.get_logger().info("→ CRUCE expirado — buscando línea")

        if not self._en_cruce and self._cruce_frames >= CRUCE_CONFIRM_N:
            self._en_cruce  = True
            self._cruce_end = now + CRUCE_DURACION_S
            self._curve_pts.clear()
            self.get_logger().info(f"→ CRUCE: recto {CRUCE_DURACION_S}s")

        if self._en_cruce:
            rem = max(0., self._cruce_end - now)
            # Visualizar zona de cruce detectada
            y0v = int(h_roi * CRUCE_SCAN_START)
            y1v = int(h_roi * CRUCE_SCAN_END)
            cv2.rectangle(roi_bgr, (0, y0v), (w - 1, y1v), (0, 200, 255), 2)
            cv2.putText(roi_bgr, f"CRUCE {rem:.1f}s",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            return 0.0

        # ── 2. VENTANA DE BÚSQUEDA ──────────────────────
        if not self._tracking:
            # Sin ancla aún: zona central fija (excluye bordes sólidos)
            x_lo = int(w * INIT_ZONE_L)
            x_hi = int(w * INIT_ZONE_R)
        else:
            half = (WINDOW_HALF_LOST_PX
                    if self.lost_frames > 5 else WINDOW_HALF_PX)
            x_lo = max(0,     int(self.last_x) - half)
            x_hi = min(w - 1, int(self.last_x) + half)

        # ── 3. CENTROIDE ────────────────────────────────
        cx = self._centroide_en_ventana(mask, x_lo, x_hi)

        # Fila Y representativa del escaneo (para curva)
        y_scan = h_roi - SCAN_BAND_PX // 2

        # ── 4. ACTUALIZAR TRACKER ───────────────────────
        if cx is not None:
            self.lost_frames = 0
            self.last_x      = cx
            self._tracking   = True
            self._curve_pts.append((cx, float(y_scan)))

            cv2.circle(roi_bgr, (int(cx), y_scan), 8, (0, 220, 220), -1)
            cv2.line(roi_bgr, (int(cx), 0), (int(cx), h_roi), (0, 220, 220), 2)
        else:
            self.lost_frames += 1
            if self.lost_frames > MAX_LOST_FRAMES:
                self.last_x      = float(setpoint)
                self._tracking   = False
                self._curve_pts.clear()
                self._resetear_modo()
                self.lost_frames = 0
                self.get_logger().warn("→ Tracker reseteado al setpoint")

        # Dibujar ventana de búsqueda (amarillo)
        y_band_start = max(0, h_roi - SCAN_BAND_PX)
        cv2.rectangle(roi_bgr,
                      (x_lo, y_band_start), (x_hi, h_roi - 1),
                      (200, 200, 0), 1)
        # Setpoint (verde) y posición actual (cian)
        cv2.line(roi_bgr, (setpoint, 0), (setpoint, h_roi), (0, 200, 0), 1)
        cv2.line(roi_bgr,
                 (int(self.last_x), 0), (int(self.last_x), h_roi),
                 (0, 255, 255), 1)

        # ── 5. CURVA (polyfit Y→X) ──────────────────────
        if len(self._curve_pts) >= 4:
            xs_c   = [p[0] for p in self._curve_pts]
            ys_c   = [p[1] for p in self._curve_pts]
            coeffs = np.polyfit(ys_c, xs_c, 1)
            slope  = float(coeffs[0])
            en_curva = abs(slope) > CURVE_SLOPE_THRESH

            if self.modo == "NORMAL":
                if en_curva:
                    self._curve_confirm += 1
                    self._curve_release  = 0
                    if self._curve_confirm >= CURVE_CONFIRM_N:
                        self.lado_curva     = "der" if slope > 0 else "izq"
                        self.modo           = "CURVA"
                        self._curve_confirm = 0
                        self.get_logger().info(
                            f"→ CURVA ({self.lado_curva}) slope={slope:+.3f}")
                else:
                    self._curve_confirm = max(0, self._curve_confirm - 1)
            else:
                if not en_curva:
                    self._curve_release += 1
                    self._curve_confirm  = 0
                    if self._curve_release >= CURVE_RELEASE_N:
                        self.modo           = "NORMAL"
                        self.lado_curva     = None
                        self._curve_release = 0
                        self.get_logger().info("→ NORMAL")
                else:
                    self._curve_release = max(0, self._curve_release - 1)

            cv2.putText(roi_bgr,
                        f"slope={slope:+.3f} C{self._curve_confirm}"
                        f" R{self._curve_release}",
                        (5, h_roi - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 100, 255), 1)

        # ── 6. ERROR ────────────────────────────────────
        return self._suavizar(self.last_x - setpoint)

    # ─────────────────────────────────────────────────────────
    # SUAVIZADO Y RESET
    # ─────────────────────────────────────────────────────────

    def _suavizar(self, error: float) -> float:
        ewma = ALPHA_SMOOTH * self.last_error + (1 - ALPHA_SMOOTH) * error
        self.last_error = ewma
        self.error_buffer.append(ewma)
        if len(self.error_buffer) > BUFFER_SIZE:
            self.error_buffer.pop(0)
        return float(np.mean(self.error_buffer))

    def _resetear_modo(self):
        self.modo           = "NORMAL"
        self.lado_curva     = None
        self.last_error     = 0.0
        self.error_buffer   = []
        self._curve_confirm = 0
        self._curve_release = 0

    # ─────────────────────────────────────────────────────────
    # SEMÁFORO
    # ─────────────────────────────────────────────────────────

    def detect_traffic_light(self, frame: np.ndarray) -> str:
        h, w    = frame.shape[:2]
        split_y = int(h * ROI_SPLIT)
        roi     = frame[0:split_y, :]
        hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        k3      = np.ones((3, 3), np.uint8)

        red1   = cv2.inRange(hsv, (0,   130, 100), (10,  255, 255))
        red2   = cv2.inRange(hsv, (170, 130, 100), (179, 255, 255))
        red    = cv2.morphologyEx(red1 | red2, cv2.MORPH_OPEN, k3)
        yellow = cv2.morphologyEx(
            cv2.inRange(hsv, (20, 160, 150), (30, 255, 255)), cv2.MORPH_OPEN, k3)
        green  = cv2.morphologyEx(
            cv2.inRange(hsv, (40, 100, 80), (85, 255, 255)), cv2.MORPH_OPEN, k3)

        best_color, best_area, best_box = "NONE", 0, None
        for color, msk in [("RED", red), ("YELLOW", yellow), ("GREEN", green)]:
            cnts, _ = cv2.findContours(
                msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area <= MIN_AREA_LIGHT or area <= best_area: continue
                x, y, bw, bh = cv2.boundingRect(c)
                if color == "YELLOW" and (bw / max(bh, 1) > 2.5 or bh < 8): continue
                best_area, best_color, best_box = area, color, (x, y, bw, bh)

        if best_box:
            x, y, bw, bh = best_box
            dc = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255),
                  "GREEN": (0, 255, 0)}.get(best_color, (255, 255, 255))
            cv2.rectangle(roi, (x, y), (x + bw, y + bh), dc, 3)
            cv2.putText(roi, best_color, (x, max(y - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, dc, 2)
        return best_color

    def update_fsm(self, detected: str):
        if detected == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate       = detected
            self.candidate_count = 1
        if self.candidate_count >= CONFIRM_FRAMES:
            self.fsm_state = self.candidate

    # ─────────────────────────────────────────────────────────
    # LOOP PRINCIPAL
    # ─────────────────────────────────────────────────────────

    def process_frame(self):
        frame = self._leer_frame()
        if frame is None: return

        self.frame_n += 1
        if self.frame_n % YOLO_EVERY_N == 0:
            with self.yolo_lock:
                if self.yolo_frame is None:
                    self.yolo_frame = frame.copy()

        sign_activa = self._actualizar_sign_fsm()
        self.sign_pub.publish(String(data=sign_activa))
        self.speed_pub.publish(Float32(data=self._speed_multiplier))

        lane_error = self.detect_lane(frame)
        detected   = self.detect_traffic_light(frame)
        self.update_fsm(detected)
        eff_light  = self._effective_light()

        self.lane_pub.publish(Float32(data=float(lane_error)))
        self.light_pub.publish(String(data=eff_light))
        self.curve_pub.publish(String(data=self.modo))
        self.dash_pub.publish(String(data="SOLIDA"))

        # ── HUD ─────────────────────────────────────────
        h, w    = frame.shape[:2]
        split_y = int(h * ROI_SPLIT)
        cv2.line(frame, (0, split_y), (w, split_y), (0, 255, 255), 2)
        cv2.line(frame, (self._setpoint, split_y), (self._setpoint, h),
                 (0, 200, 0), 1)

        for (x1, y1, x2, y2), lb, conf, dist in self.last_boxes:
            es  = lb == self._sign_candidate
            col = (0, 255, 255) if es else (0, 255, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if es else 1)
            cv2.putText(frame, f"{lb} r={dist:.3f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)

        color_modo = (0, 200, 0) if self.modo == "NORMAL" else (0, 120, 255)
        lado_str   = f"({self.lado_curva})" if self.lado_curva else ""
        slbl = {"IDLE": "", "TURNING": f" [{self._sign_turn_dir}]",
                "REINCORP": " [REINCORP]", "STOPPING": " [STOP]",
                "SLOWING": " [50%]"}.get(self._sign_action_state, "")
        cstr = (f" [{self._sign_candidate}"
                f" {self._sign_candidate_count}/{SIGN_CONFIRM_FRAMES}]"
                if self._sign_candidate != "NONE" else "")
        lc   = (0, 100, 255) if eff_light == "STOP_PRIORITY" else (0, 255, 0)
        crs  = (f" [CRUCE {max(0., self._cruce_end - time.time()):.1f}s]"
                if self._en_cruce else "")
        trk  = "TRK" if self._tracking else "INIT"
        cal  = " [CAL]" if SIGN_CALIBRATE else ""

        cv2.putText(frame, f"LIGHT:{eff_light}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, lc, 2)
        cv2.putText(frame, f"SIGN:{sign_activa}{slbl}{cstr}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2)
        cv2.putText(frame, f"SPEED:{self._speed_multiplier:.0%}",
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
        cv2.putText(frame,
                    f"ERR:{lane_error:+.1f}px X:{self.last_x:.0f}"
                    f" SP:{self._setpoint} LOST:{self.lost_frames} [{trk}]",
                    (10, split_y + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 50, 50), 2)
        cv2.putText(frame, f"MODO:{self.modo} {lado_str}{crs}",
                    (10, split_y + 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.70, color_modo, 2)
        cv2.putText(frame, f"pts:{len(self._curve_pts)}/{N_CURVE_PTS}{cal}",
                    (10, split_y + 78), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (180, 180, 180) if not SIGN_CALIBRATE else (0, 200, 255), 1)
        cv2.putText(frame, f"MAX_RATIO:{SIGN_MAX_DIST_CM:.4f}",
                    (10, split_y + 98), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1)

        if self.has_display:
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()

    # ─────────────────────────────────────────────────────────
    # DESTRUCTOR
    # ─────────────────────────────────────────────────────────

    def destroy_node(self):
        self.yolo_running = False
        if self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=2.0)
        self.gst_pipeline.set_state(Gst.State.NULL)
        if self.has_display:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

