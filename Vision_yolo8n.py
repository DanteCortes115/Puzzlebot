"""
vision_node.py — Jetson CSI + YOLOv8 señales
Fixes aplicados:
  1. _leer_frame usa try-pull-sample con timeout (no bloquea)
  2. _yolo_worker tiene sleep + limpia yolo_frame tras procesar
  3. DISPLAY check antes de imshow
  4. Pipeline GStreamer con emit-signals=true para try-pull-sample
  5. import time añadido
"""

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
from ultralytics import YOLO

Gst.init(None)

# =========================================================
# CONFIGURACIÓN
# =========================================================

FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
DEZOOM        = 1.0

# ── Detección de línea ────────────────────────────────────
BLACK_THRESHOLD        = 80
SCAN_Y_RATIO           = 0.75
CENTER_MIN_RATIO       = 0.25
CENTER_MAX_RATIO       = 0.75
ALPHA_SMOOTH           = 0.45
BUFFER_SIZE            = 3
MIN_GROUP_WIDTH        = 2
MAX_GROUP_WIDTH        = 60
UMBRAL_ENTRADA_CURVA   = 25
FRAMES_CONFIRMA_CURVA  = 2
FRAMES_CONFIRMA_NORMAL = 5
OFFSET_BORDE           = 120
MAX_ERROR_CURVA        = 160
MAX_LOST_FRAMES        = 15
MIN_AREA_LIGHT         = 300
CONFIRM_FRAMES         = 5
ROI_SPLIT              = 0.55

# ── YOLOv8 ───────────────────────────────────────────────
YOLO_MODEL   = '/home/puzzlebot/models/traffic_signs.onnx'
YOLO_EVERY_N = 8
YOLO_CONF    = 0.40

SIGN_NAMES = {
    0: "Roadwork ahead",
    1: "Turn left ahead",
    2: "Turn right ahead",
    3: "Stop",
    4: "Ahead Only",
}

# ── Corrección de color cámara ────────────────────────────
RED_CORRECTION = 0.75


# =========================================================
# NODO
# =========================================================

class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # ── Verificar display disponible ──────────────────
        self.has_display = os.environ.get('DISPLAY') is not None
        if self.has_display:
            self.get_logger().info(f"Display detectado: {os.environ.get('DISPLAY')}")
        else:
            self.get_logger().warn("Sin DISPLAY — imshow desactivado")

        # ── Publishers ────────────────────────────────────
        self.lane_pub  = self.create_publisher(Float32, '/lane_error',          10)
        self.light_pub = self.create_publisher(String,  '/traffic_light_state', 10)
        self.curve_pub = self.create_publisher(String,  '/curve_mode',          10)
        self.sign_pub  = self.create_publisher(String,  '/traffic_sign',        10)

        # ── Cámara CSI via GStreamer ───────────────────────
        # FIX: emit-signals=true necesario para try-pull-sample
        self.gst_pipeline = Gst.parse_launch(
            "nvarguscamerasrc sensor-mode=4 wbmode=1 ! "
            "video/x-raw(memory:NVMM), format=NV12, width=1280, height=720, framerate=60/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=I420 ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        self.sink = self.gst_pipeline.get_by_name('sink')

        ret = self.gst_pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error("ERROR: No se pudo iniciar el pipeline GStreamer")
        else:
            self.get_logger().info("Cámara CSI iniciada")

        # Esperar a que el pipeline llegue a PLAYING antes de leer frames
        state_ret, _state, _pending = self.gst_pipeline.get_state(timeout=5 * Gst.SECOND)
        if state_ret != Gst.StateChangeReturn.SUCCESS:
            self.get_logger().warn("Pipeline tardó en iniciar, continuando...")

        # ── YOLOv8 ───────────────────────────────────────
        self.yolo         = YOLO(YOLO_MODEL, task='detect')
        self.last_sign    = "NONE"
        self.last_boxes   = []
        self.yolo_frame   = None
        self.yolo_lock    = threading.Lock()
        self.yolo_running = True
        self.yolo_thread  = threading.Thread(target=self._yolo_worker, daemon=True)
        self.yolo_thread.start()
        self.get_logger().info("YOLOv8 iniciado")

        # ── Estado seguidor ───────────────────────────────
        self.last_error        = 0.0
        self.error_buffer      = []
        self.lost_frames       = 0
        self.modo              = "NORMAL"
        self.lado_curva        = None
        self.frames_error_alto = 0
        self.error_history     = []
        self.frames_par_visto  = 0
        self.frame_n           = 0

        # ── FSM semáforo ──────────────────────────────────
        self.fsm_state       = "NONE"
        self.candidate       = "NONE"
        self.candidate_count = 0

        self.timer = self.create_timer(0.03, self.process_frame)

    # =====================================================
    # YOLO WORKER (hilo separado)
    # FIX: sleep para no saturar CPU + limpiar frame procesado
    # =====================================================

    def _yolo_worker(self):
        while self.yolo_running:
            # Leer frame disponible
            with self.yolo_lock:
                frame = self.yolo_frame
                if frame is not None:
                    self.yolo_frame = None   # marcar como consumido

            if frame is None:
                time.sleep(0.01)             # evitar busy-wait
                continue

            try:
                results = self.yolo.predict(frame, imgsz=320, conf=YOLO_CONF, verbose=False)
                boxes   = results[0].boxes
                if len(boxes) > 0:
                    best           = max(boxes, key=lambda b: float(b.conf[0]))
                    self.last_sign = SIGN_NAMES.get(int(best.cls[0]), f"cls_{int(best.cls[0])}")
                    self.last_boxes = [(
                        tuple(map(int, box.xyxy[0])),
                        SIGN_NAMES.get(int(box.cls[0]), f"cls_{int(box.cls[0])}"),
                        float(box.conf[0])
                    ) for box in boxes]
                else:
                    self.last_sign  = "NONE"
                    self.last_boxes = []
            except Exception as e:
                self.get_logger().error(f"YOLO error: {e}")
                time.sleep(0.05)

    # =====================================================
    # LEER FRAME DE LA CÁMARA
    # FIX: try-pull-sample con timeout — nunca bloquea
    # =====================================================

    def _leer_frame(self):
        # try-pull-sample retorna None si no hay frame en 100ms
        # a diferencia de pull-sample que bloquea indefinidamente
        sample = self.sink.emit('try-pull-sample', 100 * Gst.MSECOND)
        if sample is None:
            return None

        buf  = sample.get_buffer()
        caps = sample.get_caps()
        w    = caps.get_structure(0).get_value('width')
        h    = caps.get_structure(0).get_value('height')
        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        yuv = np.frombuffer(mi.data, dtype=np.uint8).reshape((h * 3 // 2, w)).copy()
        buf.unmap(mi)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        frame[:, :, 2] = (frame[:, :, 2] * RED_CORRECTION).astype(np.uint8)
        return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # =====================================================
    # CLEAN MASK
    # =====================================================

    def clean_mask(self, mask):
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # =====================================================
    # PREPROCESADO
    # =====================================================

    def _preparar_thresh(self, frame):
        h, w    = frame.shape[:2]
        split_y = int(h * ROI_SPLIT)
        roi     = frame[split_y:h, :]
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thr  = cv2.threshold(blur, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        thr     = self.clean_mask(thr)
        return roi, thr, w

    def _fila_combinada(self, thr):
        h         = thr.shape[0]
        filas     = [int(h * 0.60), int(h * 0.72), int(h * 0.84)]
        combinada = np.zeros(thr.shape[1], dtype=np.uint8)
        for fy in filas:
            combinada = np.bitwise_or(combinada, thr[fy])
        return combinada, filas[1]

    # =====================================================
    # DETECT LANE
    # =====================================================

    def detect_lane(self, frame):
        roi, thr, w = self._preparar_thresh(frame)
        row, scan_y = self._fila_combinada(thr)

        if self.modo == "NORMAL":
            par = self._buscar_par_central(row, w)
            if par is not None:
                left_x, right_x, center_lane = par
                self.lost_frames      = 0
                self.frames_par_visto = 0
                error = float(center_lane - (w // 2))
                if abs(error) >= UMBRAL_ENTRADA_CURVA:
                    self.frames_error_alto += 1
                    if self.frames_error_alto >= FRAMES_CONFIRMA_CURVA:
                        self.lado_curva        = "der" if error > 0 else "izq"
                        self.modo              = "CURVA"
                        self.frames_error_alto = 0
                        self.frames_par_visto  = 0
                        self.error_history     = []
                        self.get_logger().info(f"→ CURVA ({self.lado_curva})")
                else:
                    self.frames_error_alto = 0
                error_final = self._suavizar(error)
                cv2.line(roi, (left_x,      0), (left_x,      roi.shape[0]), (255,   0,   0), 2)
                cv2.line(roi, (right_x,     0), (right_x,     roi.shape[0]), (255,   0,   0), 2)
                cv2.line(roi, (center_lane, 0), (center_lane, roi.shape[0]), (0,     0, 255), 3)
                cv2.circle(roi, (center_lane, scan_y), 8, (0, 255, 0), -1)
                return error_final
            else:
                self.lost_frames += 1
                if self.lost_frames > MAX_LOST_FRAMES:
                    self._resetear()
                return float(self.last_error)

        else:  # CURVA
            par = self._buscar_par_central(row, w, zona=(0.10, 0.90))
            if par is not None:
                left_x, right_x, center_lane = par
                sep = abs(right_x - left_x)
                bien_centrado = (
                    sep > 40 and
                    int(w * 0.20) < center_lane < int(w * 0.80)
                )
                if bien_centrado:
                    self.frames_par_visto += 1
                    if self.frames_par_visto >= FRAMES_CONFIRMA_NORMAL:
                        self.modo              = "NORMAL"
                        self.lado_curva        = None
                        self.frames_par_visto  = 0
                        self.frames_error_alto = 0
                        self.lost_frames       = 0
                        self.get_logger().info("→ NORMAL (reincorporado)")
                        error = float(center_lane - (w // 2))
                        return self._suavizar(error)
                else:
                    self.frames_par_visto = 0
            else:
                self.frames_par_visto = 0

            error_borde = self._seguir_borde(row, w, roi, scan_y)
            if error_borde is not None:
                self.lost_frames = 0
                return self._suavizar(error_borde)
            else:
                self.lost_frames += 1
                if self.lost_frames > MAX_LOST_FRAMES:
                    self._resetear()
                return float(self.last_error)

    # =====================================================
    # BUSCAR PAR CENTRAL
    # =====================================================

    def _buscar_par_central(self, row, w, zona=None):
        if zona is None:
            zona = (CENTER_MIN_RATIO, CENTER_MAX_RATIO)
        c_min   = int(w * zona[0])
        c_max   = int(w * zona[1])
        px      = np.where(row[c_min:c_max] > 0)[0]
        if len(px) < 2:
            return None
        px      = px + c_min
        grupos  = self._filtrar_grupos(self._agrupar(px))
        centros = [int(np.mean(g)) for g in grupos]
        n       = len(centros)
        if n == 0:    return None
        if n == 1:    cx = centros[0]; return (cx, cx, cx)
        elif n == 2:  left, right = centros[0], centros[1]
        elif n == 3:
            cf = w // 2
            pares = [(centros[0], centros[1]), (centros[1], centros[2])]
            left, right = min(pares, key=lambda p: abs((p[0]+p[1])//2 - cf))
        else:
            mid = len(centros) // 2
            left, right = centros[mid-1], centros[mid]
        if abs(right - left) < 20:
            return None
        return (left, right, (left + right) // 2)

    # =====================================================
    # SEGUIR BORDE EXTERIOR
    # =====================================================

    def _seguir_borde(self, row, w, roi, scan_y):
        c_min   = int(w * 0.03)
        c_max   = int(w * 0.97)
        px      = np.where(row[c_min:c_max] > 0)[0]
        if len(px) < 2:
            return None
        px      = px + c_min
        grupos  = self._filtrar_grupos(self._agrupar(px))
        centros = [int(np.mean(g)) for g in grupos]
        if not centros:
            return None
        cf = w // 2
        if self.lado_curva == "der":
            borde_x  = max(centros)
            objetivo = borde_x - OFFSET_BORDE
        else:
            borde_x  = min(centros)
            objetivo = borde_x + OFFSET_BORDE
        objetivo = int(np.clip(objetivo, 0, w - 1))
        error    = float(np.clip(float(objetivo - cf), -MAX_ERROR_CURVA, MAX_ERROR_CURVA))
        cv2.line(roi, (borde_x,  0), (borde_x,  roi.shape[0]), (0, 100, 255), 3)
        cv2.line(roi, (objetivo, 0), (objetivo, roi.shape[0]), (0, 255, 255), 2)
        cv2.circle(roi, (objetivo, scan_y), 8, (0, 200, 255), -1)
        return error

    # =====================================================
    # SUAVIZADO
    # =====================================================

    def _suavizar(self, error):
        ewma = ALPHA_SMOOTH * self.last_error + (1 - ALPHA_SMOOTH) * error
        self.last_error = ewma
        self.error_buffer.append(ewma)
        if len(self.error_buffer) > BUFFER_SIZE:
            self.error_buffer.pop(0)
        return float(np.mean(self.error_buffer))

    def _resetear(self):
        self.last_error        = 0.0
        self.error_buffer      = []
        self.lost_frames       = 0
        self.modo              = "NORMAL"
        self.lado_curva        = None
        self.frames_par_visto  = 0
        self.frames_error_alto = 0

    # =====================================================
    # AGRUPAR + FILTRAR
    # =====================================================

    def _agrupar(self, pixels, gap=15):
        if len(pixels) == 0:
            return []
        grupos, actual = [], [pixels[0]]
        for px in pixels[1:]:
            if px - actual[-1] <= gap:
                actual.append(px)
            else:
                grupos.append(np.array(actual))
                actual = [px]
        grupos.append(np.array(actual))
        return grupos

    def _filtrar_grupos(self, grupos):
        return [g for g in grupos
                if MIN_GROUP_WIDTH <= int(g[-1]) - int(g[0]) + 1 <= MAX_GROUP_WIDTH]

    # =====================================================
    # TRAFFIC LIGHT
    # =====================================================

    def detect_traffic_light(self, frame):
        h, w    = frame.shape[:2]
        split_y = int(h * ROI_SPLIT)
        roi     = frame[0:split_y, :]
        hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red1    = cv2.inRange(hsv, np.array([0,   120,  90]), np.array([10,  255, 255]))
        red2    = cv2.inRange(hsv, np.array([170, 120,  90]), np.array([179, 255, 255]))
        red     = self.clean_mask(red1 | red2)
        yellow  = self.clean_mask(cv2.inRange(hsv, np.array([18, 120, 120]), np.array([35, 255, 255])))
        green   = self.clean_mask(cv2.inRange(hsv, np.array([45,  80,  80]), np.array([85, 255, 255])))
        masks   = {"RED": red, "YELLOW": yellow, "GREEN": green}
        best_color, best_area, best_box = "NONE", 0, None
        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > MIN_AREA_LIGHT and area > best_area:
                    best_area, best_color, best_box = area, color, cv2.boundingRect(c)
        if best_box is not None:
            x, y, bw, bh = best_box
            cv2.rectangle(roi, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
            cv2.putText(roi, best_color, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return best_color

    def update_fsm(self, detected):
        if detected == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate       = detected
            self.candidate_count = 1
        if self.candidate_count >= CONFIRM_FRAMES:
            self.fsm_state = self.candidate

    # =====================================================
    # LOOP PRINCIPAL
    # =====================================================

    def process_frame(self):
        frame = self._leer_frame()
        if frame is None:
            # Sin frame disponible — no bloquear, simplemente saltar
            return

        # Mandar frame al hilo YOLO cada N frames
        self.frame_n += 1
        if self.frame_n % YOLO_EVERY_N == 0:
            with self.yolo_lock:
                # Solo encolar si el worker ya consumió el anterior
                if self.yolo_frame is None:
                    self.yolo_frame = frame.copy()
            self.sign_pub.publish(String(data=self.last_sign))

        lane_error = self.detect_lane(frame)
        detected   = self.detect_traffic_light(frame)
        self.update_fsm(detected)

        self.lane_pub.publish(Float32(data=float(lane_error)))
        self.light_pub.publish(String(data=self.fsm_state))
        self.curve_pub.publish(String(data=self.modo))

        # ── HUD ───────────────────────────────────────────
        h, w    = frame.shape[:2]
        split_y = int(h * ROI_SPLIT)
        cv2.line(frame, (0, split_y), (w, split_y), (0, 255, 255), 2)
        if self.modo == "NORMAL":
            zm = int(w * CENTER_MIN_RATIO)
            zx = int(w * CENTER_MAX_RATIO)
            cv2.line(frame, (zm, split_y), (zm, h), (0, 255, 0), 1)
            cv2.line(frame, (zx, split_y), (zx, h), (0, 255, 0), 1)

        # Dibujar cajas YOLO
        for (x1, y1, x2, y2), lb, c in self.last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
            cv2.putText(frame, f"{lb} {c:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 2)

        color_modo = (0, 200, 0) if self.modo == "NORMAL" else (0, 120, 255)
        lado_str   = f"({self.lado_curva})" if self.lado_curva else ""

        cv2.putText(frame, f"LIGHT: {self.fsm_state}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"SIGN:  {self.last_sign}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 128), 2)
        cv2.putText(frame, f"ERROR: {lane_error:+.1f}px",
                    (10, split_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
        cv2.putText(frame, f"MODO: {self.modo} {lado_str}",
                    (10, split_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color_modo, 2)
        cv2.putText(frame, f"LOST: {self.lost_frames}",
                    (10, split_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1)

        # FIX: imshow solo si hay display disponible
        if self.has_display:
            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Saliendo por tecla Q")
                rclpy.shutdown()

    # =====================================================
    # DESTRUCTOR
    # =====================================================

    def destroy_node(self):
        self.yolo_running = False
        if self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=2.0)
        self.gst_pipeline.set_state(Gst.State.NULL)
        if self.has_display:
            cv2.destroyAllWindows()
        super().destroy_node()


# =========================================================
# MAIN
# =========================================================

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
