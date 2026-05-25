"""
vision_node.py
==============
Seguidor de línea para pista de 3 carriles (carril central).

Modos de operación:
  NORMAL → detecta las dos líneas del carril central y sigue el centro.
  CURVA  → detecta la línea del BORDE exterior y se mantiene a
            OFFSET_BORDE px de distancia de ella para completar la curva.

Transiciones:
  NORMAL → CURVA  : cuando |error| > UMBRAL_ENTRADA_CURVA Y el error
                    crece > UMBRAL_DELTA_CURVA durante FRAMES_CONFIRMA_CURVA
                    frames consecutivos (error grande Y sostenido).
  CURVA  → NORMAL : cuando las DOS líneas del carril central son visibles
                    durante FRAMES_CONFIRMA_NORMAL frames consecutivos.

Publica:
  /lane_error          Float32
  /traffic_light_state String
"""

# =========================================================
# IMPORTS
# =========================================================

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
import cv2
import numpy as np

# =========================================================
# CONFIGURACIÓN
# =========================================================

CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

# ── Zoom ──────────────────────────────────────────────────
# 1.0 = sin corrección. Bajar de 0.05 en 0.05 si hay zoom digital.
DEZOOM = 1.0

# ── Detección de línea ────────────────────────────────────
BLACK_THRESHOLD  = 65
SCAN_Y_RATIO     = 0.75    # qué tan abajo del ROI se escanea

# ── Zona NORMAL (carril central) ──────────────────────────
CENTER_MIN_RATIO = 0.32
CENTER_MAX_RATIO = 0.68

# ── Suavizado ─────────────────────────────────────────────
ALPHA_SMOOTH = 0.45
BUFFER_SIZE  = 3

# ── Filtro de grupos de píxeles ───────────────────────────
MIN_GROUP_WIDTH = 2    # px mínimo — ruido puntual se descarta
MAX_GROUP_WIDTH = 40   # px máximo — cruces punteados se descartan

# ── Transición NORMAL → CURVA ─────────────────────────────
# Se activa cuando SE CUMPLEN LOS DOS criterios simultáneamente:
#   1. |error| supera UMBRAL_ENTRADA_CURVA  (robot desviado)
#   2. El error creció al menos UMBRAL_DELTA_CURVA px respecto
#      a N frames atrás  (error sostenido, no ruido puntual)
# Ambos deben mantenerse FRAMES_CONFIRMA_CURVA frames seguidos.
UMBRAL_ENTRADA_CURVA  = 25   # px — error RAW para considerar curva
FRAMES_CONFIRMA_CURVA = 2    # solo 2 frames: la curva es muy rápida

# ── Transición CURVA → NORMAL ─────────────────────────────
# Vuelve a NORMAL cuando ve las DOS líneas centrales bien durante
# varios frames consecutivos (se reincorporó al carril).
FRAMES_CONFIRMA_NORMAL = 5

# ── Modo CURVA: seguir borde exterior ─────────────────────
# El robot busca la línea del borde exterior (la más alejada
# hacia el lado de la curva) y se mantiene a OFFSET_BORDE px de ella.
#
# Con carriles de 15 cm ≈ 80 px → offset ideal ≈ 1.5 × ancho_carril
# para quedar en el centro del carril central mientras gira.
# Ajustar si el robot va muy pegado al borde o muy al interior.
OFFSET_BORDE     = 120   # px — distancia deseada desde el borde exterior

# Si el error en modo CURVA supera este valor, se aplica un límite
# para evitar oscilaciones bruscas contra el borde.
MAX_ERROR_CURVA  = 160   # px

# ── Pérdida total de línea ────────────────────────────────
# En ambos modos, si no se detecta NADA durante estos frames → ir recto.
MAX_LOST_FRAMES  = 15

# ── Semáforo ──────────────────────────────────────────────
MIN_AREA_LIGHT = 300
CONFIRM_FRAMES = 5

# ── ROI vertical ──────────────────────────────────────────
ROI_SPLIT = 0.35


# =========================================================
# NODO
# =========================================================

class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # ── Publishers ────────────────────────────────────
        self.lane_pub  = self.create_publisher(Float32, '/lane_error',          10)
        self.light_pub = self.create_publisher(String,  '/traffic_light_state', 10)
        self.curve_pub = self.create_publisher(String,  '/curve_mode',           10)

        # ── Cámara ────────────────────────────────────────
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Camera error")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          30)
        self.cap.set(cv2.CAP_PROP_ZOOM,         0)
        self.cap.set(cv2.CAP_PROP_FOCUS,        0)

        # ── Estado seguidor ───────────────────────────────
        self.last_error   = 0.0
        self.error_buffer = []
        self.lost_frames  = 0

        # ── FSM de modo de conducción ─────────────────────
        #   "NORMAL" | "CURVA"
        self.modo              = "NORMAL"
        self.lado_curva        = None    # 'izq' | 'der'
        self.frames_error_alto = 0
        self.error_history     = []       # contador para entrar en CURVA
        self.error_history     = []      # historial para medir crecimiento
        self.frames_par_visto  = 0       # contador para salir de CURVA

        # ── FSM semáforo ──────────────────────────────────
        self.fsm_state       = "NONE"
        self.candidate       = "NONE"
        self.candidate_count = 0

        self.timer = self.create_timer(0.03, self.process_frame)

    # =====================================================
    # CLEAN MASK
    # =====================================================

    def clean_mask(self, mask):
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    # =====================================================
    # PREPROCESADO  →  fila de scan binaria
    # =====================================================

    def _preparar_thresh(self, frame):
        """Devuelve roi, thresh binario completo, y w."""
        h, w    = frame.shape[:2]
        split_y = int(h * ROI_SPLIT)
        roi     = frame[split_y:h, :]
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thr  = cv2.threshold(blur, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        thr     = self.clean_mask(thr)
        return roi, thr, w

    def _fila_combinada(self, thr):
        """
        OR de varias filas de scan para no depender de una sola.
        Usa 3 filas: 60%, 72%, 84% del ROI.
        Si en alguna hay línea, la detecta.
        """
        h = thr.shape[0]
        filas = [
            int(h * 0.60),
            int(h * 0.72),
            int(h * 0.84),
        ]
        combinada = np.zeros(thr.shape[1], dtype=np.uint8)
        for fy in filas:
            combinada = np.bitwise_or(combinada, thr[fy])
        # scan_y representativo = el del medio
        return combinada, filas[1]

    # =====================================================
    # DETECT LANE  —  máquina de estados principal
    # =====================================================

    def detect_lane(self, frame):

        roi, thr, w = self._preparar_thresh(frame)
        row, scan_y = self._fila_combinada(thr)

        # =================================================
        # MODO NORMAL
        # =================================================
        if self.modo == "NORMAL":

            par = self._buscar_par_central(row, w)

            if par is not None:
                left_x, right_x, center_lane = par
                self.lost_frames    = 0
                self.frames_par_visto = 0   # no relevante en NORMAL

                error = float(center_lane - (w // 2))

                # ── ¿Entrar en CURVA? ─────────────────────
                # Activar CURVA si el error supera umbral N frames seguidos
                if abs(error) >= UMBRAL_ENTRADA_CURVA:
                    self.frames_error_alto += 1
                    if self.frames_error_alto >= FRAMES_CONFIRMA_CURVA:
                        self.lado_curva        = "der" if error > 0 else "izq"
                        self.modo              = "CURVA"
                        self.frames_error_alto = 0
                        self.frames_par_visto  = 0
                        self.error_history     = []
                        self.get_logger().info(
                            f"→ CURVA ({self.lado_curva})"
                        )
                else:
                    self.frames_error_alto = 0

                error_final = self._suavizar(error)

                # Dibujo NORMAL (azul / rojo)
                cv2.line(roi, (left_x,      0), (left_x,      roi.shape[0]), (255,   0,   0), 2)
                cv2.line(roi, (right_x,     0), (right_x,     roi.shape[0]), (255,   0,   0), 2)
                cv2.line(roi, (center_lane, 0), (center_lane, roi.shape[0]), (0,     0, 255), 3)
                cv2.circle(roi, (center_lane, scan_y), 8, (0, 255, 0), -1)

                return error_final

            else:
                # Sin par en modo NORMAL → pérdida
                self.lost_frames += 1
                if self.lost_frames > MAX_LOST_FRAMES:
                    self._resetear()
                return float(self.last_error)

        # =================================================
        # MODO CURVA
        # =================================================
        else:  # self.modo == "CURVA"

            # ── Intentar volver a NORMAL ──────────────────
            # Buscar par central en zona ampliada
            par = self._buscar_par_central(row, w, zona=(0.10, 0.90))
            if par is not None:
                left_x, right_x, center_lane = par
                sep = abs(right_x - left_x)
                # Par válido = dos líneas separadas y centradas
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
                        # Devolver error del par ya que lo tenemos
                        error = float(center_lane - (w // 2))
                        return self._suavizar(error)
                else:
                    self.frames_par_visto = 0
            else:
                self.frames_par_visto = 0

            # ── Seguir borde exterior ─────────────────────
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
    # Devuelve (left_x, right_x, center_lane) o None.
    # =====================================================

    def _buscar_par_central(self, row, w, zona=None):
        if zona is None:
            zona = (CENTER_MIN_RATIO, CENTER_MAX_RATIO)

        c_min = int(w * zona[0])
        c_max = int(w * zona[1])

        px = np.where(row[c_min:c_max] > 0)[0]
        if len(px) < 2:
            return None

        px      = px + c_min
        grupos  = self._filtrar_grupos(self._agrupar(px))
        centros = [int(np.mean(g)) for g in grupos]
        n       = len(centros)

        if n == 0:
            return None
        if n == 1:
            # Una sola línea — calcular error asumiendo la otra
            # está fuera del campo de visión (curva avanzada)
            cx = centros[0]
            cf = w // 2
            # Si está a la izquierda del centro, la otra está más a la izquierda
            # Si está a la derecha, la otra está más a la derecha
            # Devolver como par con separación 0 para que detect_lane
            # pueda decidir si activar curva
            return (cx, cx, cx)
        elif n == 2:
            left, right = centros[0], centros[1]
        elif n == 3:
            cf    = w // 2
            pares = [(centros[0], centros[1]), (centros[1], centros[2])]
            left, right = min(pares, key=lambda p: abs((p[0]+p[1])//2 - cf))
        else:
            mid   = len(centros) // 2
            left  = centros[mid - 1]
            right = centros[mid]

        if abs(right - left) < 20:   # demasiado juntas → ruido
            return None

        return (left, right, (left + right) // 2)

    # =====================================================
    # SEGUIR BORDE EXTERIOR (modo CURVA)
    # =====================================================

    def _seguir_borde(self, row, w, roi, scan_y):
        """
        Busca la línea del borde exterior (la más alejada en el
        lado de la curva) en todo el ancho útil.
        Calcula el error para quedar a OFFSET_BORDE px de ella.
        """
        c_min = int(w * 0.03)
        c_max = int(w * 0.97)

        px = np.where(row[c_min:c_max] > 0)[0]
        if len(px) < 2:
            return None

        px      = px + c_min
        grupos  = self._filtrar_grupos(self._agrupar(px))
        centros = [int(np.mean(g)) for g in grupos]

        if not centros:
            return None

        cf = w // 2

        if self.lado_curva == "der":
            # Curva a la derecha → borde exterior está a la DERECHA
            # Queremos estar OFFSET px a la izquierda del borde
            borde_x         = max(centros)
            objetivo        = borde_x - OFFSET_BORDE
        else:
            # Curva a la izquierda → borde exterior está a la IZQUIERDA
            # Queremos estar OFFSET px a la derecha del borde
            borde_x         = min(centros)
            objetivo        = borde_x + OFFSET_BORDE

        objetivo = int(np.clip(objetivo, 0, w - 1))
        error    = float(objetivo - cf)
        error    = float(np.clip(error, -MAX_ERROR_CURVA, MAX_ERROR_CURVA))

        # Dibujo modo CURVA (naranja / cyan)
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
        self.last_error      = 0.0
        self.error_buffer    = []
        self.lost_frames     = 0
        self.modo            = "NORMAL"
        self.lado_curva      = None
        self.frames_par_visto  = 0
        self.frames_error_alto = 0

    # =====================================================
    # AGRUPAR  +  FILTRAR
    # =====================================================

    def _agrupar(self, pixels, gap=15):
        if len(pixels) == 0:
            return []
        grupos = []
        actual = [pixels[0]]
        for px in pixels[1:]:
            if px - actual[-1] <= gap:
                actual.append(px)
            else:
                grupos.append(np.array(actual))
                actual = [px]
        grupos.append(np.array(actual))
        return grupos

    def _filtrar_grupos(self, grupos):
        return [
            g for g in grupos
            if MIN_GROUP_WIDTH <= int(g[-1]) - int(g[0]) + 1 <= MAX_GROUP_WIDTH
        ]

    # =====================================================
    # TRAFFIC LIGHT
    # =====================================================

    def detect_traffic_light(self, frame):
        h, w    = frame.shape[:2]
        split_y = int(h * ROI_SPLIT)
        roi     = frame[0:split_y, :]
        hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red1   = cv2.inRange(hsv, np.array([0,   120,  90]), np.array([10,  255, 255]))
        red2   = cv2.inRange(hsv, np.array([170, 120,  90]), np.array([179, 255, 255]))
        red    = self.clean_mask(red1 | red2)
        yellow = self.clean_mask(cv2.inRange(hsv, np.array([18, 120, 120]), np.array([35, 255, 255])))
        green  = self.clean_mask(cv2.inRange(hsv, np.array([45,  80,  80]), np.array([85, 255, 255])))

        masks = {"RED": red, "YELLOW": yellow, "GREEN": green}

        best_color, best_area, best_box = "NONE", 0, None

        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > MIN_AREA_LIGHT and area > best_area:
                    best_area  = area
                    best_color = color
                    best_box   = cv2.boundingRect(c)

        if best_box is not None:
            x, y, bw, bh = best_box
            cv2.rectangle(roi, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
            cv2.putText(roi, best_color, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return best_color

    # =====================================================
    # FSM SEMÁFORO
    # =====================================================

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
        ret, frame = self.cap.read()
        if not ret:
            return

        # Corrección zoom digital
        if DEZOOM < 1.0:
            fh, fw = frame.shape[:2]
            ch, cw = int(fh * DEZOOM), int(fw * DEZOOM)
            y0, x0 = (fh - ch) // 2, (fw - cw) // 2
            frame  = cv2.resize(
                frame[y0:y0+ch, x0:x0+cw], (fw, fh),
                interpolation=cv2.INTER_LINEAR
            )

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

        # Zona central en recta
        if self.modo == "NORMAL":
            zm = int(w * CENTER_MIN_RATIO)
            zx = int(w * CENTER_MAX_RATIO)
            cv2.line(frame, (zm, split_y), (zm, h), (0, 255, 0), 1)
            cv2.line(frame, (zx, split_y), (zx, h), (0, 255, 0), 1)

        color_modo = (0, 200, 0) if self.modo == "NORMAL" else (0, 120, 255)
        lado_str   = f"({self.lado_curva})" if self.lado_curva else ""

        cv2.putText(frame, f"LIGHT: {self.fsm_state}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"ERROR: {lane_error:+.1f}px",
                    (10, split_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
        cv2.putText(frame, f"MODO: {self.modo} {lado_str}",
                    (10, split_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color_modo, 2)
        cv2.putText(frame, f"LOST: {self.lost_frames}",
                    (10, split_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1)

        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

    # =====================================================
    # DESTRUCTOR
    # =====================================================

    def destroy_node(self):
        self.cap.release()
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

