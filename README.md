# PuzzleBot — Line Follower ROS2

Sistema de seguimiento de línea autónomo basado en ROS2 para el robot **PuzzleBot**, con detección de semáforos, señales de tráfico y obstáculos por LiDAR.
## Arquitectura del sistema

```
vision_node  ──────────────────────────────────────────────┐
  │  /lane_error          → controller_node                │
  │  /traffic_light_state → controller_node                │
  │  /curve_mode          → controller_node                │
  │  /dash_state          → controller_node                │
  │  /sign_command        → controller_node                │
  │  /speed_multiplier    → controller_node                │
  └──────────────────────────────────────────────────────  │
                                                            │
lidar_node ────────────────────────────────────────────────┤
  │  /lidar_obstacle      → controller_node                │
  │  /lidar_distance      → controller_node                │
  └──────────────────────────────────────────────────────  │
                                                            ▼
                                                   controller_node
                                                        │
                                                        ▼
                                                   /cmd_vel (Twist)
```

---

## Nodos

### `vision_node.py` (v8)

Captura video desde la cámara CSI (Jetson) mediante GStreamer e implementa:

- **Seguimiento de línea** por centroide ponderado de columnas (sin Hough).
- **Detección de curva** mediante regresión lineal (`polyfit`) sobre los últimos centroides.
- **Detección de cruce** por perfil de filas en zona media de la ROI.
- **Semáforo** con detección robusta de rojo, amarillo (3 rangos HSV) y verde.
- **Señales de tráfico** con YOLOv8 ONNX en hilo separado, filtradas por tamaño aparente y confirmadas por N frames consecutivos.

**Tópicos publicados:**

| Tópico | Tipo | Descripción |
|---|---|---|
| `/lane_error` | `Float32` | Error lateral en píxeles respecto al setpoint |
| `/traffic_light_state` | `String` | `RED` / `YELLOW` / `GREEN` / `NONE` / `STOP_PRIORITY` |
| `/curve_mode` | `String` | `CURVA` / `NORMAL` |
| `/dash_state` | `String` | `SOLIDA` / `PAUSA` |
| `/sign_command` | `String` | Comando de acción de señal |
| `/speed_multiplier` | `Float32` | Factor de velocidad (1.0 = normal, 0.5 = lento) |
| `/traffic_sign` | `String` | Señal actualmente confirmada |

---

### `controller_node.py` (v4)

Controlador PID con lógica anti-oscilación post-curva:

- Ganancias interpoladas suavemente entre modo NORMAL y CURVA (`ALPHA_GAINS_SLOW` / `ALPHA_GAINS_FAST`).
- Zona muerta (`DEAD_ZONE_PX`) para suprimir micro-correcciones cerca del centro.
- Damping adicional de angular cuando el error es pequeño.
- Reset de integral y `prev_error` al salir de curva.
- FSM de señales: `IDLE` → `TURNING` → `REINCORP` → `IDLE` / `STOPPING` / `SLOWING`.

**Tópicos suscritos:**

| Tópico | Tipo |
|---|---|
| `/lane_error` | `Float32` |
| `/traffic_light_state` | `String` |
| `/curve_mode` | `String` |
| `/dash_state` | `String` |
| `/sign_command` | `String` |
| `/speed_multiplier` | `Float32` |
| `/lidar_obstacle` | `String` |
| `/lidar_distance` | `Float32` |

**Tópico publicado:** `/cmd_vel` (`geometry_msgs/Twist`)

#### Parámetros PID

| Parámetro | Recta | Curva |
|---|---|---|
| `KP` | 0.0065 | 0.0070 |
| `KI` | 0.0000001 | 0.0000001 |
| `KD` | 0.00014 | 0.00016 |
| Velocidad lineal | 0.10 m/s | 0.055 m/s |

---

### `lidar_node.py`

Lector directo del **RPLidar A1** vía puerto serie, sin `rplidar_ros`:

- Escanea los 180° frontales (270°–360° y 0°–90°).
- Publica `DANGER` si algún punto está a menos de `ZONA_PELIGRO` (100 mm).
- Zona ciega configurable para ignorar el cable/conector.
- El controlador se detiene 2 segundos al recibir `DANGER`.

**Tópicos publicados:**

| Tópico | Tipo | Descripción |
|---|---|---|
| `/lidar_obstacle` | `String` | `DANGER` / `CLEAR` |
| `/lidar_distance` | `Float32` | Distancia mínima en mm |

---

## Modelo YOLO

- Formato: `.onnx` (YOLOv8)
- Ruta: `/home/puzzlebot/models/completo.onnx`
- Clases detectadas:

| ID | Señal |
|---|---|
| 0 | Ahead Only |
| 1 | Construction |
| 2 | Give Way |
| 3 | Turn left ahead |
| 4 | Turn right ahead |
| 5 | Roundabout |
| 6 | Stop |

---

**Procedimiento:**

1. Poner `SIGN_CALIBRATE = True` en `vision_node.py`.
2. Colocar la señal a ~7 cm de la cámara.
3. Ejecutar el nodo y anotar el valor `ratio` en los logs.
4. Asignar ese valor a `SIGN_MAX_DIST_CM`.
5. Poner `SIGN_CALIBRATE = False`.

> Cuanto más **pequeño** el valor, más cerca debe estar la señal para activarse.

---

## Requisitos

- ROS2 (Humble o superior)
- Python 3.8+
- `ultralytics`, `opencv-python`, `numpy`, `pyserial`
- GStreamer con soporte NVMM (Jetson)
- RPLidar A1 en `/dev/ttyUSB1`

---

## Lanzamiento

```bash
# Terminal 1 — visión + YOLO + semáforo
ros2 run puzzlebot vision_node

# Terminal 2 — controlador PID
ros2 run puzzlebot controller_node

# Terminal 3 — LiDAR
ros2 run puzzlebot lidar_node
```

---

## Parámetros clave ajustables

| Archivo | Parámetro | Descripción |
|---|---|---|
| `vision_node.py` | `SETPOINT_RATIO` | Fracción X del centro del carril |
| `vision_node.py` | `ROI_SPLIT` | Fracción Y donde inicia la ROI |
| `vision_node.py` | `SIGN_MAX_DIST_CM` | Umbral de tamaño para activar señal |
| `vision_node.py` | `CRUCE_ROW_RATIO` | Sensibilidad de detección de cruce |
| `controller_node.py` | `LINEAR_SPEED` | Velocidad lineal en recta |
| `controller_node.py` | `DEAD_ZONE_PX` | Zona muerta de error angular |
| `controller_node.py` | `KP / KI / KD` | Ganancias PID normales y de curva |
| `lidar_node.py` | `ZONA_PELIGRO` | Distancia de parada en mm |
| `lidar_node.py` | `OFFSET_ANGULO` | Corrección de rotación física del sensor |
