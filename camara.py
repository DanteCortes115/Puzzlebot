=================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

BLACK_THRESHOLD = 65

MIN_AREA_LIGHT = 300

CONFIRM_FRAMES = 5

DEADZONE = 20


# =========================================================
# NODE
# =========================================================

class VisionNode(Node):

    def __init__(self):

        super().__init__('vision_node')

        # =====================================================
        # PUBLISHERS
        # =====================================================

        self.lane_pub = self.create_publisher(
            Float32,
            '/lane_error',
            10
        )

        self.light_pub = self.create_publisher(
            String,
            '/traffic_light_state',
            10
        )

        # =====================================================
        # CAMERA
        # =====================================================

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():

            raise RuntimeError("Camera error")

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            30
        )

        # =====================================================
        # VARIABLES
        # =====================================================

        self.last_error = 0.0

        self.lost_frames = 0

        self.fsm_state = "NONE"

        self.candidate = "NONE"

        self.candidate_count = 0

        # =====================================================
        # TIMER
        # =====================================================

        self.timer = self.create_timer(
            0.03,
            self.process_frame
        )

    # =====================================================
    # CLEAN MASK
    # =====================================================

    def clean_mask(self, mask):

        kernel = np.ones((5,5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        return mask

    # =====================================================
    # LANE DETECTION
    # =====================================================

    def detect_lane(self, frame):

        h, w = frame.shape[:2]

        # =====================================================
        # 65% ABAJO -> LINEAS
        # =====================================================

        split_y = int(h * 0.35)

        roi = frame[split_y:h, :]

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (7,7),
            0
        )

        # =====================================================
        # THRESHOLD NEGRO
        # =====================================================

        _, thresh = cv2.threshold(
            blur,
            BLACK_THRESHOLD,
            255,
            cv2.THRESH_BINARY_INV
        )

        thresh = self.clean_mask(thresh)

        # =====================================================
        # SOLO ZONA CENTRAL
        # EVITA IRSE A LATERALES
        # =====================================================

        scan_y = int(roi.shape[0] * 0.75)

        row = thresh[scan_y]

        center_min = int(w * 0.20)
        center_max = int(w * 0.80)

        valid_pixels = np.where(
            row[center_min:center_max] > 0
        )[0]

        # =====================================================
        # NO DETECTA
        # =====================================================

        if len(valid_pixels) < 2:

            self.lost_frames += 1

            # SI PIERDE MUCHO
            # AVANZA RECTO
            if self.lost_frames > 5:

                error = 0.0

            else:

                error = self.last_error

            return error

        self.lost_frames = 0

        # =====================================================
        # AJUSTAR COORDENADAS
        # =====================================================

        valid_pixels = valid_pixels + center_min

        # =====================================================
        # DETECTAR LATERALES
        # =====================================================

        left = valid_pixels[0]

        right = valid_pixels[-1]

        # =====================================================
        # CENTRO ENTRE LINEAS
        # =====================================================

        center_lane = int(
            (left + right) / 2
        )

        # =====================================================
        # ERROR
        # =====================================================

        error = center_lane - (w // 2)

        # =====================================================
        # SUAVIZADO
        # =====================================================

        alpha = 0.75

        error = (
            alpha * self.last_error +
            (1 - alpha) * error
        )

        self.last_error = error

        # =====================================================
        # DIBUJOS
        # =====================================================

        cv2.line(
            roi,
            (left,0),
            (left,roi.shape[0]),
            (255,0,0),
            2
        )

        cv2.line(
            roi,
            (right,0),
            (right,roi.shape[0]),
            (255,0,0),
            2
        )

        cv2.line(
            roi,
            (center_lane,0),
            (center_lane,roi.shape[0]),
            (0,0,255),
            3
        )

        cv2.circle(
            roi,
            (center_lane, scan_y),
            8,
            (0,255,0),
            -1
        )

        return float(error)

    # =====================================================
    # TRAFFIC LIGHT
    # =====================================================

    def detect_traffic_light(self, frame):

        h, w = frame.shape[:2]

        split_y = int(h * 0.35)

        roi = frame[0:split_y, :]

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        # =====================================================
        # RED
        # =====================================================

        red1 = cv2.inRange(
            hsv,
            np.array([0,120,90]),
            np.array([10,255,255])
        )

        red2 = cv2.inRange(
            hsv,
            np.array([170,120,90]),
            np.array([179,255,255])
        )

        red = self.clean_mask(red1 | red2)

        # =====================================================
        # YELLOW
        # =====================================================

        yellow = self.clean_mask(
            cv2.inRange(
                hsv,
                np.array([18,120,120]),
                np.array([35,255,255])
            )
        )

        # =====================================================
        # GREEN
        # =====================================================

        green = self.clean_mask(
            cv2.inRange(
                hsv,
                np.array([45,80,80]),
                np.array([85,255,255])
            )
        )

        masks = {
            "RED": red,
            "YELLOW": yellow,
            "GREEN": green
        }

        best_color = "NONE"

        best_area = 0

        best_box = None

        for color, mask in masks.items():

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for c in contours:

                area = cv2.contourArea(c)

                if area < MIN_AREA_LIGHT:
                    continue

                x,y,w_,h_ = cv2.boundingRect(c)

                if area > best_area:

                    best_area = area
                    best_color = color
                    best_box = (x,y,w_,h_)

        # =====================================================
        # SOLO DIBUJAR EL MEJOR
        # =====================================================

        if best_box is not None:

            x,y,w_,h_ = best_box

            cv2.rectangle(
                roi,
                (x,y),
                (x+w_, y+h_),
                (0,255,0),
                3
            )

            cv2.putText(
                roi,
                best_color,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

        return best_color

    # =====================================================
    # FSM
    # =====================================================

    def update_fsm(self, detected):

        if detected == self.candidate:

            self.candidate_count += 1

        else:

            self.candidate = detected

            self.candidate_count = 1

        if self.candidate_count < CONFIRM_FRAMES:
            return

        self.fsm_state = self.candidate

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def process_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        lane_error = self.detect_lane(frame)

        detected = self.detect_traffic_light(frame)

        self.update_fsm(detected)

        # =====================================================
        # PUBLISH
        # =====================================================

        self.lane_pub.publish(
            Float32(data=float(lane_error))
        )

        self.light_pub.publish(
            String(data=self.fsm_state)
        )

        # =====================================================
        # DIVISION
        # =====================================================

        h, w = frame.shape[:2]

        split_y = int(h * 0.35)

        cv2.line(
            frame,
            (0, split_y),
            (w, split_y),
            (0,255,255),
            3
        )

        # =====================================================
        # TEXT
        # =====================================================

        cv2.putText(
            frame,
            f"LIGHT: {self.fsm_state}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"ERROR: {lane_error:.2f}",
            (20, split_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )

        # =====================================================
        # SHOW
        # =====================================================

        cv2.imshow(
            "Camera",
            frame
        )

        cv2.waitKey(1)

    # =====================================================
    # DESTROY
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


