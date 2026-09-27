import cv2
_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def get_color(class_id):
    return _COLORS[int(class_id) % len(_COLORS)]


def draw_tracked_objects(frame, tracked_objects, class_names):
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id, conf, class_id = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        class_id = int(class_id)

        class_name = class_names.get(class_id, "object")
        color = get_color(class_id)

        label = f"{class_name} | ID:{track_id} | {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return frame


def draw_fps(frame, fps):
    """Draws the current FPS counter on the top-left of the frame."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return frame