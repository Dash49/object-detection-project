import argparse
import base64
import os
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
import urllib.request
from collections import Counter
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')
SCAN_DIR = os.path.join(BASE, 'scans')
PICTURE_DIR = os.path.join(BASE, 'pictureForProj')
MODEL_PATH = os.path.join(MODEL_DIR, 'yolo11n.onnx')
MODEL_URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx'


def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        print(f'✔ มีแล้ว : {MODEL_PATH}')
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print('⬇ กำลังดาวน์โหลด yolo11n.onnx ...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + '.part')
    os.replace(MODEL_PATH + '.part', MODEL_PATH)
    print(f'✔ เสร็จ   : yolo11n.onnx ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)')


CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

IMGSZ = 640
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

net = None


def load_model():
    global net
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    net.setInput(np.zeros((1, 3, IMGSZ, IMGSZ), np.float32))
    net.forward()
    print(f'จำนวนคลาส: {len(CLASSES)} | โหลดโมเดลเรียบร้อย ✔')


def letterbox(frame, size=IMGSZ):
    h, w = frame.shape[:2]
    r = min(size / w, size / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    canvas = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, r, dx, dy


def raw_detect(img):
    padded, r, dx, dy = letterbox(img)
    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (IMGSZ, IMGSZ), swapRB=True, crop=False)
    net.setInput(blob)
    pred = net.forward()[0].T

    scores = pred[:, 4:]
    class_ids = scores.argmax(1)
    confidences = scores[np.arange(len(scores)), class_ids]
    keep = confidences >= CONF_THRESHOLD

    cx, cy, bw, bh = pred[keep, :4].T
    rects = np.stack([(cx - bw / 2 - dx) / r, (cy - bh / 2 - dy) / r, bw / r, bh / r], axis=1)
    return class_ids[keep], confidences[keep], rects


TILE_GRID = 2
TILE_OVERLAP = 0.25


def tiles(w, h, grid=TILE_GRID, overlap=TILE_OVERLAP):
    tw, th = int(w / (grid - (grid - 1) * overlap)), int(h / (grid - (grid - 1) * overlap))
    for gy in range(grid):
        for gx in range(grid):
            yield min(int(gx * tw * (1 - overlap)), w - tw), min(int(gy * th * (1 - overlap)), h - th), tw, th


def cut_by_tile(rects, x0, y0, tw, th, fw, fh, margin=0.02):
    m = margin * min(tw, th)
    x, y, w, h = rects.T
    return (((x0 > 0) & (x < m)) | ((y0 > 0) & (y < m))
            | ((x0 + tw < fw) & (x + w > tw - m)) | ((y0 + th < fh) & (y + h > th - m)))


def drop_fragments(indices, class_ids, rank, rects, ratio=0.7):
    kept = []
    for i in sorted(indices, key=lambda i: -rank[i]):
        x, y, w, h = rects[i]
        for j in kept:
            X, Y, W, H = rects[j]
            iw, ih = min(x + w, X + W) - max(x, X), min(y + h, Y + H) - max(y, Y)
            if class_ids[j] == class_ids[i] and iw > 0 and ih > 0 and iw * ih >= ratio * w * h:
                break
        else:
            kept.append(i)
    return kept


def run_model(frame, tile=False):
    fh, fw = frame.shape[:2]
    ids, confs, rects = raw_detect(frame)
    parts = [(ids, confs, rects, np.zeros(len(ids), bool))]
    if tile and min(fw, fh) >= 32:
        for x0, y0, tw, th in tiles(fw, fh):
            ids, confs, rects = raw_detect(frame[y0:y0 + th, x0:x0 + tw])
            cut = cut_by_tile(rects, x0, y0, tw, th, fw, fh)
            rects[:, :2] += (x0, y0)
            parts.append((ids, confs, rects, cut))
    class_ids, confidences, rects, cut = (np.concatenate(p) for p in zip(*parts))
    if not len(class_ids):
        return []

    rank = np.where(cut, confidences * 0.5, confidences)
    indices = cv2.dnn.NMSBoxesBatched(rects.tolist(), rank.tolist(), class_ids.tolist(),
                                      0.0, float(NMS_THRESHOLD))
    indices = np.array(indices).flatten()
    if tile:
        indices = drop_fragments(indices, class_ids, rank, rects)

    detections = []
    for i in indices:
        x, y, w, h = rects[i]
        x1, y1 = int(np.clip(x, 0, fw - 1)), int(np.clip(y, 0, fh - 1))
        x2, y2 = int(np.clip(x + w, 0, fw - 1)), int(np.clip(y + h, 0, fh - 1))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            detections.append((int(class_ids[i]), float(confidences[i]), (x1, y1, x2 - x1, y2 - y1)))
    return detections


SCENE_PROFILES = {
    'ทั้งหมด (ไม่กรอง)': None,
    'ห้องเรียน': {
        'person', 'chair', 'dining table',
        'laptop', 'mouse', 'keyboard', 'tv',
        'cell phone', 'book', 'backpack', 'handbag',
        'clock', 'scissors', 'bottle', 'cup', 'potted plant',
        'umbrella', 'sports ball',
    },
    'ร้านกาแฟ': {
        'person', 'chair', 'dining table',
        'cup', 'wine glass', 'bottle', 'bowl', 'fork', 'knife', 'spoon',
        'donut', 'cake', 'sandwich', 'pizza',
        'potted plant', 'cell phone', 'laptop', 'handbag',
    },
    'ห้องนอน': {
        'person', 'bed', 'chair', 'teddy bear',
        'laptop', 'keyboard', 'book', 'clock', 'vase', 'potted plant',
        'remote', 'cell phone', 'suitcase', 'hair drier', 'cat', 'dog',
    },
}

SCENE_EN = {
    'ทั้งหมด (ไม่กรอง)': 'All (no filter)',
    'ห้องเรียน': 'Classroom',
    'ร้านกาแฟ': 'Cafe',
    'ห้องนอน': 'Bedroom',
}

KEY_SCENES = {ord('0'): 'ทั้งหมด (ไม่กรอง)', ord('1'): 'ห้องเรียน',
              ord('2'): 'ร้านกาแฟ', ord('3'): 'ห้องนอน'}

CURRENT_SCENE = 'ห้องเรียน'

SCENE_KEYWORDS = {
    'ห้องเรียน': ('classroom', 'class', 'ห้องเรียน'),
    'ร้านกาแฟ': ('cafe', 'café', 'coffee', 'ร้านกาแฟ'),
    'ห้องนอน': ('bedroom', 'bed', 'ห้องนอน'),
}


def guess_scene(path):
    name = os.path.basename(str(path)).lower()
    for scene, words in SCENE_KEYWORDS.items():
        if any(w in name for w in words):
            return scene
    return None


def check_scene_profiles():
    rooms = {name: s for name, s in SCENE_PROFILES.items() if s}
    all_ok = True
    for scene, cls_set in rooms.items():
        wrong = cls_set - set(CLASSES)
        others = set().union(*(s for name, s in rooms.items() if name != scene))
        unique = sorted(cls_set - others)
        ok = not wrong and len(unique) >= 5
        all_ok &= ok
        print(f'{scene:<14}: {len(cls_set):>2} คลาส | เฉพาะห้องนี้ {len(unique):>2} ประเภท '
              f'{"✔" if ok else "❌"}' + (f' สะกดผิด: {wrong}' if wrong else ''))
        print(f'{"":<16}{", ".join(unique)}')
    return all_ok


ROI_ENABLED = True
ROI_ANCHOR = 'bottom'

ROI_POINTS = [(0.10, 0.45), (0.90, 0.45), (1.00, 1.00), (0.00, 1.00)]

ROI_COLOR = (0, 200, 255)


def roi_to_pixels(points, w, h):
    return np.array([[int(px * w), int(py * h)] for px, py in points], dtype=np.int32)


def anchor_point(box):
    x, y, w, h = box
    return (x + w / 2, y + h) if ROI_ANCHOR == 'bottom' else (x + w / 2, y + h / 2)


def in_roi(box, roi_px):
    return cv2.pointPolygonTest(roi_px, tuple(float(v) for v in anchor_point(box)), False) >= 0


def object_status(name, box, allowed, use_roi, roi_px):
    if allowed is not None and name not in allowed:
        return 'scene'
    if use_roi and not in_roi(box, roi_px):
        return 'outside'
    return 'counted'


def draw_roi(frame, roi_px, alpha=0.20):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_px], ROI_COLOR)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [roi_px], True, ROI_COLOR, 2)


BOX_COLOR = (255, 0, 255)
TEXT_COLOR = (0, 255, 0)
OUT_COLOR = (160, 160, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_LINES = 3
HUD_HEIGHT = 12 + 24 * HUD_LINES


def put_label(frame, text, x, y, colour=TEXT_COLOR, scale=0.55):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    y = max(y, th + 4)
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + base - 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 2, y - 2), FONT, scale, colour, 1, cv2.LINE_AA)


def detect_and_draw(frame, scene=None, use_roi=None, detections=None):
    scene = CURRENT_SCENE if scene is None else scene
    use_roi = ROI_ENABLED if use_roi is None else use_roi
    allowed = SCENE_PROFILES.get(scene)

    if detections is None:
        detections = run_model(frame)

    h, w = frame.shape[:2]
    roi_px = roi_to_pixels(ROI_POINTS, w, h)
    if use_roi:
        draw_roi(frame, roi_px)

    counter = Counter()
    for cid, score, box in detections:
        name = CLASSES[cid]
        x, y, bw, bh = box
        status = object_status(name, box, allowed, use_roi, roi_px)

        if status == 'scene':
            continue
        if status == 'outside':
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), OUT_COLOR, 1)
            continue

        counter[name] += 1
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), BOX_COLOR, 2)
        label_y = y - 4 if y - 4 >= HUD_HEIGHT + 20 else HUD_HEIGHT + 22
        put_label(frame, f'{name} {score * 100:.1f}%', x, label_y)
        ax, ay = map(int, anchor_point(box))
        cv2.circle(frame, (ax, ay), 4, TEXT_COLOR, -1)

    return frame, counter


def summary_text(counter, times='×'):
    if not counter:
        return 'ไม่พบวัตถุในพื้นที่' if times == '×' else 'no objects in area'
    return ', '.join(f'{k} {times}{v}' for k, v in counter.most_common())


def draw_hud(frame, counter, fps=None, image_mode=False, show_keys=True):
    roi_state = f'on ({ROI_ANCHOR})' if ROI_ENABLED else 'off'
    lines = [
        f'Scene: {SCENE_EN[CURRENT_SCENE]}    ROI: {roi_state}',
        f'objects in area: {sum(counter.values())}   types: {len(counter)}'
        + (f'   FPS: {fps:.1f}' if fps else ''),
        summary_text(counter, times='x'),
    ]
    panel_h = HUD_HEIGHT
    panel_w = min(frame.shape[1], 12 + max(cv2.getTextSize(t, FONT, 0.6, 1)[0][0] for t in lines))
    region = frame[0:panel_h, 0:panel_w]
    cv2.addWeighted(region, 0.35, np.zeros_like(region), 0.65, 0, region)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8, 26 + 24 * i), FONT, 0.6,
                    (0, 255, 255) if i == 0 else (240, 240, 240), 1, cv2.LINE_AA)

    if not show_keys:
        return
    help_text = '1 class | 2 cafe | 3 bedroom | 0 all | r ROI | a anchor | d draw ROI | s save | q quit'
    if image_mode:
        help_text += ' | o open'
    put_label(frame, help_text, 6, frame.shape[0] - 6, (200, 200, 200), 0.45)


def select_roi_with_mouse(base):
    global ROI_POINTS
    h, w = base.shape[:2]
    pts = []
    win = 'Draw ROI: L-click add | R-click undo | Enter save | Esc cancel'

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            view = base.copy()
            for p in pts:
                cv2.circle(view, p, 5, ROI_COLOR, -1)
            if len(pts) >= 2:
                cv2.polylines(view, [np.array(pts, np.int32)], len(pts) >= 3, ROI_COLOR, 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(20) & 0xFF
            if key == 13 and len(pts) >= 3:
                ROI_POINTS = [(round(x / w, 3), round(y / h, 3)) for x, y in pts]
                print('✔ บันทึก ROI แล้ว — คัดลอกไปใส่ในโค้ดเพื่อใช้ครั้งหน้า:\nROI_POINTS =', ROI_POINTS)
                break
            if key == 27:
                print('ยกเลิก — ใช้ ROI เดิม')
                break
    finally:
        cv2.destroyWindow(win)
        cv2.waitKey(1)


CAM_INDEX = 0


def open_camera(index=CAM_INDEX):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform.startswith('win') else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def grab_still(cap):
    ok, frame = False, None
    for _ in range(5):
        ok, frame = cap.read()
    return frame if ok else None


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
IMAGE_TYPES = [('Image files', ' '.join('*' + e for e in IMAGE_EXTS)), ('All files', '*.*')]


def pick_file(title='เลือกไฟล์ภาพที่จะสแกน', parent=None):
    root = parent or tk.Tk()
    if parent is None:
        root.withdraw()
        root.attributes('-topmost', True)
    path = filedialog.askopenfilename(parent=root, title=title, filetypes=IMAGE_TYPES,
                                      initialdir=PICTURE_DIR if os.path.isdir(PICTURE_DIR) else BASE)
    if parent is None:
        root.destroy()
    return path or None


def read_image(path):
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, MemoryError, cv2.error):
        return None


def test_one_frame(frame, detections=None):
    out, counter = detect_and_draw(frame.copy(), detections=detections)
    print(f'Scene: {CURRENT_SCENE} | ROI: {"เปิด" if ROI_ENABLED else "ปิด"}')
    print('ผล:', summary_text(counter))
    return out, counter


def save_screenshot(frame):
    os.makedirs(SCAN_DIR, exist_ok=True)
    name = SCENE_EN[CURRENT_SCENE].split(' (')[0].replace(' ', '_')
    stem = os.path.join(SCAN_DIR, f'Project_{name}_{time.strftime("%Y%m%d_%H%M%S")}')
    path, n = f'{stem}.jpg', 2
    while os.path.exists(path):
        path, n = f'{stem}_{n}.jpg', n + 1
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError(f'บันทึกภาพไม่ได้ : {path}')
    buf.tofile(path)
    print(f'📷 บันทึกภาพหน้าจอ : {path}')
    return path


WINDOW = 'Scene/ROI Detection (1/2/3 scene, 0 all, r ROI, q quit)'


def bind(trackbar, var_name, max_value, minimum):
    def _cb(value):
        globals()[var_name] = max(value, minimum) / 100.0
    cv2.createTrackbar(trackbar, WINDOW, int(globals()[var_name] * 100), max_value, _cb)


def handle_key(key, frame_for_roi, view):
    global CURRENT_SCENE, ROI_ENABLED, ROI_ANCHOR
    if key in (ord('q'), 27):
        return True
    if key in KEY_SCENES:
        CURRENT_SCENE = KEY_SCENES[key]
        print('Scene ->', CURRENT_SCENE)
    elif key == ord('r'):
        ROI_ENABLED = not ROI_ENABLED
        print('ROI ->', ROI_ENABLED)
    elif key == ord('a'):
        ROI_ANCHOR = 'center' if ROI_ANCHOR == 'bottom' else 'bottom'
        print('จุดอ้างอิง ->', ROI_ANCHOR)
    elif key == ord('d'):
        select_roi_with_mouse(frame_for_roi)
    elif key == ord('s'):
        save_screenshot(view)
    return False


def run_live():
    cap = open_camera(CAM_INDEX)
    first = grab_still(cap)
    if first is None:
        print('❌ เปิดกล้องไม่ได้ — ตรวจสอบ CAM_INDEX หรือสิทธิ์การใช้กล้อง')
        cap.release()
        return 1

    test_one_frame(first)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    bind('conf x100', 'CONF_THRESHOLD', 95, 5)
    bind('nms x100', 'NMS_THRESHOLD', 90, 10)

    prev, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('❌ อ่านภาพจากกล้องไม่ได้')
                break
            raw = frame.copy()
            view, counter = detect_and_draw(frame)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6) if fps else 1 / max(now - prev, 1e-6)
            prev = now
            draw_hud(view, counter, fps)
            cv2.imshow(WINDOW, view)

            if handle_key(cv2.waitKey(1) & 0xFF, raw, view):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


TILING = True
MAX_WINDOW = (1280, 860)


def fit_window(image):
    h, w = image.shape[:2]
    s = min(1.0, MAX_WINDOW[0] / w, MAX_WINDOW[1] / h)
    cv2.resizeWindow(WINDOW, int(w * s), int(h * s))


def scan(path, image):
    print(f'\n🔍 สแกนไฟล์ : {path}')
    detections = run_model(image, tile=TILING)
    out, counter = test_one_frame(image, detections)
    return detections, out, counter


def run_image(path, show=True, save=False):
    global CURRENT_SCENE
    image = read_image(path)
    if image is None:
        print(f'❌ อ่านภาพไม่ได้ : {path}')
        return 1

    detections, out, counter = scan(path, image)
    cache_key = (CONF_THRESHOLD, NMS_THRESHOLD)
    if save:
        draw_hud(out, counter, image_mode=True)
        save_screenshot(out)
    if not show:
        return 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fit_window(image)
    bind('conf x100', 'CONF_THRESHOLD', 95, 5)
    bind('nms x100', 'NMS_THRESHOLD', 90, 10)
    try:
        while True:
            if cache_key != (CONF_THRESHOLD, NMS_THRESHOLD):
                cache_key = (CONF_THRESHOLD, NMS_THRESHOLD)
                detections = run_model(image, tile=TILING)
            view, counter = detect_and_draw(image.copy(), detections=detections)
            draw_hud(view, counter, image_mode=True)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('o'):
                new_path = pick_file()
                new_image = read_image(new_path) if new_path else None
                if new_image is not None:
                    path, image = new_path, new_image
                    CURRENT_SCENE = guess_scene(path) or CURRENT_SCENE
                    detections, _, _ = scan(path, image)
                    cache_key = (CONF_THRESHOLD, NMS_THRESHOLD)
                    fit_window(image)
                elif new_path:
                    print(f'❌ อ่านภาพไม่ได้ : {new_path}')
                continue
            if handle_key(key, image, view):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


def run_folder(folder, show=True, fixed_scene=None):
    global CURRENT_SCENE
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))
    if not paths:
        print(f'❌ ไม่พบไฟล์ภาพใน : {folder}')
        return 1
    for path in paths:
        CURRENT_SCENE = fixed_scene or guess_scene(path) or 'ทั้งหมด (ไม่กรอง)'
        run_image(path, show=show, save=True)
    return 0


STATUS_TH = {'counted': 'นับ (ในพื้นที่)', 'outside': 'นอก ROI', 'scene': 'ไม่ใช่ของในห้องนี้'}
STATUS_ORDER = ('counted', 'outside', 'scene')
HIGHLIGHT_COLOR = (0, 255, 255)
UI_FONT = 'Tahoma'
GUI_MAX_SIDE = 1600
DRAFT_MSG = 'วาด ROI : คลิกซ้ายเพิ่มจุด  ·  คลิกขวาลบจุดล่าสุด  ·  Enter หรือ "บันทึก ROI" เพื่อบันทึก  ·  Esc ยกเลิก'


def object_rows(detections, w, h):
    allowed = SCENE_PROFILES.get(CURRENT_SCENE)
    roi_px = roi_to_pixels(ROI_POINTS, w, h)
    rows = [(CLASSES[cid], score, box, object_status(CLASSES[cid], box, allowed, ROI_ENABLED, roi_px))
            for cid, score, box in detections]
    return sorted(rows, key=lambda r: (STATUS_ORDER.index(r[3]), -r[1]))


class App:
    def __init__(self, root, path=None, save=False, scene=None):
        self.root = root
        self.path = self.image = self.view = self.photo = None
        self.detections, self.rows = [], []
        self.selected = self.draft = self.resize_job = None
        self.elapsed = 0.0
        self.scale, self.offset = 1.0, (0, 0)

        root.title('Object Detection — ห้องเรียน / ร้านกาแฟ / ห้องนอน')
        root.geometry('1360x820')
        root.minsize(960, 600)
        try:
            root.state('zoomed')
        except tk.TclError:
            pass
        for name in ('TkDefaultFont', 'TkTextFont', 'TkHeadingFont', 'TkMenuFont'):
            tkfont.nametofont(name).configure(family=UI_FONT, size=10)
        ttk.Style(root).configure('Treeview', rowheight=26)

        self.build_toolbar()
        self.status = tk.StringVar(value='เลือกรูปภาพเพื่อเริ่มสแกน')
        ttk.Label(root, textvariable=self.status, anchor='w', padding=(8, 3)).pack(side='bottom', fill='x')
        self.build_panels()
        self.bind_keys()
        root.after(100, lambda: self.start(path, save, scene))

    def build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side='top', fill='x')
        ttk.Button(bar, text='เลือกรูปภาพ…', command=self.open_file).pack(side='left')
        ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=10)

        ttk.Label(bar, text='สถานที่').pack(side='left')
        self.scene_var = tk.StringVar(value=CURRENT_SCENE)
        scene_box = ttk.Combobox(bar, textvariable=self.scene_var, values=list(SCENE_PROFILES),
                                 state='readonly', width=16)
        scene_box.pack(side='left', padx=(6, 0))
        scene_box.bind('<<ComboboxSelected>>', lambda e: self.set_scene(self.scene_var.get()))
        ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=10)

        self.roi_var = tk.BooleanVar(value=ROI_ENABLED)
        ttk.Checkbutton(bar, text='ใช้ ROI', variable=self.roi_var, command=self.on_roi).pack(side='left')
        ttk.Label(bar, text='จุดอ้างอิง').pack(side='left', padx=(10, 0))
        self.anchor_var = tk.StringVar(value=ROI_ANCHOR)
        anchor_box = ttk.Combobox(bar, textvariable=self.anchor_var, values=['bottom', 'center'],
                                  state='readonly', width=7)
        anchor_box.pack(side='left', padx=(6, 0))
        anchor_box.bind('<<ComboboxSelected>>', lambda e: self.on_anchor())
        self.roi_btn = ttk.Button(bar, text='วาด ROI ใหม่', command=self.toggle_draft)
        self.roi_btn.pack(side='left', padx=(10, 0))
        ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=10)

        ttk.Label(bar, text='Confidence').pack(side='left')
        self.conf_var = tk.DoubleVar(value=CONF_THRESHOLD * 100)
        self.conf_label = ttk.Label(bar, text=f'{CONF_THRESHOLD * 100:.0f}%', width=5)
        conf = ttk.Scale(bar, from_=5, to=95, variable=self.conf_var, length=160,
                         command=lambda v: self.conf_label.config(text=f'{float(v):.0f}%'))
        conf.pack(side='left', padx=6)
        conf.bind('<ButtonRelease>', lambda e: self.on_conf())
        conf.bind('<KeyRelease>', lambda e: self.on_conf())
        self.conf_label.pack(side='left')

        ttk.Button(bar, text='บันทึกผล', command=self.save).pack(side='right')

    def build_panels(self):
        pane = ttk.PanedWindow(self.root, orient='horizontal')
        pane.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(pane, bg='#202124', highlightthickness=0, width=600)
        pane.add(self.canvas, weight=1)
        side = ttk.Frame(pane, padding=(10, 8), width=540)
        side.pack_propagate(False)
        pane.add(side, weight=0)

        ttk.Label(side, text='รายละเอียดการสแกน', font=(UI_FONT, 12, 'bold')).pack(anchor='w')
        self.text = tk.Text(side, height=12, width=40, wrap='word', font=(UI_FONT, 10), relief='flat',
                            bg='#f6f6f6', padx=8, pady=6, tabs=(130,), state='disabled')
        self.text.pack(fill='x', pady=(6, 10))
        self.text.tag_configure('h', font=(UI_FONT, 10, 'bold'))
        self.text.tag_configure('dim', foreground='#808080')

        ttk.Label(side, text='วัตถุที่ตรวจพบ (คลิกแถวหรือกรอบบนภาพเพื่อไฮไลต์)',
                  font=(UI_FONT, 11, 'bold')).pack(anchor='w')
        table = ttk.Frame(side)
        table.pack(fill='both', expand=True, pady=(6, 0))
        self.tree = ttk.Treeview(table, columns=('no', 'name', 'conf', 'pos', 'size', 'status'),
                                 show='headings', selectmode='browse')
        for col, title, width, anchor in (('no', '#', 34, 'center'), ('name', 'วัตถุ', 98, 'w'),
                                          ('conf', 'ความมั่นใจ', 72, 'e'), ('pos', 'ตำแหน่ง (x, y)', 92, 'center'),
                                          ('size', 'ขนาด (ก × ส)', 86, 'center'), ('status', 'สถานะ', 110, 'w')):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, minwidth=30, anchor=anchor, stretch=col == 'status')
        scroll = ttk.Scrollbar(table, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        self.tree.tag_configure('counted', foreground='#0b7a28')
        self.tree.tag_configure('outside', foreground='#8a8a8a')
        self.tree.tag_configure('scene', foreground='#b5b5b5')
        self.tree.bind('<<TreeviewSelect>>', self.on_select)

        self.canvas.bind('<Configure>', self.on_resize)
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<Button-3>', self.on_right_click)

    def bind_keys(self):
        for key in ('o', 'O'):
            self.root.bind(f'<Control-{key}>', lambda e: self.open_file())
        for key in ('s', 'S'):
            self.root.bind(f'<Control-{key}>', lambda e: self.save())
        self.root.bind('<Escape>', lambda e: self.cancel_draft())
        self.root.bind('<Return>', lambda e: self.finish_draft())
        for key, scene in KEY_SCENES.items():
            self.root.bind(f'<KeyPress-{chr(key)}>', lambda e, s=scene: self.set_scene(s))

    def start(self, path, save, scene):
        self.render_empty()
        self.open_file(path, scene)
        if save:
            self.save()

    def open_file(self, path=None, scene=None):
        self.cancel_draft()
        path = path or pick_file(parent=self.root)
        if not path:
            return
        image = read_image(path)
        if image is None or min(image.shape[:2]) < 3:
            messagebox.showerror('อ่านภาพไม่ได้', f'เปิดไฟล์นี้ไม่ได้ :\n{path}', parent=self.root)
            return
        s = GUI_MAX_SIDE / max(image.shape[:2])
        if s < 1:
            image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        previous = self.path, self.image
        self.path, self.image = path, image
        self.set_scene(scene or guess_scene(path) or CURRENT_SCENE, refresh=False)
        if not self.detect():
            self.path, self.image = previous
            self.refresh()

    def detect(self):
        if self.image is None:
            return False
        self.status.set(f'กำลังสแกน {os.path.basename(self.path)} …')
        self.root.config(cursor='watch')
        self.root.update_idletasks()
        start = time.time()
        try:
            detections = run_model(self.image, tile=TILING)
        except (cv2.error, ValueError, ZeroDivisionError, MemoryError) as e:
            messagebox.showerror('สแกนไม่สำเร็จ', str(e), parent=self.root)
            return False
        finally:
            self.root.config(cursor='')
        self.detections, self.elapsed = detections, time.time() - start
        self.refresh()
        return True

    def refresh(self):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        self.rows = object_rows(self.detections, w, h)
        self.selected = None
        self.render()
        self.fill_text()
        self.fill_tree()
        counted = sum(r[3] == 'counted' for r in self.rows)
        self.status.set(DRAFT_MSG if self.draft is not None else
                        f'{os.path.basename(self.path)}  ·  สแกนเสร็จใน {self.elapsed:.2f} วินาที  ·  '
                        f'ตรวจพบ {len(self.rows)} ชิ้น  ·  อยู่ในพื้นที่ {counted} ชิ้น')

    def set_scene(self, scene, refresh=True):
        global CURRENT_SCENE
        CURRENT_SCENE = scene
        self.scene_var.set(scene)
        if refresh:
            self.refresh()

    def on_roi(self):
        global ROI_ENABLED
        ROI_ENABLED = self.roi_var.get()
        self.refresh()

    def on_anchor(self):
        global ROI_ANCHOR
        ROI_ANCHOR = self.anchor_var.get()
        self.refresh()

    def on_conf(self):
        global CONF_THRESHOLD
        value = round(self.conf_var.get()) / 100
        if value != round(CONF_THRESHOLD * 100) / 100:
            CONF_THRESHOLD = value
            self.detect()

    def render(self):
        h, w = self.image.shape[:2]
        thick = max(2, round(max(h, w) / 400))
        if self.draft is not None:
            view = self.image.copy()
            pts = np.array(self.draft, np.int32).reshape(-1, 2)
            for x, y in pts:
                cv2.circle(view, (int(x), int(y)), thick * 3, ROI_COLOR, -1)
            if len(pts) >= 2:
                cv2.polylines(view, [pts], len(pts) >= 3, ROI_COLOR, thick)
        else:
            view, counter = detect_and_draw(self.image.copy(), detections=self.detections)
            draw_hud(view, counter, show_keys=False)
            if self.selected is not None:
                name, score, (x, y, bw, bh), _ = self.rows[self.selected]
                cv2.rectangle(view, (x - thick, y - thick), (x + bw + thick, y + bh + thick),
                              HIGHLIGHT_COLOR, thick * 2)
                put_label(view, f'{name} {score * 100:.1f}%', x, max(y - thick * 2, HUD_HEIGHT + 22),
                          HIGHLIGHT_COLOR, 0.7)
        self.view = view
        self.show()

    def show(self):
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        h, w = self.view.shape[:2]
        s = min(cw / w, ch / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(self.view, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self.photo = tk.PhotoImage(data=base64.b64encode(cv2.imencode('.png', small)[1].tobytes()))
        self.scale, self.offset = s, ((cw - nw) // 2, (ch - nh) // 2)
        self.canvas.delete('all')
        self.canvas.create_image(*self.offset, image=self.photo, anchor='nw')

    def render_empty(self):
        self.canvas.delete('all')
        self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                                text='กด "เลือกรูปภาพ…" หรือ Ctrl+O เพื่อเริ่มสแกน',
                                fill='#bbbbbb', font=(UI_FONT, 14))

    def on_resize(self, event):
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(80, self.redraw)

    def redraw(self):
        self.resize_job = None
        if self.view is None:
            self.render_empty()
        else:
            self.show()

    def report_lines(self, with_objects=False):
        h, w = self.image.shape[:2]
        counted = Counter(r[0] for r in self.rows if r[3] == 'counted')
        by_status = Counter(r[3] for r in self.rows)
        roi = f'เปิด  (จุดอ้างอิง {ROI_ANCHOR})' if ROI_ENABLED else 'ปิด'
        lines = [
            ('ข้อมูลภาพ', 'h'),
            (f'ไฟล์\t{os.path.basename(self.path)}', ''),
            (f'ขนาดภาพ\t{w} × {h} px', ''),
            (f'สถานที่\t{CURRENT_SCENE}  ({SCENE_EN[CURRENT_SCENE]})', ''),
            (f'ROI\t{roi}', ''),
            (f'Confidence\t≥ {CONF_THRESHOLD * 100:.0f}%   ·   ใช้เวลา {self.elapsed:.2f} วินาที', ''),
            ('', ''),
            (f'วัตถุในพื้นที่ : {sum(counted.values())} ชิ้น  ·  {len(counted)} ประเภท', 'h'),
        ]
        lines += [(f'   {name}\t× {n}', '') for name, n in counted.most_common()] \
            or [('   ไม่พบวัตถุในพื้นที่', 'dim')]
        lines += [('', ''), (f'ตรวจพบทั้งหมด {len(self.rows)} ชิ้น', 'h')]
        lines += [(f'   {STATUS_TH[s]}\t{by_status[s]} ชิ้น', '' if s == 'counted' else 'dim')
                  for s in STATUS_ORDER if by_status[s]]
        if with_objects:
            lines += [('', ''), ('รายการวัตถุ', 'h')]
            lines += [(f'{i + 1:>3}. {name:<14} {score * 100:5.1f}%   x={x} y={y} w={bw} h={bh}   {STATUS_TH[s]}', '')
                      for i, (name, score, (x, y, bw, bh), s) in enumerate(self.rows)]
        return lines

    def fill_text(self):
        lines = self.report_lines()
        self.text.config(state='normal', height=min(len(lines), 24))
        self.text.delete('1.0', 'end')
        for line, tag in lines:
            self.text.insert('end', line + '\n', tag)
        self.text.config(state='disabled')

    def fill_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, (name, score, (x, y, w, h), status) in enumerate(self.rows):
            self.tree.insert('', 'end', iid=str(i), tags=(status,),
                             values=(i + 1, name, f'{score * 100:.1f}%', f'{x}, {y}', f'{w} × {h}', STATUS_TH[status]))

    def on_select(self, event=None):
        sel = self.tree.selection()
        selected = int(sel[0]) if sel else None
        if selected != self.selected:
            self.selected = selected
            if self.draft is None:
                self.render()

    def to_image(self, event):
        h, w = self.image.shape[:2]
        x, y = (event.x - self.offset[0]) / self.scale, (event.y - self.offset[1]) / self.scale
        return (x, y) if 0 <= x < w and 0 <= y < h else None

    def on_click(self, event):
        if self.image is None or (point := self.to_image(event)) is None:
            return
        x, y = point
        if self.draft is not None:
            self.draft.append((x, y))
            self.render()
            return
        hits = [i for i, (_, _, (bx, by, bw, bh), s) in enumerate(self.rows)
                if s != 'scene' and bx <= x <= bx + bw and by <= y <= by + bh]
        if hits:
            i = min(hits, key=lambda i: self.rows[i][2][2] * self.rows[i][2][3])
            self.tree.selection_set(str(i))
            self.tree.see(str(i))

    def on_right_click(self, event):
        if self.draft:
            self.draft.pop()
            self.render()

    def toggle_draft(self):
        if self.image is None:
            return
        if self.draft is not None:
            self.finish_draft()
            return
        self.draft = []
        self.roi_btn.config(text='บันทึก ROI')
        self.status.set(DRAFT_MSG)
        self.render()

    def finish_draft(self):
        global ROI_POINTS, ROI_ENABLED
        if self.draft is None:
            return
        if len(self.draft) < 3:
            self.status.set('ต้องมีอย่างน้อย 3 จุดจึงจะบันทึก ROI ได้')
            return
        h, w = self.image.shape[:2]
        ROI_POINTS = [(round(x / w, 3), round(y / h, 3)) for x, y in self.draft]
        ROI_ENABLED = True
        self.roi_var.set(True)
        print('ROI_POINTS =', ROI_POINTS)
        self.end_draft()

    def cancel_draft(self):
        if self.draft is not None:
            self.end_draft()

    def end_draft(self):
        self.draft = None
        self.roi_btn.config(text='วาด ROI ใหม่')
        self.refresh()

    def save(self):
        if self.image is None:
            return
        self.cancel_draft()
        view, counter = detect_and_draw(self.image.copy(), detections=self.detections)
        draw_hud(view, counter, show_keys=False)
        path = save_screenshot(view)
        report = os.path.splitext(path)[0] + '.txt'
        with open(report, 'w', encoding='utf-8') as f:
            f.write('\n'.join(line for line, _ in self.report_lines(with_objects=True)) + '\n')
        self.status.set(f'บันทึกแล้ว : {path}  และ  {os.path.basename(report)}')


def run_gui(path=None, save=False, scene=None):
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    App(root, path, save, scene)
    root.mainloop()
    return 0


def main():
    global CURRENT_SCENE, ROI_ENABLED, ROI_ANCHOR, CONF_THRESHOLD, NMS_THRESHOLD, CAM_INDEX, TILING

    parser = argparse.ArgumentParser(description='Object Detection เฉพาะพื้นที่ 3 สถานที่ (Scene + ROI)')
    parser.add_argument('--source', default='pick',
                        help='pick = เปิดหน้าต่างเลือกไฟล์ (ค่าเริ่มต้น), เลขกล้อง, path ของไฟล์ภาพ หรือโฟลเดอร์')
    parser.add_argument('--scene', type=int, default=None, choices=[0, 1, 2, 3],
                        help='0 ทั้งหมด, 1 ห้องเรียน, 2 ร้านกาแฟ, 3 ห้องนอน (ไม่ระบุ = เดาจากชื่อไฟล์)')
    parser.add_argument('--no-roi', action='store_true', help='ปิดการกรองด้วย ROI')
    parser.add_argument('--anchor', default=ROI_ANCHOR, choices=['bottom', 'center'])
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD)
    parser.add_argument('--nms', type=float, default=NMS_THRESHOLD)
    parser.add_argument('--no-tile', action='store_true', help='(ภาพนิ่ง) ไม่แบ่งช่องสแกน — เร็วขึ้นแต่เจอของเล็กน้อยลง')
    parser.add_argument('--draw-roi', action='store_true', help='วาด ROI ด้วยเมาส์ก่อนเริ่ม')
    parser.add_argument('--save', action='store_true', help='(ภาพนิ่ง) บันทึกผลลง scans/ ทันที')
    parser.add_argument('--no-show', action='store_true', help='(ภาพนิ่ง) ไม่เปิดหน้าต่าง')
    args = parser.parse_args()

    fixed_scene = None if args.scene is None else KEY_SCENES[ord(str(args.scene))]
    ROI_ENABLED = not args.no_roi
    ROI_ANCHOR = args.anchor
    CONF_THRESHOLD, NMS_THRESHOLD = args.conf, args.nms
    TILING = not args.no_tile

    print('Python :', sys.version.split()[0], '| OpenCV :', cv2.__version__)
    download_model()
    load_model()
    if not check_scene_profiles():
        return 1

    if args.source == 'pick':
        if not (args.no_show or args.draw_roi):
            CURRENT_SCENE = fixed_scene or CURRENT_SCENE
            return run_gui(save=args.save, scene=fixed_scene)
        args.source = pick_file()
        if not args.source:
            print('ยกเลิก — ไม่ได้เลือกไฟล์')
            return 1
    elif not args.source.isdigit() and not os.path.exists(args.source) \
            and os.path.exists(os.path.join(BASE, args.source)):
        args.source = os.path.join(BASE, args.source)

    is_folder = os.path.isdir(args.source)
    CURRENT_SCENE = fixed_scene or guess_scene(args.source) or CURRENT_SCENE

    if args.draw_roi:
        if args.source.isdigit():
            cap = open_camera(int(args.source))
            still = grab_still(cap)
            cap.release()
        elif is_folder:
            first = sorted(f for f in os.listdir(args.source) if f.lower().endswith(IMAGE_EXTS))[:1]
            still = read_image(os.path.join(args.source, first[0])) if first else None
        else:
            still = read_image(args.source)
        if still is not None:
            select_roi_with_mouse(still)

    if args.source.isdigit():
        CAM_INDEX = int(args.source)
        return run_live()
    if is_folder:
        return run_folder(args.source, show=not args.no_show, fixed_scene=fixed_scene)
    if args.no_show:
        return run_image(args.source, show=False, save=args.save)
    return run_gui(args.source, save=args.save, scene=fixed_scene)


if __name__ == '__main__':
    sys.exit(main())
