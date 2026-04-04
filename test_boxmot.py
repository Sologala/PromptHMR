from pathlib import Path

import cv2
import numpy as np
from collections import defaultdict
from boxmot import BotSort
from pipeline.tools import interpolate_bboxes, recursive_to_dict
from python_libs.chumpy.chumpy import ch
from ultralytics import YOLO
from tqdm import tqdm
detector = YOLO("data/yolo11x.pt")

tracker = BotSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device="0",
    half=False,
)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("video_path", type=str, default="/home/wen/Documents/33507773976-1-192.mp4")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)

trk_results = defaultdict(lambda: defaultdict(list))

fid = 0
ntotal_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

step_mode = False
for fid in tqdm(range(ntotal_frames), desc="Tracking"):
    ok, frame = cap.read()
    if not ok:
        break



    results = detector(frame, verbose=False, classes=0,
                       device="0", conf=0.25)[0]
    # print(results.boxes)
    detections = np.empty((0, 6), dtype=np.float32)
    if results.boxes is not None and len(results.boxes) > 0:
        detections = results.boxes.data.cpu().numpy()

    tracks = tracker.update(detections, frame)
    tracker.plot_results(frame, show_trajectories=True)

    
    for track in tracks:
        x1, y1, x2, y2, id, conf, cls, det_ind = track
        area = (x2 - x1) * (y2 - y1)
        if area < 100:
            continue

        box_xyxy = [x1, y1, x2, y2]
        trk_results[id]['bboxes'].append(np.array(box_xyxy))
        trk_results[id]['frames'].append(fid)

    # AABB output: (N, 8) = (x1, y1, x2, y2, id, conf, cls, det_ind)
    # OBB output: (N, 9) = (cx, cy, w, h, angle, id, conf, cls, det_ind)
    # Use det_ind to map a track back to the detector output

    cv2.imshow("BoxMOT", frame)

    waittime = step_mode if 0 else 1
    ch = cv2.waitKey(waittime) & 0xFF
    if ch == ord("q"):
        break
    elif ch == ord("s"):
        step_mode = not step_mode
    

print(f"Total tracks: {len(tracks)}")

for k in trk_results:
    bboxes = np.stack(trk_results[k]['bboxes'])
    frames = np.array(trk_results[k]['frames'])     

    interp_bboxes, interp_frames, interp_masks = interpolate_bboxes(
        bboxes, frames, None, fn='linear')


    trk_results[k]['track_id'] = k
    trk_results[k]['frames'] = interp_frames
    trk_results[k]['bboxes'] = interp_bboxes
    trk_results[k]['detected'] = len(interp_bboxes)

trk_results = recursive_to_dict(trk_results)
print(trk_results)

cap.release()
cv2.destroyAllWindows()