"""Physical-plausibility judgment rules and trajectory checks."""
import json
import re
from pathlib import Path

MIN_TRACK_AREA = 32
EDGE_MARGIN_FRACTION = 0.03

RELIABLE = {
    'wave_or_abstract', 'particles_or_debris', 'human_intrusion', 'new_object',
    'duplication', 'deformation', 'identity_change', 'scene_or_camera_change',
    'abnormal_afterimage',
}


def classify(answer, track):
    label = str(answer.get('label', '')).strip().upper()
    issue = str(answer.get('issue_type', '')).strip().lower()
    if label not in {'PASS', 'FAIL', 'UNCERTAIN'}:
        raise ValueError('Invalid VLM label')
    if issue not in RELIABLE | {'disappearance', 'penetration', 'none', 'uncertain'}:
        raise ValueError('Invalid issue type')
    failed = (label == 'FAIL' and issue in RELIABLE | {'penetration'}) or track['interior_disappearance']
    return 'FAIL' if failed else 'PASS'


def parse_answer(response):
    text = response['choices'][0]['message']['content'].strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text)

def trajectory_audit(path: str):
    data = json.loads(Path(path).read_text())
    width = data["video_contract"]["width"]
    height = data["video_contract"]["height"]
    edge_margin = round(min(width, height) * EDGE_MARGIN_FRACTION)
    object_ids = sorted({
        key for frame in data["records"] for key in frame.get("objects", {})
    })
    objects = []
    has_interior_disappearance = False
    for object_id in object_ids:
        sequence = []
        for frame in data["records"]:
            obj = frame.get("objects", {}).get(object_id)
            area = (obj or {}).get("total_area", 0) or 0
            components = (obj or {}).get("components") or []
            sequence.append((frame["frame_index"], area, components[0] if components else None))
        positive = [x for x in sequence if x[1] >= MIN_TRACK_AREA and x[2]]
        if not positive:
            fate = "untrackable"
            has_interior_disappearance = True
            objects.append({"object_id": object_id, "fate": fate})
            continue
        last = positive[-1]
        later = [x for x in sequence if x[0] > last[0]]
        disappears = bool(later) and all(x[1] < MIN_TRACK_AREA for x in later)
        x, y, w, h = last[2]["bbox_xywh"]
        near_edge = (
            x <= edge_margin
            or y <= edge_margin
            or x + w >= width - edge_margin
            or y + h >= height - edge_margin
        )
        if not disappears:
            fate = "remains_tracked"
        elif near_edge:
            fate = "normal_frame_exit"
        else:
            fate = "interior_track_loss"
            has_interior_disappearance = True
        objects.append({
            "object_id": object_id,
            "last_frame": last[0],
            "last_area": last[1],
            "last_bbox_xywh": [x, y, w, h],
            "near_outer_frame": near_edge,
            "fate": fate,
        })
    return {
        "objects": objects,
        "interior_disappearance": has_interior_disappearance,
        "frame_size": [width, height],
        "min_track_area": MIN_TRACK_AREA,
        "edge_margin_fraction": EDGE_MARGIN_FRACTION,
        "effective_edge_margin_px": edge_margin,
    }
