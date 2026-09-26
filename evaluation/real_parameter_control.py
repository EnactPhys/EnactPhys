"""Directional response readout for matched PhysDelta-Real task pairs."""
import json
import math
from pathlib import Path

def response(path: Path, scene: str, object_index: int) -> float | None:
    if not path.is_file(): return None
    payload = json.loads(path.read_text())
    if scene == "S03" and (payload.get("status") != "COMPLETE" or len(payload.get("records", [])) != 49):
        return None
    points = []
    for frame in payload["records"]:
        point = frame["objects"].get(str(object_index), {}).get("centroid_xy", [None, None])
        if point[0] is not None and point[1] is not None: points.append((float(point[0]), float(point[1])))
    if len(points) < 2: return None
    x0, y0 = points[0]
    if scene in {"S02", "S11"}:
        return max(y - y0 for _, y in points)
    if scene in {"S01", "S03", "S04", "S06", "S08", "S10"}:
        return max(abs(x - x0) for x, _ in points)
    return max(math.hypot(x - x0, y - y0) for x, y in points)
