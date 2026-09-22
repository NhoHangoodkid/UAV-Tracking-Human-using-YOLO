"""Unified UAV detection and tracking engine combining YOLO and multi-object trackers."""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
UAV_BOTSORT = ROOT / "src" / "uav_botsort.yaml"
UAV_BYTETRACK = ROOT / "src" / "uav_bytetrack.yaml"


class UAVTracker:
    """High-level interface for UAV human detection and multi-object tracking."""

    def __init__(
        self,
        weights_path,
        tracker_type="botsort",
        imgsz=1280,
        conf=0.15,
        iou=0.5,
        device="0",
    ):
        """Initialize YOLO model and tracking engine."""
        self.weights_path = Path(weights_path)
        tracker_name = tracker_type.lower()
        if "botsort" in tracker_name and UAV_BOTSORT.exists():
            self.tracker_config = str(UAV_BOTSORT)
        elif "bytetrack" in tracker_name and UAV_BYTETRACK.exists():
            self.tracker_config = str(UAV_BYTETRACK)
        else:
            self.tracker_config = f"{tracker_name}.yaml"
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(str(self.weights_path))

    def track_stream(self, source, vid_stride=1):
        """Process a video source and yield tracked targets per frame."""
        results = self.model.track(
            source=source,
            imgsz=self.imgsz,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            persist=True,
            device=self.device,
            stream=True,
            vid_stride=vid_stride,
            verbose=False,
        )

        for result in results:
            frame = result.orig_img
            track_ids = []
            bboxes = []
            confs = []

            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                bboxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().tolist()

            yield frame, track_ids, bboxes, confs
