import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm
from facenet_pytorch import MTCNN
import ffmpeg

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def clamp_box(x1, y1, x2, y2, w, h):
    """
    Clamp a box to image boundaries. Returns None if invalid.
    """
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def get_device() -> str:
    """
    Force CPU on macOS if FORCE_CPU=1 to avoid MPS pooling issues.
    """
    if os.getenv("FORCE_CPU", "0") == "1":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_last_conv_layer_name(model: nn.Module) -> str:
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = name
    if last_conv is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv


class GradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str):
        self.model = model
        modules = dict([*model.named_modules()])
        if target_layer_name not in modules:
            raise ValueError(f"Layer '{target_layer_name}' not found in model.")
        self.target_layer = modules[target_layer_name]
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x).squeeze(1)  # [B]
        probs = torch.sigmoid(logits)
        probs.sum().backward()

        grads = self.gradients
        acts  = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return probs.detach().cpu().numpy(), cam.detach().cpu().numpy()


def overlay_heatmap(face_rgb_224: np.ndarray, cam_224: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    face_rgb_224: (224,224,3) RGB
    cam_224: (1,224,224) float 0..1
    returns (224,224,3) BGR heatmap overlay
    """
    heat = (cam_224[0] * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(face_rgb_224, cv2.COLOR_RGB2BGR)
    over = cv2.addWeighted(heat, alpha, img_bgr, 1 - alpha, 0)
    return over


def iou(a, b):
    """
    Intersection over Union for two boxes: (x1,y1,x2,y2)
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-8
    return inter / union


def draw_verdict_only(frame_bgr, label: str, anchor=(20, 50)):
    """
    Draw ONLY the verdict label (FAKE/REAL) on the frame.
    """
    text = label.upper()
    color = (0, 0, 255) if text == "FAKE" else (0, 200, 0)

    x, y = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame_bgr, (x - 12, y - th - 18), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(frame_bgr, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return frame_bgr


class DeepfakeDetector:
    def __init__(self, model_path: str, model_name: str = "tf_efficientnet_b0.ns_jft_in1k"):
        self.device = get_device()
        self.tf = build_transforms()

        self.model = timm.create_model(model_name, pretrained=False, num_classes=1)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        last_conv = find_last_conv_layer_name(self.model)
        self.gradcam = GradCAM(self.model, last_conv)

    def detect_face_box(self, rgb: np.ndarray):
        """
        Returns (x1,y1,x2,y2) or None
        """
        box, _ = self.mtcnn.detect(rgb)
        if box is None or len(box) == 0:
            return None
        x1, y1, x2, y2 = box[0]
        return int(x1), int(y1), int(x2), int(y2)

    def predict_face_frame(self, face_rgb_224: np.ndarray):
        x = self.tf(face_rgb_224).unsqueeze(0).to(self.device)
        prob, cam = self.gradcam(x)
        return float(prob[0]), cam[0]  # cam[0] shape (1,224,224)

    def predict_image(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        box = self.detect_face_box(rgb)
        if box is None:
            return {"error": "No face detected."}

        h, w = image_bgr.shape[:2]
        cb = clamp_box(*box, w, h)
        if cb is None:
            return {"error": "Invalid face box after clamping."}
        x1, y1, x2, y2 = cb

        face = rgb[y1:y2, x1:x2]
        if face.size == 0:
            return {"error": "Face crop was empty."}

        face224 = cv2.resize(face, (224, 224))
        prob, cam = self.predict_face_frame(face224)

        label = "FAKE" if prob >= 0.5 else "REAL"
        heat224 = overlay_heatmap(face224, cam)

        out = image_bgr.copy()
        roi = out[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]

        heat_resized = cv2.resize(heat224, (rw, rh), interpolation=cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = heat_resized

        # NOTE: Image output stays as pure heatmap overlay; no text added here.
        return {
            "label": label,
            "probability": prob,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "overlay_bgr": out,
        }

    def predict_video(self, in_path: str, out_path: str, sample_every: int = 1):
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        probs = []
        frame_idx = 0

        # ---- Flicker control knobs (tune via env vars) ----
        alpha_prob = float(os.getenv("SMOOTH_PROB_ALPHA", "0.90"))  # EMA for probability
        alpha_cam  = float(os.getenv("SMOOTH_CAM_ALPHA", "0.85"))   # EMA for cam mask
        hold_frames = int(os.getenv("HOLD_FACE_FRAMES", "6"))       # reuse last face box if detection drops
        min_iou_keep = float(os.getenv("MIN_IOU_KEEP", "0.20"))     # reject sudden jumps

        # ---- State for smoothing ----
        ema_prob = None
        cam_ema = None

        last_box = None
        last_box_ttl = 0

        # ---- Stable verdict via hysteresis ----
        fake_on = float(os.getenv("FAKE_ON", "0.55"))
        real_on = float(os.getenv("REAL_ON", "0.45"))
        verdict = "REAL"

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            out = frame.copy()

            # Only infer every N frames
            do_infer = (frame_idx % sample_every == 0)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            box = None
            if do_infer:
                det = self.detect_face_box(rgb)
                if det is not None:
                    cb = clamp_box(*det, W, H)
                    if cb is not None and last_box is not None:
                        # If box jumps too much, keep old box for stability
                        if iou(cb, last_box) < min_iou_keep:
                            cb = last_box
                        box = cb
                    else:
                        box = cb

                # If detection failed, reuse last box for a few frames
                if box is None and last_box is not None and last_box_ttl > 0:
                    box = last_box
            else:
                # Not inferring; reuse last box if available
                if last_box is not None and last_box_ttl > 0:
                    box = last_box

            # Update box TTL tracking
            if box is not None:
                last_box = box
                last_box_ttl = hold_frames
            else:
                last_box_ttl = max(0, last_box_ttl - 1)

            if box is not None:
                x1, y1, x2, y2 = box
                face = rgb[y1:y2, x1:x2]

                if face.size != 0:
                    if do_infer:
                        face224 = cv2.resize(face, (224, 224))
                        prob, cam = self.predict_face_frame(face224)  # cam: (1,224,224)
                        probs.append(prob)

                        # Smooth probability
                        ema_prob = prob if ema_prob is None else (alpha_prob * ema_prob + (1 - alpha_prob) * prob)

                        # Smooth CAM mask
                        cam_ema = cam if cam_ema is None else (alpha_cam * cam_ema + (1 - alpha_cam) * cam)

                        # Hysteresis to keep verdict stable
                        if verdict == "REAL" and ema_prob >= fake_on:
                            verdict = "FAKE"
                        elif verdict == "FAKE" and ema_prob <= real_on:
                            verdict = "REAL"

                    # Apply the smoothed cam to reduce flicker
                    if cam_ema is not None:
                        face224_for_overlay = cv2.resize(face, (224, 224))
                        heat224 = overlay_heatmap(face224_for_overlay, cam_ema)

                        roi = out[y1:y2, x1:x2]
                        rh, rw = roi.shape[:2]
                        heat_resized = cv2.resize(heat224, (rw, rh), interpolation=cv2.INTER_LINEAR)
                        out[y1:y2, x1:x2] = heat_resized

            # Draw ONLY verdict text (no percentage/probability in video)
            draw_verdict_only(out, verdict, anchor=(20, 50))

            writer.write(out)
            frame_idx += 1

        cap.release()
        writer.release()

        avg_prob = float(np.mean(probs)) if len(probs) else 0.0
        label = "FAKE" if avg_prob >= 0.4 else "REAL"
        return {"label": label, "avg_probability": avg_prob, "frames_scored": len(probs)}
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm
from facenet_pytorch import MTCNN

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def clamp_box(x1, y1, x2, y2, w, h):
    """
    Clamp a box to image boundaries. Returns None if invalid.
    """
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def get_device() -> str:
    """
    Force CPU on macOS if FORCE_CPU=1 to avoid MPS pooling issues.
    """
    if os.getenv("FORCE_CPU", "0") == "1":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_last_conv_layer_name(model: nn.Module) -> str:
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = name
    if last_conv is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv


class GradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str):
        self.model = model
        modules = dict([*model.named_modules()])
        if target_layer_name not in modules:
            raise ValueError(f"Layer '{target_layer_name}' not found in model.")
        self.target_layer = modules[target_layer_name]
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x).squeeze(1)  # [B]
        probs = torch.sigmoid(logits)
        probs.sum().backward()

        grads = self.gradients
        acts  = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return probs.detach().cpu().numpy(), cam.detach().cpu().numpy()


def overlay_heatmap(face_rgb_224: np.ndarray, cam_224: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    face_rgb_224: (224,224,3) RGB
    cam_224: (1,224,224) float 0..1
    returns (224,224,3) BGR heatmap overlay
    """
    heat = (cam_224[0] * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(face_rgb_224, cv2.COLOR_RGB2BGR)
    over = cv2.addWeighted(heat, alpha, img_bgr, 1 - alpha, 0)
    return over


def iou(a, b):
    """
    Intersection over Union for two boxes: (x1,y1,x2,y2)
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-8
    return inter / union


def draw_verdict_only(frame_bgr, label: str, anchor=(20, 50)):
    """
    Draw ONLY the verdict label (FAKE/REAL) on the frame.
    """
    text = label.upper()
    color = (0, 0, 255) if text == "FAKE" else (0, 200, 0)

    x, y = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame_bgr, (x - 12, y - th - 18), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(frame_bgr, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return frame_bgr


class DeepfakeDetector:
    def __init__(self, model_path: str, model_name: str = "tf_efficientnet_b0.ns_jft_in1k"):
        self.device = get_device()
        self.tf = build_transforms()

        self.model = timm.create_model(model_name, pretrained=False, num_classes=1)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        last_conv = find_last_conv_layer_name(self.model)
        self.gradcam = GradCAM(self.model, last_conv)

    def detect_face_box(self, rgb: np.ndarray):
        """
        Returns (x1,y1,x2,y2) or None
        """
        box, _ = self.mtcnn.detect(rgb)
        if box is None or len(box) == 0:
            return None
        x1, y1, x2, y2 = box[0]
        return int(x1), int(y1), int(x2), int(y2)

    def predict_face_frame(self, face_rgb_224: np.ndarray):
        x = self.tf(face_rgb_224).unsqueeze(0).to(self.device)
        prob, cam = self.gradcam(x)
        return float(prob[0]), cam[0]  # cam[0] shape (1,224,224)

    def predict_image(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        box = self.detect_face_box(rgb)
        if box is None:
            return {"error": "No face detected."}

        h, w = image_bgr.shape[:2]
        cb = clamp_box(*box, w, h)
        if cb is None:
            return {"error": "Invalid face box after clamping."}
        x1, y1, x2, y2 = cb

        face = rgb[y1:y2, x1:x2]
        if face.size == 0:
            return {"error": "Face crop was empty."}

        face224 = cv2.resize(face, (224, 224))
        prob, cam = self.predict_face_frame(face224)

        label = "FAKE" if prob >= 0.5 else "REAL"
        heat224 = overlay_heatmap(face224, cam)

        out = image_bgr.copy()
        roi = out[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]

        heat_resized = cv2.resize(heat224, (rw, rh), interpolation=cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = heat_resized

        # NOTE: Image output stays as pure heatmap overlay; no text added here.
        return {
            "label": label,
            "probability": prob,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "overlay_bgr": out,
        }

    def predict_video(self, in_path: str, out_path: str, sample_every: int = 1):
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 25

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if W <= 0 or H <= 0:
            cap.release()
            raise RuntimeError("Invalid video dimensions from input.")

        # ---- HARD CAPS (prevents 502 on Render) ----
        max_seconds = int(os.getenv("MAX_VIDEO_SECONDS", "10"))
        max_frames_env = int(os.getenv("MAX_VIDEO_FRAMES", "140"))
        max_frames_by_time = int(fps * max_seconds)
        max_total_frames = max(1, min(max_frames_env, max_frames_by_time))

        # ---- Compute Grad-CAM less frequently ----
        cam_every = int(os.getenv("CAM_EVERY", "8"))  # compute CAM once every N inference steps
        cam_every = max(1, cam_every)

        # Write to AVI (reliable), then transcode to MP4 (H.264)
        tmp_out = out_path.replace(".mp4", "._tmp.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(tmp_out, fourcc, fps, (W, H))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("VideoWriter failed to open (MJPG/AVI). Cannot write output.")

        probs = []
        frame_idx = 0
        infer_steps = 0  # counts how many times we did inference (sampled frames)

        # ---- Flicker control knobs ----
        alpha_prob = float(os.getenv("SMOOTH_PROB_ALPHA", "0.90"))
        alpha_cam = float(os.getenv("SMOOTH_CAM_ALPHA", "0.85"))
        hold_frames = int(os.getenv("HOLD_FACE_FRAMES", "6"))
        min_iou_keep = float(os.getenv("MIN_IOU_KEEP", "0.20"))

        ema_prob = None
        cam_ema = None
        last_box = None
        last_box_ttl = 0

        fake_on = float(os.getenv("FAKE_ON", "0.55"))
        real_on = float(os.getenv("REAL_ON", "0.45"))
        verdict = "REAL"

        wrote_any = False

        try:
            while True:
                # Stop early to avoid timeouts/OOM
                if frame_idx >= max_total_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                out = frame.copy()
                do_infer = (frame_idx % sample_every == 0)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ---- Face box logic with hold + jump rejection ----
                box = None
                if do_infer:
                    det = self.detect_face_box(rgb)
                    if det is not None:
                        cb = clamp_box(*det, W, H)
                        if cb is not None and last_box is not None:
                            if iou(cb, last_box) < min_iou_keep:
                                cb = last_box
                            box = cb
                        else:
                            box = cb

                    if box is None and last_box is not None and last_box_ttl > 0:
                        box = last_box
                else:
                    if last_box is not None and last_box_ttl > 0:
                        box = last_box

                if box is not None:
                    last_box = box
                    last_box_ttl = hold_frames
                else:
                    last_box_ttl = max(0, last_box_ttl - 1)

                # ---- Inference / overlay ----
                if box is not None:
                    x1, y1, x2, y2 = box
                    face = rgb[y1:y2, x1:x2]

                    if face.size != 0:
                        if do_infer:
                            infer_steps += 1
                            face224 = cv2.resize(face, (224, 224))

                            # Compute CAM only every cam_every steps (BIG speedup)
                            do_cam = (infer_steps % cam_every == 1)  # 1, 1+cam_every, ...

                            if do_cam:
                                # heavy path: prob + cam (backward pass)
                                prob, cam = self.predict_face_frame(face224)
                                # Smooth cam
                                cam_ema = cam if cam_ema is None else (alpha_cam * cam_ema + (1 - alpha_cam) * cam)
                            else:
                                # cheap path: prob only (forward pass)
                                x = self.tf(face224).unsqueeze(0).to(self.device)
                                with torch.no_grad():
                                    logits = self.model(x).squeeze(1)
                                    prob = float(torch.sigmoid(logits).item())
                                cam = None  # keep previous cam_ema (sticky heatmap)

                            probs.append(prob)

                            # Smooth probability
                            ema_prob = prob if ema_prob is None else (alpha_prob * ema_prob + (1 - alpha_prob) * prob)

                            # Stable verdict via hysteresis
                            if verdict == "REAL" and ema_prob >= fake_on:
                                verdict = "FAKE"
                            elif verdict == "FAKE" and ema_prob <= real_on:
                                verdict = "REAL"

                        # Apply the LAST cam_ema (sticky)
                        if cam_ema is not None:
                            face224_for_overlay = cv2.resize(face, (224, 224))
                            heat224 = overlay_heatmap(face224_for_overlay, cam_ema)

                            roi = out[y1:y2, x1:x2]
                            rh, rw = roi.shape[:2]
                            heat_resized = cv2.resize(heat224, (rw, rh), interpolation=cv2.INTER_LINEAR)
                            out[y1:y2, x1:x2] = heat_resized

                # Verdict only
                draw_verdict_only(out, verdict, anchor=(20, 50))

                # Ensure size
                if out.shape[1] != W or out.shape[0] != H:
                    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)

                writer.write(out)
                wrote_any = True
                frame_idx += 1

        finally:
            cap.release()
            writer.release()

        if not wrote_any:
            try:
                os.remove(tmp_out)
            except Exception:
                pass
            raise RuntimeError("No frames were written to the output video.")

        # Transcode AVI -> MP4 (H.264)
        try:
            (
                ffmpeg
                .input(tmp_out)
                .output(
                    out_path,
                    vcodec="libx264",
                    pix_fmt="yuv420p",
                    movflags="faststart",
                    preset="veryfast",
                    crf=23,
                    an=None,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else "no stderr"
            raise RuntimeError("ffmpeg failed: " + stderr)
        finally:
            try:
                os.remove(tmp_out)
            except Exception:
                pass

        avg_prob = float(np.mean(probs)) if len(probs) else 0.0
        label = "FAKE" if avg_prob >= 0.4 else "REAL"
        return {"label": label, "avg_probability": avg_prob, "frames_scored": len(probs)}
