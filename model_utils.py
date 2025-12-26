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

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % sample_every != 0:
                writer.write(frame)
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box = self.detect_face_box(rgb)

            out = frame.copy()

            if box is not None:
                cb = clamp_box(*box, W, H)
                if cb is not None:
                    x1, y1, x2, y2 = cb
                    face = rgb[y1:y2, x1:x2]

                    if face.size != 0:
                        face224 = cv2.resize(face, (224, 224))
                        prob, cam = self.predict_face_frame(face224)
                        probs.append(prob)

                        heat224 = overlay_heatmap(face224, cam)

                        roi = out[y1:y2, x1:x2]
                        rh, rw = roi.shape[:2]
                        heat_resized = cv2.resize(heat224, (rw, rh), interpolation=cv2.INTER_LINEAR)
                        out[y1:y2, x1:x2] = heat_resized

                        txt = f"Fake prob: {prob:.3f}"
                        y_text = max(20, y1 - 10)
                        cv2.rectangle(out, (x1, max(0, y1 - 30)), (min(W, x1 + 260), y1), (0, 0, 0), -1)
                        cv2.putText(
                            out, txt, (x1 + 5, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )

            writer.write(out)
            frame_idx += 1

        cap.release()
        writer.release()

        avg_prob = float(np.mean(probs)) if len(probs) else 0.0
        label = "FAKE" if avg_prob >= 0.5 else "REAL"
        return {"label": label, "avg_probability": avg_prob, "frames_scored": len(probs)}
