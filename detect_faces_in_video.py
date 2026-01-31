import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# Load model
def load_model(weights_path):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

# Inference on a single frame
def detect_faces(model, frame, threshold=0.6):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb).unsqueeze(0)
    with torch.no_grad():
        preds = model(tensor)[0]

    for box, score in zip(preds['boxes'], preds['scores']):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

# Load video and run detection
def run_on_video(video_path, model_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Could not open video.")
        return

    model = load_model(model_path)

    # Optional: video writer to save output
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Running face detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_faces(model, frame)

        cv2.imshow("Face Detection", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")

# Use
run_on_video("test_video.mp4", "fasterrcnn_face.pth", "output_video.mp4")
