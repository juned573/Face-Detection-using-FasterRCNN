
# Face Detection using Faster R-CNN

This project uses a Faster R-CNN model implemented with PyTorch to detect faces in images and videos. It includes scripts for training, dataset preparation, and inference on video files.

## Project Structure


Fasterrcnn_Project/
convert_annotations.py       # Converts WIDER FACE annotations
dataset.py                   # PyTorch Dataset class for loading images and annotations
detect_faces_in_video.py     # Runs face detection on a video file
train.py                     # Trains the Faster R-CNN model on the dataset
fasterrcnn_face.pth          # Trained model weights
test_video.mp4               # Sample input video
output_video.mp4             # Output video with face detections
wider_face_split/            # Contains dataset split and annotations
requirements.txt             # Python dependencies


## Steps to Follow

### 1. Install Requirements

bash
pip install -r #requirements.txt

### 2. Train the Model

Make sure your dataset is prepared and annotations are converted.

python train.py #training the data wil take time as the dataset has been reduced fromm 12000images to 90images to train it faster


### 3. Run Face Detection on Video

bash
python detect_faces_in_video.py #test_video.mp4...will run
The output will be saved as `output_video.mp4`.

## Notes

- The model is based on `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
- You may need a GPU for faster training and inference or else other models could be used YOLO or SSD
- Ensure your annotation files are in the correct format before training.

## Requirements

See `requirements.txt` or install:

bash
pip install torch torchvision opencv-python


## Author
MD Juned Eqbal
