import gradio as gr
from ultralytics import YOLO
from PIL import Image
import cv2  # ✅ fix: lowercase

# Load the trained YOLO model
model = YOLO('best.pt')

# Define the prediction function
def detect_objects(img):
    # Run inference
    results = model(img)
    # Get the plotted image with bounding boxes and convert BGR to RGB
    annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_img)

# Create the Gradio interface
app = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Car Defects Object Detection using YOLO",
    description="Upload an image and the model will detect objects."
)

# Launch the app
# Launch app
if __name__ == "__main__":
    app.launch()
