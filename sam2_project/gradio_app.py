import gradio as gr
import numpy as np
import cv2
from PIL import Image as _Image
from gradio_image_prompter import ImagePrompter
from sam2.build_sam import build_few_shot_predictor
import torch
import os 
import pickle

VALID_IMAGE_FORMAT = {"png", "jpg", "jpeg"}
# Global variable to store input points
input_points = []
input_images = []
images = []
global_labels = []
global_memory_path = ""

def get_user_input(user_input_path: str) -> list:
    if user_input_path.endswith(".mp4"):
        return extract_frames(user_input_path)
    elif user_input_path.split(".")[-1] in VALID_IMAGE_FORMAT:
        return get_image_user_input(user_input_path)
    
    raise NotImplementedError
    
def get_image_user_input(user_input_image_path: str) -> list:
    frame = cv2.imread(user_input_image_path)
    assert frame is not None, "Could not read the given image"
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    global input_images
    input_images = [frame_rgb].copy()
    return [frame_rgb]


def extract_frames(video_path: str) -> list:
    """Extract frames from a video at 2 FPS."""
    frames = []
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the interval for 2 FPS
    interval = int(fps / 1)  # Number of frames to skip for 2 FPS

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:  # Capture every 2nd frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame_rgb)
        count += 1

    cap.release()
    global input_images
    input_images = frames.copy()
    return frames

def get_video_with_memory(user_input_path: str, memory_path: str) -> list:
    """Generate masks for each frame based on memory."""
    frames = extract_frames(user_input_path)
    # Example of applying masks (for now, just returning original frames)
    return frames

def add_mask_to_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to the frame."""
    # Ensure the frame and mask have the same dimensions
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # If the mask is a single channel (binary), convert it to 3 channels
    if len(mask.shape) == 2:  # Single channel
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR

    # Check if mask has 3 channels
    if mask.shape[2] != 3:
        raise ValueError("Mask must have 3 channels")

    # Blend the frame and the mask
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blended_frame = cv2.addWeighted(frame, 1.0, mask, 0.2, 0)  

    blended_frame = cv2.drawContours(blended_frame, contours, -1, 255, thickness=1)
                
    return blended_frame

def create_image_gallery(frames: list, masks: list) -> list:
    """Generate a list of images from frames and masks for the gallery."""
    mixed_frames = [add_mask_to_frame(frames[i], masks[i]) for i in range(len(frames))]
    return mixed_frames  # Return the list of mixed frames

def save_prompt(points: dict) -> None:
    """Save prompt in temp state memory"""
    global input_points, images, global_labels

    input_points.append([[p[0], p[1]] for p in points['points']])  # Save points to global variable
    images.append(points["image"])
    global_labels.append([1 for _ in range(len(points["points"]))])

def reset_points() -> None:
    global points, images, input_images, global_labels
    points, images, input_images, global_labels = [], [], [], []

# Gradio Interface
def main(sam2_cfg: str, sam2_checkpoint: str):
    predictor = build_few_shot_predictor(sam2_cfg, sam2_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    boxes = None
    with gr.Blocks() as demo:
        gr.Markdown("## Video Annotation and Masking App")

        # Checkbox for using memory
        with gr.Row() as r:
            memory_path = gr.File(file_types=['.pt'])
            memory_path_input = gr.Textbox(label="Memory Path", placeholder="Enter memory path here (where to save a reprensentation of the object)")
        
        with gr.Row() as r:
            user_input = gr.File(file_types=[".mp4", ".png", ".jpg", "jpeg"], label="Input Video or image")
            frame_gallery = gr.Gallery(label="Extracted Frames")
            image_prompter = ImagePrompter(label="Annotate Selected Frame", interactive=True)

        # Button to apply/save memory
        with gr.Row() as r:
            add_memory_button = gr.Button("Add image and prompt to memory")
            reset_memory_button = gr.Button("Reset Memroy")
            apply_button = gr.Button("Apply / Save Memory")
        
        # Outputs
        with gr.Row() as r:
            output_gallery = gr.Gallery(label="Output Gallery")

        # Extract frames from video and display in gallery
        user_input.change(lambda user_input_path: get_user_input(user_input_path), inputs=user_input, outputs=frame_gallery)

        # Select frame for annotation
        frame_gallery.select(lambda image: image, inputs=frame_gallery, outputs=image_prompter)

        reset_memory_button.click(reset_points)

        add_memory_button.click(
            lambda points: save_prompt(points),
            inputs=image_prompter,
            outputs=None
        )

        # Process based on memory usage
        def process_memory(file_memory: str | None, exit_memory_path: str):
            use_memory = True if file_memory is not None else False
            if not use_memory:
                print("Will initialize memory with the input point prompt")
                global input_points, global_labels
                memory, num_obj_ptr_tokens = None, 0
                points = input_points.copy()
                labels = global_labels.copy()

            else:
                points, labels = None, None
                memory = torch.load(file_memory).to(device)
                print(memory.shape)
                # TODO Remove this
                num_obj_ptr_tokens = memory.shape[0] // 1025 # obj ptr token size
                print(num_obj_ptr_tokens)
            global images, input_images
            np_images = images if len(images) > 0 else input_images
            res_mask, res = predictor.init_state(np_images=np_images,
                                    memory=memory, points=points,
                                    labels=labels, box=boxes, device=device, num_obj_ptr_tokens=num_obj_ptr_tokens
                                    )
            if not use_memory:
                mem_dir, new_memory_name = os.path.split(exit_memory_path)
                new_memory_name = new_memory_name + ".pt" if not new_memory_name.endswith(".pt") else new_memory_name
                torch.save(res.to("cpu"), os.path.join(mem_dir, new_memory_name))
                masks = [(_res > 0).astype(np.uint8) * 255 for _res in res_mask]

                return create_image_gallery(frames=np_images, masks=masks) 
            else:
                masks = [(_res > 0).astype(np.uint8) * 255 for _res in res_mask]
                return create_image_gallery(frames=np_images, masks=masks)

        # Process button action
        apply_button.click(process_memory, inputs=[memory_path, memory_path_input], outputs=[output_gallery])

    demo.launch()

if __name__ == "__main__":
    sam2_cfg = "sam2_hiera_l"
    sam2_checkpoint = r"checkpoints\sam2_hiera_large.pt"
    main(sam2_cfg=sam2_cfg, sam2_checkpoint=sam2_checkpoint)
