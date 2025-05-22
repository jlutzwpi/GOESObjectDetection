import sys, getopt
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner
from PIL import Image, ImageDraw, ImageFont
import os
import glob
from datetime import datetime
import re
import cv2

# Configuration
input_location = "/home/justin/goes19/fd/ch13_enhanced"
output_location = "/home/justin/edge_impulse_goes/cropped-ei"
input_dir = input_location  # Input images
output_dir = output_location  # Cropped images
crop_box = (2484, 500, 3106, 900)  # (left, top, right, bottom) for New England
max_images = 50  # Limit to last 20 images
font_size = 24  # Font size for timestamp
text_color = (255, 255, 255)  # White text
text_position = (10, 10)  # Top-left corner
# the model name from edge impulse
model = "/home/justin/edge_impulse_goes/goes-weather-linux-aarch64-v1.eim"
gif_frames = []
output_gif = "/home/justin/edge_impulse_goes/ne_weather.gif"
frame_duration = 400
loop = 0

# Initialize the Edge Impulse runner
def initialize_runner(model_path):
    if not os.path.exists(model_path):
        raise Exception(f"Model file not found: {model_path}")
    
    runner = ImageImpulseRunner(model_path)
    try:
        model_info = runner.init()
        print("Model initialized successfully")
        print("Model info:", model_info)
        return runner
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise
        
def get_recent_images(base_dir, num_images=20):
    # Find all image files in date-named subdirectories
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(base_dir, "*/", ext)))
    
    # Sort by modification time (newest first)
    image_paths.sort(key=os.path.getmtime, reverse=True)
    
    # Return the most recent num_images
    print(f"[{datetime.now()}] Found {len(image_paths)} images, selecting {min(num_images, len(image_paths))}")
    return image_paths[:num_images]

# Function to extract timestamp from filename
def get_timestamp(filename):
    match = re.search(r'(\d{8}_\d{4})', filename)
    if match:
        dt = datetime.strptime(match.group(1), '%Y%m%d_%H%M')
        return dt.strftime('%Y-%m-%d %H:%M')  # Format: 2025-05-16 12:30
    # Fallback to modification time
    mtime = os.path.getmtime(filename)
    return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')

def crop_images(runner):
    # Font (use DejaVuSans, available on Raspberry Pi OS)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        print("DejaVuSans.ttf not found, using default font")
        font = ImageFont.load_default()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the 20 most recent images
    image_files = get_recent_images(input_dir, num_images=20)
    if not image_files:
        print(f"[{datetime.now()}] No images found in {input_dir}")
        return

    # Crop to the New England and add timestamp
    for file in image_files:
        output_path = os.path.join(output_dir, f"cropped_{os.path.basename(file)}")
        if os.path.exists(output_path):
            print(f"Skipping {file}: Already cropped")
            continue
        
        print(f"Processing {file}")
        img = Image.open(file)
        
        # Verify image size
        if img.size != (5424, 5424):
            print(f"Skipping {file}: Expected 5424x5424, got {img.size}")
            continue
        
        # Crop
        cropped_img = img.crop(crop_box)
        
        #convert to numpy array
        # Ensure image is in RGB mode
        if cropped_img.mode != "RGB":
            cropped_img = cropped_img.convert("RGB")
        img_array = np.array(cropped_img)  # Shape: (height, width, 3), RGB
        
        #run the Edge Impulse object detection classifier
        features, ei_img = runner.get_features_from_image(img_array)
        ei_result = runner.classify(features)
        
        # Process bounding box results
        detections = ei_result["result"]["bounding_boxes"]
        if not detections:
            print("No objects detected.")
        
        img_bgr = cv2.cvtColor(ei_img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        for detection in detections:
            print(f"Label: {detection['label']}, Score: {detection['value']:.4f}, "
                  f"Box: (x={detection['x']}, y={detection['y']}, w={detection['width']}, h={detection['height']})")
            x, y, width, height = detection["x"], detection["y"], detection["width"], detection["height"]
            top_left = (x, y)
            bottom_right = (x + width, y + height)
            cv2.rectangle(img_bgr, top_left, bottom_right, (0, 255, 0), 2)
            #cv2.putText(img_bgr, f"{detection['label']}: {detection['value']:.2f}",
            #            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert back to RGB
        cropped_img = Image.fromarray(img)  # Convert NumPy array to PIL Image
        # Add timestamp
        draw = ImageDraw.Draw(cropped_img)
        timestamp = get_timestamp(file)
        draw.text(text_position, timestamp, font=font, fill=text_color)
        #add to gif_frames
        gif_frames.append(cropped_img)
        #write the cropped image
        cv2.imwrite(output_path, img)
        print(f"Saved {output_path}")

#write the list of images to a gif        
def write_gif():
    if gif_frames:
        print(f"Saving GIF to {output_gif}")
        gif_frames[0].save(
            output_gif,
            save_all=True,
            append_images=gif_frames[::-1], #start from the end of the list and work to the first element
            duration=frame_duration,
            loop=loop
        )
        print("GIF created successfully!")
    else:
        print("No frames to save")

def main():
    
    # Initialize the runner
    runner = initialize_runner(model)
    
    try:
        
        # crop image and run inference
        crop_images(runner)
        # write images to gif
        write_gif()
        
    finally:
        # Clean up the runner
        runner.stop()
        print("Runner stopped")

if __name__ == "__main__":
    main()
