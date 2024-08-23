import os
import cv2

def extract_frames(video_path, output_path):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()

def get_all_files(tract_path):
    file_paths = []
    for root, _, files in os.walk(tract_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def main():
    # Set the file paths for the tracts
    lower_tract = "Machine Learning Components/Object Detection & Tracking Component/data/lower-gi-tract"
    upper_tract = "Machine Learning Components/Object Detection & Tracking Component/data/upper-gi-tract"
    
    # Get all the file paths for the tracts
    lower_files = get_all_files(lower_tract)
    upper_files = get_all_files(upper_tract)
    
    # Set the file path for the output tracts
    lower_output = "Machine Learning Components/Object Detection & Tracking Component/preprocessed_data/lower-gi-tract"
    upper_output = "Machine Learning Components/Object Detection & Tracking Component/preprocessed_data/upper-gi-tract"
    
    # Extract frames from the videos for each file for the lower tract
    for file in lower_files:
        relative_path = os.path.relpath(file, lower_tract)
        relative_path = os.path.dirname(relative_path)
        output_dir = os.path.join(lower_output, relative_path, os.path.basename(file), "frames")
        extract_frames(file, output_dir)
    
    # Extract frames from the videos for each file for the upper tract
    for file in upper_files:
        relative_path = os.path.relpath(file, upper_tract)
        relative_path = os.path.dirname(relative_path)
        output_dir = os.path.join(upper_output, relative_path, os.path.basename(file), "frames")
        extract_frames(file, output_dir)

if __name__ == "__main__":
    main()