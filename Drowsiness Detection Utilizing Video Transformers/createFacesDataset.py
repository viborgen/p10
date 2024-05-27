import csv
import cv2
import os
from tqdm import tqdm
import numpy as np
import sys
from contextlib import contextmanager
from retinaface import RetinaFace


path = 'datasets/uta-reallife-drowsiness-dataset'  # replace with your directory path

#lists subjects needed to be processed. Each subject has a corresponding folder.
def list_subjects(path):
    subjects = {}
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.isdigit():
                subjects[int(dir)] = os.path.join(root, dir)
    return subjects

#load video path
def load_video(video, path):
    extensions = ['.mov', '.MOV', '.mp4', '.m4v', '.MP4']  # add more extensions if needed
    for ext in extensions:
        video_path = os.path.join(path, video + ext)
        if os.path.isfile(video_path):
            return video_path, ext
    print(f"Unable to open video {video} in path {path}")
    return None, None



# Load the Haar cascade xml file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def resample_and_split_video(video_path, output_dir, new_fps, clip_length):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video: {video_path}")
        return

    # Get the original video's FPS and calculate the frame skip rate
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = round(original_fps / new_fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = frame_count / original_fps

    # Calculate the number of frames per clip
    frames_per_clip = new_fps * clip_length

    # Extract the subject number and video number from the video path
    path_parts = os.path.normpath(video_path).split(os.sep)
    subject_number = path_parts[-2]
    video_number = os.path.splitext(path_parts[-1])[0]
    clip_prefix = f"{subject_number}_{video_number}"

    # Create a subdirectory for this subject and video
    output_subdir = os.path.join(output_dir, 'fps'+str(new_fps), 'len'+str(clip_length), subject_number, video_number)
    os.makedirs(output_subdir, exist_ok=True)

    # Calculate the number of frames to skip at the start and end
    start_skip = original_fps * 20  # 20 seconds
    end_skip = original_fps * 20  # 20 seconds
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_start = int(start_skip)
    process_end = total_frames - int(end_skip)
    
    clip_number = 0
    frame_number = 0
    global correct_frame_number
    correct_frame_number = 0
    out = None  # Initialize out to None
    smoothed_bbox = None
    alpha = 0.05  # smoothing factor, adjust as needed
    divergence_threshold_percent = 0.03
    detect_face = 5#detect face every x frame
    #expand_amount = 300
    expand_ratio = 0.15 #15 percent
    global distance #used to measure how much the given predicted frame has moved since last
    distance = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_number >= process_end:
            break
        # Only process frames between process_start and process_end
        if frame_number >= process_start:
            # Only process every frame_skip-th frame
            if frame_number % frame_skip == 0:
                #print(f"Correct frame number: {correct_frame_number}, Frame number: {frame_number},Frame count: {frame_count}")
                # Detect faces in the frame
                # Initialize the smoothed bounding box coordinates
                frame_height, frame_width = frame.shape[:2]
                # Calculate the frame's diagonal length
                frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
                expand_amount = int(max(frame_height, frame_width) * expand_ratio)
                divergence_threshold = divergence_threshold_percent * frame_diagonal  # adjust the percentage as needed
                if correct_frame_number % detect_face == 0:
                    faces = RetinaFace.detect_faces(frame)
                    #faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) #openCV solution

                if len(faces) > 0:
                    #(x, y, w, h) = faces[0] #openCV
                    # Check if RetinaFace was used
                    if isinstance(faces, dict) and 'face_1' in faces:
                        x, y, x2, y2 = faces['face_1']['facial_area']
                        w = x2 - x
                        h = y2 - y
                    else:
                        print('face not detected')

                    # Decrease the starting coordinates by half the expand amount, ensuring they stay within the image boundaries
                    x = max(0, x - expand_amount // 2)
                    y = max(0, y - expand_amount // 2)

                    # Increase the dimensions by the expand amount, ensuring they stay within the image boundaries
                    w = min(frame_width - x, w + expand_amount)
                    h = min(frame_height - y, h + expand_amount)
                    if smoothed_bbox is None:
                        # If this is the first frame, initialize the smoothed bounding box
                        smoothed_bbox = [x, y, w, h]
                    else:
                         # Calculate the center of the new and old bounding boxes
                        new_center = np.array([x + w / 2, y + h / 2])
                        old_center = np.array([smoothed_bbox[0] + smoothed_bbox[2] / 2, smoothed_bbox[1] + smoothed_bbox[3] / 2])
                        # Calculate the Euclidean distance between the two centers
                        distance = np.linalg.norm(new_center - old_center)    
                        if distance > divergence_threshold:
                            smoothed_bbox[0] = int(alpha * x + (1 - alpha) * smoothed_bbox[0])
                            smoothed_bbox[1] = int(alpha * y + (1 - alpha) * smoothed_bbox[1])
                            smoothed_bbox[2] = int(alpha * w + (1 - alpha) * smoothed_bbox[2])
                            smoothed_bbox[3] = int(alpha * h + (1 - alpha) * smoothed_bbox[3])

                    # Crop the frame to the smoothed bounding box
                    frame = frame[smoothed_bbox[1]:smoothed_bbox[1]+smoothed_bbox[3], smoothed_bbox[0]:smoothed_bbox[0]+smoothed_bbox[2]]

                else:
                    print('face not detected')
                    detect_face = new_fps
                    if smoothed_bbox is not None:
                        frame = frame[smoothed_bbox[1]:smoothed_bbox[1]+smoothed_bbox[3], smoothed_bbox[0]:smoothed_bbox[0]+smoothed_bbox[2]]
                    if smoothed_bbox is None:
                        continue
                #log smoothed bbox to csv
                with open(f'/home/ubuntu/work/drowsiness_experiment/videos/facesFinal/csv/c{subject_number}_{video_number}.csv', 'a', newline='') as csvfile:
                    fieldnames = ['subject', 'video', 'frame', 'x', 'y', 'w', 'h']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if smoothed_bbox is not None:
                        writer.writerow({'subject': subject_number, 'video': video_number, 'frame': correct_frame_number, 'x': smoothed_bbox[0], 'y': smoothed_bbox[1], 'w': smoothed_bbox[2], 'h': smoothed_bbox[3]})
                
                # Resize the frame to 224x224
                frame = cv2.resize(frame, (224, 224))

                # If this is the first frame of a new clip, open a new video writer
                if int(correct_frame_number) % frames_per_clip == 0:
                    #print(f"Creating new clip: {clip_prefix}_{clip_number}.mp4")
                    print(f"Correct frame number: {correct_frame_number}, Frame number: {frame_number},Frame count: {frame_count}")
                    if out is not None:  # Check if out is not None before calling out.release()
                        out.release()
                    output_path = os.path.join(output_subdir, f"{clip_prefix}_{clip_number}.mp4")
                    print(output_path)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, new_fps, (224, 224))  # Change the frame size to 224x224
                    smoothed_bbox = None  # Reset the smoothed bounding box
                    clip_number += 1

                # Write the frame to the current clip
                if out is not None:  # Check if out is not None before calling out.write(frame)
                    out.write(frame)
                correct_frame_number += 1
        frame_number += 1

    if out is not None:  # Check if out is not None before calling out.release()
        out.release()
    cap.release()


createVideo = True
if createVideo:
    subjects = list_subjects(path)
    for subject in tqdm(sorted(subjects)):
        for video in ['0', '10']:
            print(f"Resampling video {video} for subject {subject}")
            video_path,_ = load_video(video, subjects[subject])
            output_dir = "/videos/facesFinal"  
            # Create the directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            new_fps = 10
            clip_length = 60  # in seconds
            resample_and_split_video(video_path, output_dir, new_fps, clip_length)