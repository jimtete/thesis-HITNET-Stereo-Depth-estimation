import cv2

def save_frames(video_file, output_name):
    cap = cv2.VideoCapture(video_file)

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 25 == 0:
            height, width, _ = frame.shape
            left_frame = frame #[:, :width//2, :]
            right_frame = frame #[:, width//2:, :]

            left_filename = f"{output_name}_left_{saved_frame_count:05d}.png"
            right_filename = f"{output_name}_right_{saved_frame_count:05d}.png"

            cv2.imwrite(left_filename, left_frame)
            cv2.imwrite(right_filename, right_frame)

            saved_frame_count += 1

        frame_count += 1

    cap.release()

    print(f"Saved {saved_frame_count} frames as PNG images.")

# Usage example
video_file = "right_stage_2.mp4"
output_name = "./ateith_dataset/ATEITH_2306/image"
save_frames(video_file, output_name)
