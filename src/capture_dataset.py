import cv2
import os
import time

def capture_dataset(output_dir="../data/private/my_dataset", num_images=50, interval=1.0):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Saving {num_images} images to {output_dir}")
    print("Move your head slowly to capture different angles and expressions.")
    print("Starting in 3 seconds...")
    
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    count = 0
    last_capture_time = time.time()

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        current_time = time.time()
        time_elapsed = current_time - last_capture_time

        # Display instructions and progress on the frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Captured: {count}/{num_images}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Change pose/distance...", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Dataset Capture", display_frame)

        # Capture a frame exactly every 'interval' seconds
        if time_elapsed >= interval:
            img_path = os.path.join(output_dir, f"face_{count:03d}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Captured: {img_path}")
            
            # Flash effect for visual feedback
            cv2.imshow("Dataset Capture", 255 - display_frame)
            cv2.waitKey(50) 
            
            last_capture_time = current_time
            count += 1

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Dataset capture complete.")

if __name__ == "__main__":
    capture_dataset()