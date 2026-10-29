import cv2
import os

def capture_baseline():
    # Adjusted path to step out of helpers/ and src/
    save_dir = "../../data/private"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "baseline_webcam.jpg")

    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press SPACE to capture, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam.")
            break

        display_frame = frame.copy()
        cv2.putText(display_frame, "Look straight ahead with a neutral expression", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, "Press SPACE to save baseline", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Baseline Capture", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.imwrite(save_path, frame)
            print(f"Success! Baseline saved to: {save_path}")
            break
        elif key == ord('q'):
            print("Capture aborted.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_baseline()