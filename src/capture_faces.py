#!/usr/bin/env python3
"""
Simplified Face Capture Script
--------------------------------
You will be asked:

1. Person Name
2. Number of Images to Capture

Press SPACE to capture each image.
Press Q to quit early.
"""

import os
import cv2
import time

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    print("=== FACE CAPTURE TOOL ===")

    # Ask for name
    person_name = input("Enter person's name: ").strip()
    if not person_name:
        print("Name cannot be empty!")
        return

    # Ask number of images
    try:
        num_images = int(input("How many images to capture? (recommended: 15–30): "))
    except ValueError:
        print("Invalid number!")
        return

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "known_faces", person_name)
    ensure_dir(out_dir)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print(f"\nCapturing {num_images} images for '{person_name}'")
    print("Press SPACE to capture an image.")
    print("Press Q to quit early.\n")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Frame failed… trying again.")
            time.sleep(0.1)
            continue

        # Display instructions on frame
        display = frame.copy()
        cv2.putText(display, f"{person_name}: {count}/{num_images}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(display, "SPACE = capture | Q = quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Capture Faces", display)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            print("Quitting…")
            break

        # Capture image
        if key == 32:  # SPACEBAR
            filename = os.path.join(out_dir, f"{person_name}_{count+1:03d}.jpg")
            cv2.imwrite(filename, frame)
            print("Saved:", filename)
            count += 1
            time.sleep(0.25)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Captured {count} images for '{person_name}'.")
    print("You may now run face_embedding.py to build embeddings.")

if __name__ == "__main__":
    main()
