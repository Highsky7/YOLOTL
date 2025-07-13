#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility Script: BEV (Birds-Eye View) Parameter Setup
---------------------------------------------------
(Modified Version) Supports webcams, saved videos, and images, based on 1280x720 resolution.
    - Specify video/camera source (--source)
    - Set BEV parameters using a single image if an image file is provided.
    - Set BEV parameters with a continuously updating frame for video files/webcams.

Modifications:
- Changed to allow the user to manually select all 4 points required for BEV setup.
- The functionality to save the 4 selected source coordinates (src_points) to a txt file is maintained.

After setting the points, press the 's' key to save the BEV parameters to npz and txt files.
"""

import cv2
import numpy as np
import argparse
import os

# Global variable: 4 points selected from the source image
src_points = []
max_points = 4

def parse_args():
    parser = argparse.ArgumentParser(description="BEV Parameter Setup Utility")
    parser.add_argument('--source', type=str, default='2',
                        help='Video/camera source. A number (e.g., 0, 1, ...) for a webcam, or a file path for a video or image.')
    parser.add_argument('--warp-width', type=int, default=640,
                        help='Width of the BEV result image (default 640)')
    parser.add_argument('--warp-height', type=int, default=640,
                        help='Height of the BEV result image (default 640)')
    parser.add_argument('--out-npz', type=str, default='bev_params_5.npz',
                        help='Filename for the output NPZ parameter file')
    parser.add_argument('--out-txt', type=str, default='selected_bev_src_points_manual.txt',
                        help='Filename for the output TXT coordinate file')
    return parser.parse_args()

def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse click events to receive 4 points in sequence.
    """
    global src_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < max_points:
            src_points.append((x, y))
            point_order = ["Left-Bottom", "Right-Bottom", "Left-Top", "Right-Top"]
            current_point_index = len(src_points) - 1
            print(f"[INFO] Added {point_order[current_point_index]} point: ({x}, {y}) ({len(src_points)}/{max_points})")

            if len(src_points) == max_points:
                print("[INFO] All 4 points selected. Press 's' to save or 'r' to reset.")
        else:
            print("[WARNING] All 4 points have already been selected. Press 'r' to reset or 's' to save.")


def main():
    global src_points  # Declare use of global variable
    args = parse_args()
    source = args.source
    warp_w, warp_h = args.warp_width, args.warp_height

    is_image = False
    cap = None
    static_img = None
    
    # Use current_frame_width as a local variable within the main function
    current_frame_width = 0

    if source.isdigit():
        cap_idx = int(source)
        cap = cv2.VideoCapture(cap_idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera({cap_idx}).")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        current_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Real-time video mode via webcam ({current_frame_width}x{frame_height_cam})")
    else:
        ext = os.path.splitext(source)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        if ext in image_extensions:
            static_img = cv2.imread(source)
            if static_img is None:
                print(f"[ERROR] Could not open image file({source}).")
                return
            is_image = True
            current_frame_width = static_img.shape[1]
            print(f"[INFO] Single image mode via image file ({source}, {static_img.shape[1]}x{static_img.shape[0]})")
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[ERROR] Could not open video file({source}).")
                return
            current_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[INFO] Video mode via video file ({source}, {current_frame_width}x{frame_height_vid})")

    dst_points_default = np.float32([
        [0,       warp_h],    # Bottom-left
        [warp_w,  warp_h],    # Bottom-right
        [0,       0],         # Top-left
        [warp_w,  0]          # Top-right
    ])

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original", mouse_callback)

    print("\n[INFO] Select 4 points from the original image with a left mouse click.")
    print("       Click order: 1. Left-Bottom -> 2. Right-Bottom -> 3. Left-Top -> 4. Right-Top")
    print("       'r' key: Reset (clears all selected points)")
    print("       's' key: Save BEV parameters and exit")
    print("       'q' key: Exit (without saving)\n")

    while True:
        if is_image:
            frame = static_img.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                if cap and cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("[INFO] Reached the end of the video. Using the last frame.")
                    if frame is None:
                         print("[ERROR] Could not retrieve the last frame. Exiting.")
                         break
                else:
                    print("[WARNING] Failed to read frame or video ended -> Exiting")
                    break
        
        disp = frame.copy()
        point_labels = ["1 (L-Bot)", "2 (R-Bot)", "3 (L-Top)", "4 (R-Top)"]
        for i, pt in enumerate(src_points):
            cv2.circle(disp, pt, 5, (0, 255, 0), -1)
            label = point_labels[i] if i < len(point_labels) else f"{i+1}"
            cv2.putText(disp, label, (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(src_points) == 4:
             cv2.polylines(disp, [np.array(src_points, dtype=np.int32)], isClosed=True, color=(0,0,255), thickness=2)

        cv2.imshow("Original", disp)

        bev_result = np.zeros((warp_h, warp_w, 3), dtype=np.uint8)
        if len(src_points) == 4:
            # The point selection order is 'Left-Bottom, Right-Bottom, Left-Top, Right-Top',
            # which matches the order of dst_points_default.
            src_np = np.float32(src_points)
            M = cv2.getPerspectiveTransform(src_np, dst_points_default)
            bev_result = cv2.warpPerspective(frame, M, (warp_w, warp_h))
        cv2.imshow("BEV", bev_result)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' key pressed -> Exiting (without saving)")
            break
        elif key == ord('r'):
            print("[INFO] 'r' key pressed -> Resetting 4-point coordinates")
            src_points = []
        elif key == ord('s'):
            if len(src_points) < 4:
                print("[WARNING] Saving is only possible after selecting all 4 points.")
            else:
                print("[INFO] 's' key pressed -> Saving BEV parameters and exiting")
                out_file_npz = args.out_npz
                out_file_txt = args.out_txt

                src_arr = np.float32(src_points)
                dst_arr = dst_points_default

                np.savez(out_file_npz,
                         src_points=src_arr,
                         dst_points=dst_arr,
                         warp_w=warp_w,
                         warp_h=warp_h)
                print(f"[INFO] BEV parameters saved successfully to '{out_file_npz}'.")

                try:
                    with open(out_file_txt, 'w') as f:
                        f.write("# Selected BEV Source Points (x, y) for original image\n")
                        f.write("# Order: Left-Bottom, Right-Bottom, Left-Top, Right-Top\n")
                        for i, point in enumerate(src_points):
                            f.write(f"{point[0]}, {point[1]} # {point_labels[i]}\n")
                    print(f"[INFO] Selected coordinates saved successfully to '{out_file_txt}'.")
                except Exception as e:
                    print(f"[ERROR] An error occurred while saving the TXT file: {e}")
                break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] bev_utils.py terminated.")

if __name__ == '__main__':
    main()