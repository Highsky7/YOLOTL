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
- When a left point is selected, the right point is automatically generated symmetrically across the x-axis.
- Added functionality to save the 4 selected source coordinates (src_points) to a txt file.

After setting the points, press the 's' key to save the BEV parameters to npz and txt files.
"""

import cv2
import numpy as np
import argparse
import os

# Global variable: 4 points selected from the source image
src_points = []
max_points = 4
current_frame_width = 0 # Global variable to store the current frame width

def parse_args():
    parser = argparse.ArgumentParser(description="BEV Parameter Setup Utility")
    parser.add_argument('--source', type=str, default='2',
                        help='Video/camera source. A number (e.g., 0, 1, ...) for a webcam, or a file path for a video or image.')
    parser.add_argument('--warp-width', type=int, default=640,
                        help='Width of the BEV result image (default 640)')
    parser.add_argument('--warp-height', type=int, default=640,
                        help='Height of the BEV result image (default 640)')
    parser.add_argument('--out-npz', type=str, default='bev_params_3.npz',
                        help='Filename for the output NPZ parameter file')
    parser.add_argument('--out-txt', type=str, default='selected_bev_src_points_3.txt',
                        help='Filename for the output TXT coordinate file')
    return parser.parse_args()

def mouse_callback(event, x, y, flags, param):
    global src_points, current_frame_width
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_frame_width == 0:
            print("[WARNING] Frame width has not been set yet. Please wait for the frame to be displayed.")
            return

        if len(src_points) < max_points:
            if len(src_points) == 0: # First click (bottom-left point)
                src_points.append((x, y))
                # Automatically add the symmetrical point (bottom-right point)
                symmetric_x = current_frame_width - 1 - x
                src_points.append((symmetric_x, y))
                print(f"[INFO] Added bottom-left point: ({x}, {y})")
                print(f"[INFO] Auto-added bottom-right point: ({symmetric_x}, {y}) (Total {len(src_points)}/4)")
            elif len(src_points) == 2: # Third click (top-left point)
                src_points.append((x, y))
                # Automatically add the symmetrical point (top-right point)
                symmetric_x = current_frame_width - 1 - x
                src_points.append((symmetric_x, y))
                print(f"[INFO] Added top-left point: ({x}, {y})")
                print(f"[INFO] Auto-added top-right point: ({symmetric_x}, {y}) (Total {len(src_points)}/4)")
                print("[INFO] All 4 points selected. Press 's' to save or 'r' to reset.")
            else:
                # This case occurs when trying to manually click the second or fourth point (does not happen in the current logic)
                print(f"[INFO] {len(src_points)} points selected. Please select the next point.")

        else:
            print("[WARNING] All 4 points are already registered. Press 'r' to reset or 's' to save.")

def main():
    global src_points, current_frame_width  # Declare use of global variables
    args = parse_args()
    source = args.source
    warp_w, warp_h = args.warp_width, args.warp_height

    is_image = False
    cap = None
    static_img = None

    if source.isdigit():
        cap_idx = int(source)
        cap = cv2.VideoCapture(cap_idx, cv2.CAP_V4L2)  # Open camera using V4L2
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera({cap_idx}).")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        frame_width_cam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        current_frame_width = frame_width_cam # Set initial frame width
        print(f"[INFO] Real-time video mode via webcam ({frame_width_cam}x{frame_height_cam})")
    else:
        ext = os.path.splitext(source)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        if ext in image_extensions:
            static_img = cv2.imread(source)
            if static_img is None:
                print(f"[ERROR] Could not open image file({source}).")
                return
            is_image = True
            current_frame_width = static_img.shape[1] # Set image width
            print(f"[INFO] Single image mode via image file ({source}, {static_img.shape[1]}x{static_img.shape[0]})")
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[ERROR] Could not open video file({source}).")
                return
            frame_width_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            current_frame_width = frame_width_vid # Set initial frame width
            print(f"[INFO] Video mode via video file ({source}, {frame_width_vid}x{frame_height_vid})")

    dst_points_default = np.float32([
        [0,       warp_h],    # Bottom-left
        [warp_w,  warp_h],    # Bottom-right
        [0,       0],         # Top-left
        [warp_w,  0]          # Top-right
    ])

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original", mouse_callback)

    print("[INFO] Select points from the original image with a left mouse click.")
    print("      1. Select bottom-left point -> bottom-right point is auto-generated")
    print("      2. Select top-left point -> top-right point is auto-generated")
    print("      'r' key: Reset (re-select all 4 points)")
    print("      's' key: Save BEV parameters and exit")
    print("      'q' key: Exit (without saving)")

    while True:
        if is_image:
            frame = static_img.copy()
            # In image mode, current_frame_width is already set
        else:
            ret, frame = cap.read()
            if not ret:
                if cap and cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("[INFO] Reached the end of the video. Using the last frame.")
                    # At the end of the video, we can exit the loop or keep using the last frame.
                    # Here, we maintain the loop to allow the user to press 's' or 'q'.
                    # Set a flag or use a copy of the frame to avoid trying to read a new frame.
                    if frame is None: # Exit if even the last frame is not available
                         print("[ERROR] Could not retrieve the last frame. Exiting.")
                         break
                else:
                    print("[WARNING] Failed to read frame or video ended -> Exiting")
                    break
            if frame is not None and current_frame_width != frame.shape[1] : # To handle potential dynamic resolution changes
                current_frame_width = frame.shape[1]

        disp = frame.copy()
        # Display numbers according to the point selection order (bottom-left-1, bottom-right-2, top-left-3, top-right-4)
        point_labels = ["1 (L-Bot)", "2 (R-Bot)", "3 (L-Top)", "4 (R-Top)"]
        for i, pt in enumerate(src_points):
            cv2.circle(disp, pt, 5, (0, 255, 0), -1)
            label = point_labels[i] if i < len(point_labels) else f"{i+1}"
            cv2.putText(disp, label, (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(src_points) == 4: # If all 4 points are selected, connect them with a polygon
             cv2.polylines(disp, [np.array(src_points, dtype=np.int32)], isClosed=True, color=(0,0,255), thickness=2)

        cv2.imshow("Original", disp)

        bev_result = np.zeros((warp_h, warp_w, 3), dtype=np.uint8)
        if len(src_points) == 4:
            # src_points order: user click (L-Bot), auto-gen (R-Bot), user click (L-Top), auto-gen (R-Top)
            # The order for src_np for getPerspectiveTransform should be L-Bot, R-Bot, L-Top, R-Top.
            # The current storage order of src_points matches this (0: L-Bot, 1: R-Bot, 2: L-Top, 3: R-Top)
            src_np = np.float32(src_points)
            M = cv2.getPerspectiveTransform(src_np, dst_points_default)
            bev_result = cv2.warpPerspective(frame, M, (warp_w, warp_h))
        cv2.imshow("BEV", bev_result)

        key = cv2.waitKey(30) & 0xFF # Adjust waitKey value considering video playback speed (e.g., 30ms)
        if key == ord('q'):
            print("[INFO] 'q' key pressed -> Exiting (without saving)")
            break
        elif key == ord('r'):
            print("[INFO] 'r' key pressed -> Resetting 4-point coordinates")
            src_points = []
        elif key == ord('s'):
            if len(src_points) < 4:
                print("[WARNING] Less than 4 points selected. Please click 2 points to set all 4 and try again.")
            else:
                print("[INFO] 's' key pressed -> Saving BEV parameters and exiting")
                out_file_npz = args.out_npz
                out_file_txt = args.out_txt

                src_arr = np.float32(src_points)
                dst_arr = dst_points_default

                # Save NPZ file (maintaining original logic)
                np.savez(out_file_npz,
                         src_points=src_arr,
                         dst_points=dst_arr,
                         warp_w=warp_w,
                         warp_h=warp_h)
                print(f"[INFO] BEV parameters saved successfully to '{out_file_npz}'.")

                # Save TXT file
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