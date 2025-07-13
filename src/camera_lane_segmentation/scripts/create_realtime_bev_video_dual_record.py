#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
import time

# =================================================================
# (Note: This script requires the LoadCamera class to be defined in the utils.utils module.)
# =================================================================
from utils.utils import LoadCamera
# =================================================================


def do_bev_transform(image, bev_param_file):
    """
    Performs Bird's-Eye-View (BEV) transformation on the input image.
    (This function is identical to the original and has not been modified.)
    """
    if not Path(bev_param_file).exists():
        print(f"[ERROR] BEV parameter file not found: {bev_param_file}")
        sys.exit(1)

    params = np.load(bev_param_file)
    src_points = params['src_points']
    dst_points = params['dst_points']
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev_image = cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)
    
    return bev_image

def make_parser():
    """
    Parses arguments for script execution.
    """
    parser = argparse.ArgumentParser(description="A script that records real-time camera video as both original and BEV (Bird's-Eye-View) transformations, each with H.264 codec.")
    parser.add_argument('--source', type=str,
                        default='2',
                        help='Camera index. Typically "0" for built-in, "1" for external, etc. E.g., 0')
    parser.add_argument('--img-size', type=int, default=640, help='Image resolution to process (passed to LoadCamera class)')
    parser.add_argument('--param-file', type=str, default='/home/highsky/dol_dol_dol_ws/bev_params_y_5.npz', help='Path to the BEV parameter file. E.g., ./bev_params_1.npz')
    parser.add_argument('--output-dir', type=str, default='runs/bev_output', help='Folder where the resulting videos will be saved')
    return parser

def bev_transform_and_save_realtime(opt):
    """
    Main logic: Reads real-time video using LoadCamera, and records both original and BEV transformed frames
    separately with H.264 codec. Press 'q' to stop recording and exit the program.
    """
    try:
        dataset = LoadCamera(opt.source, img_size=opt.img_size)
    except Exception as e:
        print(f"[ERROR] Could not open camera: {opt.source}")
        print(f"  - Details: {e}")
        return

    # MODIFIED: Declare two writer objects for two video recordings.
    writer_original = None
    writer_bev = None
    
    # Set save paths
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # MODIFIED: Create separate save paths for the original and BEV videos.
    output_path_original = output_dir / f"original_output_{timestamp}_h264.mp4"
    output_path_bev = output_dir / f"bev_output_{timestamp}_h264.mp4"
    
    print("=====================================================")
    # MODIFIED: Update the info message to indicate that two files will be saved.
    print(f"  Starting simultaneous H.264 recording of real-time original and BEV transformed video")
    print(f"  - Input Camera: {opt.source}")
    print(f"  - Processing Resolution: {opt.img_size} (using LoadCamera)")
    print(f"  - BEV Parameters: {opt.param_file}")
    print(f"  - Original Video Save Path: {output_path_original}")
    print(f"  - BEV Video Save Path: {output_path_bev}")
    print("\n  Press 'q' in the result window to stop recording and exit.")
    print("=====================================================")

    params = np.load(opt.param_file)
    output_w = int(params['warp_w'])
    output_h = int(params['warp_h'])

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        
        # Initialize VideoWriters on the first frame
        # MODIFIED: Execute when both writers have not been initialized.
        if writer_original is None and writer_bev is None:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                print("[WARNING] Could not get FPS from camera. Setting to 30.")
                fps = 30
            
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use H.264 codec

            # ADDED: Get the width and height of the original video.
            h, w, _ = im0s.shape
            
            # ADDED: Initialize VideoWriter for the original video.
            writer_original = cv2.VideoWriter(str(output_path_original), fourcc, fps, (w, h))
            
            # MODIFIED: Initialize VideoWriter for the BEV video.
            writer_bev = cv2.VideoWriter(str(output_path_bev), fourcc, fps, (output_w, output_h))

            # MODIFIED: Check if both writers opened successfully.
            if not writer_original.isOpened() or not writer_bev.isOpened():
                print("\n[ERROR] Could not create video file with H.264 codec.")
                print("  - Please check if the H.264 codec is installed on your system.")
                print("  - Alternatively, try other FourCC codes (e.g., 'avc1', 'X264', 'mp4v').")
                break # Stop the loop

        input_frame = im0s
        bev_frame = do_bev_transform(input_frame, opt.param_file)

        # ADDED: Record the original frame.
        if writer_original is not None and writer_original.isOpened():
            writer_original.write(input_frame)
            
        # MODIFIED: Record the BEV frame.
        if writer_bev is not None and writer_bev.isOpened():
            writer_bev.write(bev_frame)

        cv2.imshow('Real-time Input (Recording...)', input_frame)
        cv2.imshow('BEV Transformed Video (Recording...)', bev_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' was pressed. Stopping recording.")
            break
    
    # Release resources
    if isinstance(dataset, LoadCamera) and dataset.cap:
        dataset.cap.release()
        
    # ADDED: Release the original video writer.
    if writer_original is not None and writer_original.isOpened():
        writer_original.release()
        
    # MODIFIED: Release the BEV video writer.
    if writer_bev is not None and writer_bev.isOpened():
        writer_bev.release()

    # MODIFIED: Update the completion message.
    print("\n[COMPLETE] Recording of original and BEV videos is complete.")
    print(f"You can find the results at the following paths:")
    print(f"  - Original: '{output_path_original}'")
    print(f"  - BEV: '{output_path_bev}'")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    
    bev_transform_and_save_realtime(args)