#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

# =================================================================
# ★★★ KEY CHANGE 1: Directly import the LoadImages class from the original utils file ★★★
# =================================================================
from utils.utils import LoadImages
# =================================================================


def do_bev_transform(image, bev_param_file):
    """
    Performs Bird's-Eye-View (BEV) transformation on the input image.
    (This function has not been changed.)
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
    (Added img-size argument for consistency with the original code)
    """
    parser = argparse.ArgumentParser(description="A script to transform and save a video file into a BEV (Bird's-Eye-View) video.")
    parser.add_argument('--source', type=str,
                        default='/home/highsky/Downloads/2025-07-09-161352.mp4',
                        help='Path to the source video file to be transformed. E.g., /path/to/video.mp4')
    parser.add_argument('--img-size', type=int, default=640, help='Image resolution to process (passed to LoadImages class)')
    parser.add_argument('--param-file', type=str, default='./bev_params_y_5.npz', help='Path to the BEV parameter file. E.g., ./bev_params_1.npz')
    parser.add_argument('--nosave', action='store_true', help='Do not save the resulting video')
    parser.add_argument('--output-dir', type=str, default='runs/bev_output', help='Folder where the resulting video will be saved')
    return parser

def bev_transform_and_save(opt):
    """
    Main logic: Reads a video using LoadImages, transforms each frame, and saves the result.
    """
    source_path = Path(opt.source)
    if not source_path.is_file():
        print(f"[ERROR] Video file not found: {source_path}")
        return

    # =================================================================
    # ★★★ KEY CHANGE 2: Use LoadImages class instead of cv2.VideoCapture ★★★
    # =================================================================
    dataset = LoadImages(opt.source, img_size=opt.img_size)
    # =================================================================

    writer = None
    output_path = "" # Declare outside to resolve writer scope issues

    # If --nosave option is not used, set the save path
    if not opt.nosave:
        output_dir = Path(opt.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"bev_{source_path.name}"
        print("=====================================================")
        print(f"  Starting video transformation (Save Mode)")
        print(f"  - Source Video: {opt.source}")
        print(f"  - Processing Resolution: {opt.img_size}x{opt.img_size} (aspect ratio maintained, using LoadImages)")
        print(f"  - BEV Parameters: {opt.param_file}")
        print(f"  - Save Path: {output_path}")
        print("=====================================================")
    else:
        print("=====================================================")
        print(f"  Starting video transformation (Real-time View Mode, No Save)")
        print("=====================================================")

    params = np.load(opt.param_file)
    output_w = int(params['warp_w'])
    output_h = int(params['warp_h'])

    # =================================================================
    # ★★★ KEY CHANGE 3: Use a loop structure optimized for LoadImages ★★★
    # =================================================================
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        
        # Initialize VideoWriter on the first frame
        if not opt.nosave and writer is None:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            # =================================================================
            # ★★★ Change to H.264 codec ★★★
            # Use 'avc1' or 'h264' instead of 'mp4v'. 'avc1' has better compatibility with mp4.
            # =================================================================
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_w, output_h))
        
        # `im0s` is already a frame resized while maintaining aspect ratio
        input_frame = im0s

        # Perform BEV transformation
        bev_frame = do_bev_transform(input_frame, opt.param_file)

        if writer is not None:
            writer.write(bev_frame)

        # View real-time results
        cv2.imshow('Input (from LoadImages)', input_frame)
        cv2.imshow('BEV Transformed Video', bev_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display progress
        progress = ((frame_idx + 1) / len(dataset)) * 100
        print(f"\r  Progress: {frame_idx + 1}/{len(dataset)} ({progress:.2f}%)", end="")
    
    # Release resources
    if vid_cap:
        vid_cap.release()
    if writer is not None:
        writer.release()
        print("\n\n[COMPLETE] BEV video transformation and saving with H.264 codec is complete.")
        print(f"You can find the result at '{output_path}'.")
    else:
        print("\n\n[COMPLETE] Video processing has finished.")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    
    bev_transform_and_save(args)