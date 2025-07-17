# YOLOTL: YOLO-based Top-view Lane Segmentation and Steering Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%235C3EE8.svg?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![ROS](https://img.shields.io/badge/ROS-%2322314E.svg?style=for-the-badge&logo=ROS&logoColor=white)](http://www.ros.org/)

YOLOTL is a complete system for real-time lane detection and autonomous steering control. It leverages a YOLOv8-based segmentation model, trained and optimized using the Roboflow 3.0 architecture, for high-precision semantic segmentation on a Bird's-Eye-View (BEV) image, enabling robust lane-following capabilities for robotics and autonomous vehicle applications.

| Live Demo 1 | Live Demo 2 |
| :---: | :---: |
| <img src="./YOLOTL1.gif" alt="YOLOTL1" height="400" width="auto"> | <img src="./YOLOTL2.gif" alt="YOLOTL2" height="400" width="auto"> |

## 📋 Table of Contents

*   [Getting Started](#-getting-started)
*   [Usage](#-usage)
*   [Model Zoo](#-model-zoo)
*   [Dataset](#-dataset)
*   [Features](#-features)
*   [How It Works](#-how-it-works)
*   [Citation](#-citation)
*   [License](#-license)

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   PyTorch
-   OpenCV
-   NumPy
-   Ultralytics YOLOv8 (for running inference with the Roboflow-trained model)
-   ROS (for the ROS-integrated version)
    -   `rospy`, `cv_bridge`, `sensor_msgs`, `std_msgs`, etc.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/YOLOTL.git
    cd YOLOTL
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install torch torchvision torchaudio
    pip install opencv-python numpy ultralytics
    ```

3.  **(For ROS users) Build the Catkin workspace:**
    ```bash
    cd /path/to/your/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

## 📃 Usage

You can run the project in two modes: with ROS for live integration or as a standalone script on a video file.

### 1. Live Demo with ROS

This mode is for controlling a real robot or vehicle equipped with a camera and ROS.

1.  **Launch your camera node:**
    ```bash
    # Example:
    roslaunch your_camera_package your_camera.launch
    ```

2.  **Run the lane follower node:**
    ```bash
    rosrun camera_lane_segmentation demo_with_ros.py --weights /path/to/your/weights.pt --param-file /path/to/your/bev_params.npz
    ```
    The node subscribes to `/usb_cam/image_raw` and publishes the steering angle to `/auto_steer_angle_lane`.

### 2. Standalone Demo on a Video File

This mode is for testing the algorithm on a pre-recorded video.

```bash
python src/camera_lane_segmentation/scripts/demo.py --weights /path/to/your/weights.pt --source /path/to/your/video.mp4 --param-file /path/to/your/bev_params.npz
```

**Arguments:**
-   `--weights`: Path to the pre-trained YOLO model weights (`.pt` file).
-   `--source`: Path to the input video file or a camera index (e.g., `0` for webcam).
-   `--param-file`: Path to the BEV transformation parameters (`.npz` file). You can generate your own using the calibration scripts in `src/camera_lane_segmentation/scripts/utils`.

## Model Zoo

The pre-trained model is available on Hugging Face:

| Resource | Download Link |
| :---: | :---: |
| **YOLOTL Model** | [Hugging Face](https://huggingface.co/Highsky7/YOLOTL) |

## Dataset

The dataset used for training is available on Hugging Face:

| Resource | Download Link |
| :---: | :---: |
| **Topview_Lane Raw Dataset** | [Hugging Face](https://huggingface.co/datasets/Highsky7/Topview_Lane) |
| **Topview_Lane Annotated Dataset** | [Roboflow](https://universe.roboflow.com/highsky/bev_lane) |

## Features

-   **High-Precision Lane Segmentation:** Utilizes a YOLOv8-based model trained on Roboflow for segmenting lane markings from a Bird's-Eye-View (BEV) perspective, ensuring high accuracy.
-   **Robust Steering Control:** Implements the Pure Pursuit algorithm to calculate the precise steering angle required to follow the detected lane center.
-   **Intelligent Lane Tracking:** Remembers the position of left and right lanes across frames. This provides stability through temporary occlusions (e.g., when one lane line disappears) and enables reliable differentiation between the two lanes.
-   **Adaptive Control:** Features a dynamic lookahead distance in the Pure Pursuit controller. The lookahead distance automatically adjusts based on the vehicle's speed (throttle), improving stability and smoothness across different velocities.
-   **Flexible Integration:** Offers both a seamless ROS integration for robotics projects and a standalone demo script that can run on any video file for quick testing and development.
-   **Easy Calibration:** Includes helper scripts for generating the BEV transformation parameters from a source image, making it easy to adapt to different camera setups.

## How It Works

The system follows a modular pipeline to process images and generate steering commands:

1.  **Camera Input:** Receives a raw image stream from a camera (via ROS) or a video file.
2.  **BEV Transformation:** Warps the input image into a top-down Bird's-Eye-View (BEV) perspective using pre-calibrated parameters.
3.  **Lane Segmentation:** The BEV image is fed into the trained model (a YOLOv8-based architecture from Roboflow), which outputs a binary mask of the detected lane lines.
4.  **Lane Filtering & Tracking:** The output mask is cleaned using morphological operations. The system then identifies and tracks the left and right lane lines, smoothing the results over time for stability.
5.  **Center Path Calculation:** A central path is computed based on the final positions of the left and right lanes.
6.  **Pure Pursuit Control:** The Pure Pursuit algorithm calculates the optimal steering angle to guide the vehicle along the generated center path.
7.  **Output:** The final steering angle is published to a ROS topic or displayed in the demo window.

## ©Citation

This project's model architecture is based on YOLOv8 by Ultralytics, and the model was trained and deployed using the Roboflow platform. If you use this project in your research, please consider citing both:

### YOLOv8 by Ultralytics

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *YOLO by Ultralytics*. [GitHub Repository](https://github.com/ultralytics/ultralytics).

```bibtex
@misc{yolov8,
  author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  title = {YOLO by Ultralytics},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ultralytics/ultralytics}}
}
```

### Roboflow Platform

Roboflow, Inc. (2024). *Roboflow Platform*. [Website](https://roboflow.com).

```bibtex
@misc{roboflow,
  author = {Roboflow},
  title = {Roboflow Platform},
  year = {2024},
  publisher = {Roboflow, Inc.},
  howpublished = {\url{https://roboflow.com}}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
