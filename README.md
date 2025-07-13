# YOLOTL: YOLO-based Top-view Lane Segmentation and Steering Control

YOLOTL is a ROS-based project for real-time lane detection and steering angle calculation using a pre-trained YOLO model for top-view lane segmentation. It provides a robust solution for developing autonomous driving capabilities, with a focus on accuracy and performance.

![example](example.mp4)
![example2](example2.mp4)

## Features

*   **Lane Detection:** Utilizes a YOLOv8 model to perform semantic segmentation on a Bird's-Eye View (BEV) transformed image, identifying lane markings with high precision.
*   **Steering Angle Calculation:** Implements a Pure Pursuit algorithm to calculate the required steering angle based on the detected lane center, enabling autonomous lane following.
*   **ROS Integration:** Seamlessly integrates with the Robot Operating System (ROS), subscribing to camera image topics and publishing steering commands.
*   **Standalone Demo:** Includes a demo script that can run independently of ROS, using a video file as input for quick testing and visualization.
*   **Dynamic Lookahead:** The Pure Pursuit algorithm features a dynamic lookahead distance that adjusts based on the vehicle's throttle, improving stability at varying speeds.
*   **BEV Transformation:** Includes scripts for both automatic and manual calibration of the BEV transformation, allowing for easy adaptation to different camera setups.

## Dependencies

*   Python 3.8+
*   PyTorch
*   OpenCV
*   NumPy
*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
*   ROS (for ROS integration)
    *   `rospy`
    *   `cv_bridge`
    *   `sensor_msgs`
    *   `std_msgs`
    *   `nav_msgs`
    *   `geometry_msgs`
    *   `visualization_msgs`
    *   `tf2_ros`
    *   `tf2_geometry_msgs`

You can install the required Python packages using pip:

```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy ultralytics
```

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/YOLOTL.git
    ```

2.  **Build the ROS package:**

    ```bash
    cd /path/to/your/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

## Usage

### Standalone Demo (without ROS)

To run the lane detection and steering angle calculation on a video file, use the `demo.py` script:

```bash
python src/camera_lane_segmentation/scripts/demo.py --weights /path/to/your/weights.pt --source /path/to/your/video.mp4 --param-file /path/to/your/bev_params.npz
```

*   `--weights`: Path to the pre-trained YOLO model weights.
*   `--source`: Path to the input video file or camera index (e.g., `0` for webcam).
*   `--param-file`: Path to the BEV transformation parameters file.

### ROS Node

To launch the ROS node for real-time lane following:

1.  **Start your camera node:**

    ```bash
    roslaunch your_camera_package your_camera.launch
    ```

2.  **Run the lane follower node:**

    ```bash
    rosrun camera_lane_segmentation demo_with_ros.py --weights /path/to/your/weights.pt --param-file /path/to/your/bev_params.npz
    ```

The node will subscribe to the `/usb_cam/image_raw` topic and publish the steering angle to the `/auto_steer_angle_lane` topic.

## Configuration

*   **BEV Parameters:** The BEV transformation is defined by a `.npz` file containing the source and destination points. You can generate your own parameters using the provided calibration scripts in `src/camera_lane_segmentation/scripts/utils`.
*   **Model Weights:** The pre-trained YOLO model weights are required for lane segmentation. You can train your own model or use a pre-trained one.

## Citation

This project uses the YOLOv8 model from Ultralytics. If you use this project in your research, please cite the original YOLOv8 paper.

## License

The license for this project is currently set to `TODO`. Please choose an appropriate open-source license for your project.