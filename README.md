# Object Detection and Obstacle Avoidance System with Raspberry Pi

![Raspberry Pi](https://github.com/burntbits/IGNAVI/assets/69628550/89c37f1f-55c2-41e5-838c-423200237e1f)

## Overview

The Object Detection and Obstacle Avoidance System is implemented using a Raspberry Pi and a Pi camera module. The system captures images, processes them on a remote server to create depth information, and then provides audio feedback about nearby objects and their distances.

## Implementation

- Monocular Depth Estimation: The NeWCRFs algorithm, implemented in IGNAVI, enables accurate depth estimation from monocular images. This feature enhances obstacle detection capabilities, allowing users to navigate their surroundings more confidently.

## Repository

The project repository can be found at: [https://github.com/aliyun/NeWCRFs](https://github.com/aliyun/NeWCRFs)

Please refer to the repository for the latest updates, code, and detailed documentation related to the NeWCRFs algorithm implementation.

### Setup

1. Set up the Raspberry Pi with Raspberry Pi OS and configure the camera module using raspi-config.

2. Connect the Raspberry Pi and the remote processing server to the same network by configuring the SSID and password.

3. Use SSH to connect securely to the Raspberry Pi remotely.

4. Set up an FTP server, such as vsftpd, on the Raspberry Pi for sending data to the processing server.

5. Install the eSpeak module and check if it is working by running a simple script.

### Working

1. Stream video from the Pi camera using the raspivid software. The video is streamed via TCP using a specified resolution, frame rate, and output port.

2. Use the eSpeak module to read out the file containing the audio feedback.


## Project Workflow

The video from the camera is streamed using TCP/IP, a reliable network protocol that ensures accurate delivery of data. The TCP/IP protocol handles the quality and precise delivery of the data stream. It achieves reliability through positive acknowledgement and retransmission.

The input frames are processed to create depth images, and a contour is drawn around the objects to be avoided. The depth information is then used to determine the distance of the objects from the user. The system provides text feedback about the location of the obstacle.

![Streaming using TCP](https://github.com/burntbits/IGNAVI/assets/69628550/ae49624b-7520-4178-96c3-b9707c9d63c0)

## Results

The system successfully detects objects and provides accurate distance information. The final output includes a depth image with contours around the objects and text feedback indicating the location of the obstacle.

![contours img](https://github.com/burntbits/IGNAVI/assets/69628550/72772b9b-334c-4e1e-8a58-1bde9dcdc67c)


![Final Output](https://github.com/burntbits/IGNAVI/assets/69628550/08801372-9588-43d4-b746-a7dd1919aac7)

## Future Work

Future work on this project may include:

- Enhancing the accuracy of object detection and distance measurement.
- Implementing real-time processing on the Raspberry Pi for faster feedback.
- Integrating additional sensors for improved obstacle detection.
- Exploring alternative audio feedback methods


