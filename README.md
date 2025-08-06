# Local Video Streaming with Python Sockets

This project implements a local video streaming system using Python sockets. It consists of two scripts that work together to stream video frames over TCP sockets on localhost.

## Features

- **Real-time video streaming** using TCP sockets
- **JPEG encoding** for efficient frame transmission
- **Automatic video looping** when the video ends
- **FPS monitoring** and display
- **NSFW detection** with automatic blurring (receiver only)
- **Graceful connection handling** and cleanup

## Files

- `sender.py` - Reads video file and streams frames over TCP
- `receiver.py` - Receives and displays video stream using OpenCV with NSFW detection
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Prerequisites

1. **Python 3.7+** installed on your system
2. **A video file** named `video.mp4` in the same directory as the scripts
3. **OpenCV**, **NumPy**, and **Transformers** (installed via requirements.txt)

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your `video.mp4` file in the same directory as the scripts.

## Usage

### Step 1: Start the Sender

Open a terminal/command prompt and run:
```bash
python sender.py
```

The sender will:
- Start a TCP server on `localhost:9999`
- Wait for a client connection
- Begin streaming video frames once connected

### Step 2: Start the Receiver

Open another terminal/command prompt and run:
```bash
python receiver.py
```

The receiver will:
- Connect to the sender
- Display the video stream in a window
- Show FPS and frame count overlay
- **Check for NSFW content every second**
- **Automatically blur NSFW frames**
- **Display NSFW status and confidence**

### Controls

- **Press 'q'** in the video window to quit the receiver
- **Ctrl+C** in either terminal to stop the respective script

## NSFW Detection Features

The receiver includes advanced NSFW detection:

- **Real-time Analysis**: Checks every frame every second
- **Automatic Blurring**: Heavily blurs NSFW content with dark overlay
- **Status Display**: Shows NSFW/SFW status with confidence scores
- **Model Integration**: Uses Hugging Face's NSFW detection model
- **Error Handling**: Gracefully handles model loading failures

### NSFW Detection Status

- **Green text**: SFW content detected
- **Red text**: NSFW content detected (frame will be blurred)
- **Gray text**: NSFW detection disabled (model unavailable)

## How It Works

### Sender (`sender.py`)
1. Creates a TCP server socket on localhost:9999
2. Waits for client connection
3. Reads video frames using OpenCV
4. Encodes each frame as JPEG (80% quality)
5. Sends frame size followed by frame data
6. Maintains real-time playback speed
7. Loops video when it ends

### Receiver (`receiver.py`)
1. Connects to the sender's TCP server
2. Receives frame size and data
3. Decodes JPEG frames back to OpenCV format
4. **Checks for NSFW content every second**
5. **Applies blur to NSFW frames**
6. Displays frames in real-time with status overlay
7. Shows FPS, frame count, and NSFW status
8. Handles connection errors gracefully

## Technical Details

- **Protocol**: TCP sockets for reliable frame delivery
- **Encoding**: JPEG compression (80% quality) to reduce bandwidth
- **Framing**: Each frame is prefixed with its size (4 bytes)
- **Serialization**: Pickle for Python object serialization
- **Display**: OpenCV window with real-time FPS counter
- **NSFW Detection**: Transformers pipeline with Falconsai/nsfw_image_detection model
- **Blurring**: Gaussian blur + dark overlay for NSFW content

## Troubleshooting

### Common Issues

1. **"Video file not found"**
   - Make sure `video.mp4` exists in the same directory as `sender.py`

2. **"Connection refused"**
   - Start `sender.py` before `receiver.py`
   - Check if port 9999 is already in use

3. **"Module not found"**
   - Install dependencies: `pip install -r requirements.txt`

4. **"NSFW detection not available"**
   - Check your internet connection (first run downloads the model)
   - The system will work without NSFW detection
   - Model download may take time on first run

5. **Poor performance**
   - Reduce video resolution or frame rate
   - Lower JPEG quality in `sender.py` (line with `IMWRITE_JPEG_QUALITY`)
   - NSFW detection adds some latency (checks every second)

### Performance Tips

- Use smaller video files for better performance
- Ensure both scripts run on the same machine
- Close other applications to free up system resources
- Consider reducing video resolution if streaming is slow
- NSFW detection requires model download on first run

## Customization

### Change Port
Edit the port number in both scripts:
```python
# In sender.py and receiver.py
self.port = 9999  # Change to any available port
```

### Change Video Quality
In `sender.py`, modify the JPEG quality:
```python
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # 80% quality
```

### Change Video File
In `sender.py`, change the video path:
```python
sender.stream_video('your_video.mp4')  # Change filename
```

### Adjust NSFW Detection
In `receiver.py`, modify detection settings:
```python
self.nsfw_check_interval = 1.0  # Check every 1 second
# or
self.nsfw_check_interval = 0.5  # Check every 0.5 seconds
```

### Change Blur Strength
In `receiver.py`, modify the blur function:
```python
def blur_frame(self, frame, blur_strength=50):  # Increase for more blur
```

## System Requirements

- **OS**: Windows, macOS, or Linux
- **Python**: 3.7 or higher
- **Memory**: At least 2GB RAM recommended
- **Storage**: Enough space for your video file
- **Internet**: Required for NSFW model download (first run only)

## License

This project is open source and available under the MIT License. # stream-01
# stream-01
# stream-01
