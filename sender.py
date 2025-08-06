import cv2
import socket
import struct
import pickle
import time
import sys
import os

class VideoSender:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
    def start_server(self):
        """Start the server and wait for client connection"""
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"Server started on {self.host}:{self.port}")
            print("Waiting for client connection...")
            
            self.client_socket, self.client_address = self.socket.accept()
            print(f"Client connected from {self.client_address}")
            
        except Exception as e:
            print(f"Error starting server: {e}")
            sys.exit(1)
    
    def stream_video(self, video_path):
        """Stream video frames to the client"""
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found!")
            print("Please place a video.mp4 file in the same directory as this script.")
            sys.exit(1)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            sys.exit(1)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.033  # Default to 30 fps if can't detect
        
        print(f"Video FPS: {fps}")
        print(f"Frame delay: {frame_delay:.3f} seconds")
        print("Starting video stream...")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached. Restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Encode frame as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # 80% quality
                _, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                
                # Prepare frame data
                frame_data = pickle.dumps(encoded_frame)
                frame_size = len(frame_data)
                
                # Send frame size first, then frame data
                try:
                    self.client_socket.sendall(struct.pack("L", frame_size))
                    self.client_socket.sendall(frame_data)
                    
                    frame_count += 1
                    
                    # Maintain real-time playback speed
                    elapsed_time = time.time() - start_time
                    expected_time = frame_count * frame_delay
                    
                    if elapsed_time < expected_time:
                        time.sleep(expected_time - elapsed_time)
                    
                    # Print progress every 30 frames
                    if frame_count % 30 == 0:
                        print(f"Sent {frame_count} frames")
                        
                except (socket.error, ConnectionResetError):
                    print("Client disconnected")
                    break
                    
        except KeyboardInterrupt:
            print("\nStreaming stopped by user")
        finally:
            cap.release()
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'client_socket'):
                self.client_socket.close()
            self.socket.close()
            print("Server closed")
        except:
            pass

def main():
    sender = VideoSender()
    
    try:
        sender.start_server()
        sender.stream_video('video.mp4')
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sender.cleanup()

if __name__ == "__main__":
    main() 