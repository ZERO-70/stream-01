import cv2
import socket
import struct
import pickle
import time
import sys
import os
import numpy as np
import threading
import subprocess

class VideoSender:
    def __init__(self, host='localhost', port=9999, audio_port=9998):
        self.host = host
        self.port = port
        self.audio_port = audio_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Audio streaming socket
        self.audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.audio_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Audio streaming parameters
        self.chunk_size = 1024
        self.streaming = False
        
    def start_server(self):
        """Start the video and audio servers and wait for client connections"""
        try:
            # Start video server
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"Video server started on {self.host}:{self.port}")
            
            # Start audio server
            self.audio_socket.bind((self.host, self.audio_port))
            self.audio_socket.listen(1)
            print(f"Audio server started on {self.host}:{self.audio_port}")
            
            print("Waiting for client connections...")
            
            # Accept video connection
            self.client_socket, self.client_address = self.socket.accept()
            print(f"Client connected to video stream from {self.client_address}")
            
            # Accept audio connection
            self.audio_client_socket, self.audio_client_address = self.audio_socket.accept()
            print(f"Client connected to audio stream from {self.audio_client_address}")
            
        except Exception as e:
            print(f"Error starting server: {e}")
            sys.exit(1)
    
    def extract_audio(self, video_path):
        """Extract audio from video file using FFmpeg"""
        print("Extracting audio from video file...")
        audio_path = 'temp_audio.raw'
        
        try:
            # Use FFmpeg to extract raw audio data
            cmd = [
                'ffmpeg', '-i', video_path,
                '-f', 's16le',  # 16-bit signed little-endian
                '-ac', '2',     # stereo
                '-ar', '44100', # 44.1kHz sample rate
                '-y',           # overwrite output files
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Audio extracted to {audio_path}")
                return audio_path
            else:
                print(f"FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def stream_audio(self, audio_path):
        """Stream raw audio data to the client"""
        try:
            # Send audio parameters to client
            params = {
                'channels': 2,
                'rate': 44100,
                'chunk_size': self.chunk_size
            }
            params_data = pickle.dumps(params)
            self.audio_client_socket.sendall(struct.pack("L", len(params_data)))
            self.audio_client_socket.sendall(params_data)
            
            print(f"Starting audio stream: 2 channels, 44100Hz")
            
            # Stream raw audio data
            with open(audio_path, 'rb') as f:
                while self.streaming:
                    try:
                        # Read chunk of audio data
                        data = f.read(self.chunk_size * 4)  # 4 bytes per sample (2 channels * 2 bytes)
                        
                        if len(data) == 0:
                            # Loop audio when it ends
                            f.seek(0)
                            data = f.read(self.chunk_size * 4)
                        
                        if len(data) > 0:
                            # Send audio chunk size and data
                            self.audio_client_socket.sendall(struct.pack("L", len(data)))
                            self.audio_client_socket.sendall(data)
                            
                            # Control audio playback speed (roughly 44100 samples per second)
                            time.sleep(self.chunk_size / 44100.0)
                        
                    except (socket.error, ConnectionResetError):
                        print("Client disconnected from audio stream")
                        break
            
        except Exception as e:
            print(f"Error streaming audio: {e}")
    
    def stream_video(self, video_path):
        """Stream video frames and audio to the client"""
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found!")
            print("Please place a video.mp4 file in the same directory as this script.")
            sys.exit(1)
        
        # Extract audio from video
        audio_path = self.extract_audio(video_path)
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            sys.exit(1)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.033  # Default to 30 fps if can't detect
        
        print(f"Video FPS: {fps}")
        print(f"Frame delay: {frame_delay:.3f} seconds")
        print("Starting video and audio streams...")
        
        # Flag to control audio streaming
        self.streaming = True
        
        # Start audio streaming in a separate thread
        audio_thread = threading.Thread(target=self.stream_audio, args=(audio_path,), daemon=True)
        audio_thread.start()
        
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
                    print("Client disconnected from video stream")
                    break
                    
        except KeyboardInterrupt:
            print("\nStreaming stopped by user")
        finally:
            self.streaming = False
            cap.release()
            # Clean up temporary audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"Removed temporary audio file: {audio_path}")
                except:
                    pass
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.streaming = False
            if hasattr(self, 'client_socket'):
                self.client_socket.close()
            if hasattr(self, 'audio_client_socket'):
                self.audio_client_socket.close()
            self.socket.close()
            self.audio_socket.close()
            print("Video and audio servers closed")
        except Exception as e:
            print(f"Error during cleanup: {e}")
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