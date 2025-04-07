"""
Audio and Video Processing Module for Manus Clone

This module handles audio and video processing capabilities including:
- Speech recognition and text-to-speech
- Audio file manipulation and analysis
- Video processing and frame extraction
- Video editing and composition
- Media conversion and optimization
"""

import os
import json
import time
import uuid
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioVideoProcessor:
    """Audio and Video Processing class for multimedia content handling"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Audio and Video Processor
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.media_dir = config.get('MEDIA_FOLDER', os.path.join(os.path.expanduser('~'), 'manus_clone', 'media'))
        self.temp_dir = config.get('TEMP_FOLDER', os.path.join(os.path.expanduser('~'), 'manus_clone', 'temp'))
        self.ffmpeg_path = config.get('FFMPEG_PATH', 'ffmpeg')
        self.ffprobe_path = config.get('FFPROBE_PATH', 'ffprobe')
        
        # Ensure directories exist
        os.makedirs(self.media_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Check for required dependencies
        self._check_dependencies()
        
        logger.info("Audio and Video Processor initialized successfully")
    
    def _check_dependencies(self) -> None:
        """
        Check if required dependencies are installed
        """
        try:
            # Check FFmpeg
            result = subprocess.run([self.ffmpeg_path, "-version"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=False)
            if result.returncode != 0:
                logger.warning(f"FFmpeg not found at {self.ffmpeg_path}. Some video processing features may not work.")
            else:
                logger.info(f"FFmpeg found: {result.stdout.splitlines()[0] if result.stdout else 'Unknown version'}")
            
            # Check FFprobe
            result = subprocess.run([self.ffprobe_path, "-version"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=False)
            if result.returncode != 0:
                logger.warning(f"FFprobe not found at {self.ffprobe_path}. Some media analysis features may not work.")
            else:
                logger.info(f"FFprobe found: {result.stdout.splitlines()[0] if result.stdout else 'Unknown version'}")
                
        except Exception as e:
            logger.error(f"Error checking dependencies: {str(e)}")
    
    def get_media_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a media file
        
        Args:
            file_path: Path to the media file
            
        Returns:
            Dictionary with media information
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": "File not found"}
        
        try:
            # Use FFprobe to get media information
            cmd = [
                self.ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ]
            
            result = subprocess.run(cmd, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=False)
            
            if result.returncode != 0:
                logger.error(f"Error getting media info: {result.stderr}")
                return {"error": f"FFprobe error: {result.stderr}"}
            
            # Parse JSON output
            info = json.loads(result.stdout)
            
            # Extract relevant information
            media_info = {
                "filename": os.path.basename(file_path),
                "format": info.get("format", {}).get("format_name", "unknown"),
                "duration": float(info.get("format", {}).get("duration", 0)),
                "size": int(info.get("format", {}).get("size", 0)),
                "bit_rate": int(info.get("format", {}).get("bit_rate", 0)),
                "streams": []
            }
            
            # Process streams
            for stream in info.get("streams", []):
                stream_info = {
                    "index": stream.get("index", 0),
                    "codec_type": stream.get("codec_type", "unknown"),
                    "codec_name": stream.get("codec_name", "unknown"),
                }
                
                # Add type-specific information
                if stream.get("codec_type") == "video":
                    stream_info.update({
                        "width": stream.get("width", 0),
                        "height": stream.get("height", 0),
                        "frame_rate": eval(stream.get("r_frame_rate", "0/1")),
                        "pix_fmt": stream.get("pix_fmt", "unknown"),
                    })
                elif stream.get("codec_type") == "audio":
                    stream_info.update({
                        "sample_rate": stream.get("sample_rate", 0),
                        "channels": stream.get("channels", 0),
                        "channel_layout": stream.get("channel_layout", "unknown"),
                    })
                
                media_info["streams"].append(stream_info)
            
            return media_info
        except Exception as e:
            logger.error(f"Error analyzing media file: {str(e)}")
            return {"error": f"Error analyzing media file: {str(e)}"}
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None, 
                                format: str = "mp3", bitrate: str = "192k") -> str:
        """
        Extract audio from a video file
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the extracted audio (if None, generates a path)
            format: Audio format (mp3, wav, etc.)
            bitrate: Audio bitrate
            
        Returns:
            Path to the extracted audio file
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return ""
        
        try:
            # Generate output path if not provided
            if not output_path:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(self.media_dir, f"{base_name}_audio.{format}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Extract audio using FFmpeg
            cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "libmp3lame" if format == "mp3" else "pcm_s16le" if format == "wav" else format,
                "-ab", bitrate,
                "-ar", "44100",  # Sample rate
                "-y",  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=False)
            
            if result.returncode != 0:
                logger.error(f"Error extracting audio: {result.stderr}")
                return ""
            
            logger.info(f"Audio extracted from {video_path} to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            return ""
    
    def extract_frames_from_video(self, video_path: str, output_dir: Optional[str] = None,
                                 fps: Optional[float] = None, max_frames: Optional[int] = None,
                                 format: str = "jpg", quality: int = 90) -> List[str]:
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the extracted frames (if None, generates a path)
            fps: Frames per second to extract (if None, extracts all frames)
            max_frames: Maximum number of frames to extract
            format: Image format (jpg, png, etc.)
            quality: Image quality (0-100)
            
        Returns:
            List of paths to the extracted frames
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
        
        try:
            # Generate output directory if not provided
            if not output_dir:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_dir = os.path.join(self.media_dir, f"{base_name}_frames")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Get video information
            video_info = self.get_media_info(video_path)
            if "error" in video_info:
                logger.error(f"Error getting video info: {video_info['error']}")
                return []
            
            # Calculate frame extraction parameters
            video_duration = video_info.get("duration", 0)
            video_stream = next((s for s in video_info.get("streams", []) if s.get("codec_type") == "video"), None)
            
            if not video_stream:
                logger.error(f"No video stream found in {video_path}")
                return []
            
            original_fps = video_stream.get("frame_rate", 30)
            
            # Determine frame extraction rate
            extract_fps = fps if fps is not None else original_fps
            
            # Calculate total frames to extract
            total_frames = int(video_duration * extract_fps)
            if max_frames and total_frames > max_frames:
                # Adjust fps to match max_frames
                extract_fps = max_frames / video_duration
            
            # Extract frames using FFmpeg
            output_pattern = os.path.join(output_dir, f"frame_%04d.{format}")
            
            cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-vf", f"fps={extract_fps}",
                "-q:v", str(quality // 10),  # Quality (1-10)
                "-y",  # Overwrite output files
                output_pattern
            ]
            
            result = subprocess.run(cmd, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=False)
            
            if result.returncode != 0:
                logger.error(f"Error extracting frames: {result.stderr}")
                return []
            
            # Get list of extracted frames
            frame_files = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) 
                          if f.startswith("frame_") and f.endswith(f".{format}")]
            
            logger.info(f"Extracted {len(frame_files)} frames from {video_path} to {output_dir}")
            return frame_files
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def convert_audio_format(self, audio_path: str, output_path: Optional[str] = None,
                            format: str = "mp3", bitrate: str = "192k") -> str:
        """
        Convert audio to a different format
        
        Args:
            audio_path: Path to the audio file
            output_path: Path to save the converted audio (if None, generates a path)
            format: Target audio format (mp3, wav, etc.)
            bitrate: Audio bitrate
            
        Returns:
            Path to the converted audio file
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return ""
        
        try:
            # Generate output path if not provided
            if not output_path:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(self.media_dir, f"{base_name}_converted.{format}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert audio using FFmpeg
            cmd = [
                self.ffmpeg_path,
                "-i", audio_path,
                "-acodec", "libmp3lame" if format == "mp3" else "pcm_s16le" if format == "wav" else format,
                "-ab", bitrate,
                "-ar", "44100",  # Sample rate
                "-y",  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=False)
            
            if result.returncode != 0:
                logger.error(f"Error converting audio: {result.stderr}")
                return ""
            
            logger.info(f"Audio converted from {audio_path} to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error converting audio: {str(e)}")
            return ""
    
    def convert_video_format(self, video_path: str, output_path: Optional[str] = None,
                            format: str = "mp4", codec: str = "libx264", 
                            resolution: Optional[str] = None, bitrate: Optional[str] = None) -> str:
        """
        Convert video to a different format
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the converted video (if None, generates a path)
            format: Target video format (mp4, avi, etc.)
            codec: Video codec
            resolution: Target resolution (e.g., "1280x720")
            bitrate: Video bitrate
            
        Returns:
            Path to the converted video file
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return ""
        
        try:
            # Generate output path if not provided
            if not output_path:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(self.media_dir, f"{base_name}_converted.{format}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build FFmpeg command
            cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-c:v", codec,
            ]
            
            # Add resolution if specified
            i
(Content truncated due to size limit. Use line ranges to read in chunks)