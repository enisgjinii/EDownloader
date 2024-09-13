import sys
import os
import re
import yt_dlp
import customtkinter as ctk
from threading import Thread
from tkinter import messagebox, StringVar, DoubleVar, filedialog
import logging
from PIL import Image
import requests
from io import BytesIO
import webbrowser
import json
from datetime import datetime
import subprocess
import pyperclip
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Set up logging
log_file = f'youtube_downloader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the MobileNetV2 model once and share it across all threads
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

class DownloadThread(Thread):
    def __init__(self, url, save_path, progress_callback, finished_callback, error_callback, ydl_opts):
        super().__init__()
        self.url = url
        self.save_path = save_path
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.error_callback = error_callback
        self.ydl = None
        self.ydl_opts = ydl_opts

    def run(self):
        try:
            self.ydl = yt_dlp.YoutubeDL(self.ydl_opts)
            self.ydl.download([self.url])
            self.finished_callback("Download completed successfully!")
        except Exception as e:
            self.error_callback(str(e))
            logging.error(f"Download error: {str(e)}")

    def stop(self):
        if self.ydl:
            self.ydl._progress_hooks.clear()
            self.finished_callback("Download stopped by user.")

class FrameAnalysisThread(Thread):
    def __init__(self, video_path, thumbnail_path, display_callback, log_callback, similarity_threshold, max_results=5):
        super().__init__()
        self.video_path = video_path
        self.thumbnail_path = thumbnail_path
        self.display_callback = display_callback
        self.log_callback = log_callback
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results

    def run(self):
        try:
            similar_frames = self.find_similar_frames()
            self.display_callback(similar_frames)
        except Exception as e:
            self.log_callback(f"Error during frame analysis: {str(e)}")

    def find_similar_frames(self):
        # Use the globally loaded model
        global model

        # Load and preprocess the thumbnail image
        thumbnail = cv2.imread(self.thumbnail_path)
        if thumbnail is None:
            self.log_callback("Failed to read thumbnail image.")
            return []

        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
        thumbnail = cv2.resize(thumbnail, (224, 224), interpolation=cv2.INTER_AREA)
        thumbnail_array = np.expand_dims(thumbnail, axis=0)
        thumbnail_array = preprocess_input(thumbnail_array)

        # Extract features from the thumbnail
        thumbnail_features = model.predict(thumbnail_array, batch_size=32, verbose=0)

        # Initialize video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log_callback("Could not open video file.")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps != 0 else 0

        similar_frames = []
        max_results = self.max_results  # Limit to 5 similar frames

        # Adjusted frame extraction interval
        total_frames_to_analyze = min(200, frame_count)  # Limit total frames to analyze based on actual frame count
        interval = max(int(frame_count / total_frames_to_analyze), 1)  # At least 1 frame interval

        frames = []
        frame_timestamps = []

        # Collect frames to process
        for frame_idx in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)
            timestamp = frame_idx / fps if fps != 0 else 0
            frame_timestamps.append(timestamp)

        cap.release()

        if not frames:
            self.log_callback("No frames extracted from the video.")
            return []

        # Convert frames to a NumPy array for batch processing
        frame_arrays = np.array([
            cv2.cvtColor(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            for frame in frames
        ])
        frame_arrays = preprocess_input(frame_arrays)

        # Extract features from frames in batches
        batch_size = 32
        frame_features = []
        for i in range(0, len(frame_arrays), batch_size):
            batch = frame_arrays[i:i+batch_size]
            features = model.predict(batch, batch_size=batch_size, verbose=0)
            frame_features.append(features)
        frame_features = np.vstack(frame_features)

        # Compute cosine similarity between thumbnail and all frames
        similarities = cosine_similarity(
            thumbnail_features.reshape(1, -1),
            frame_features
        )[0]

        # Collect all frames with their similarities and timestamps
        all_frames = list(zip(frames, similarities, frame_timestamps))

        # Filter frames based on similarity threshold
        all_frames = [frame for frame in all_frames if frame[1] >= self.similarity_threshold]

        # Sort frames by similarity score
        all_frames.sort(key=lambda x: -x[1])  # Higher similarity first

        # Take top max_results frames
        top_frames = all_frames[:max_results]

        # Prepare the final similar frames list
        similar_frames = []
        for frame, sim, timestamp in top_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            similar_frames.append((frame_pil, sim, timestamp))
            self.log_callback(f"Timestamp {timestamp:.2f}s: Similarity={sim:.4f}")

        return similar_frames

class YouTubeDownloader(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YouTube Downloader")
        self.geometry("1000x800")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Initialize attributes before creating tabs
        self.download_threads = []
        self.analysis_thread = None
        self.formats = []
        self.video_info = None
        self.video_file_path = ""
        self.thumbnail_file_path = ""
        self.selected_frame = None  # To store the user-selected frame
        self.frames = []  # Store all frames for downloading
        self.download_directory = os.getcwd()  # Default download directory
        self.audio_only = False  # Flag for audio-only downloads

        self.create_tabs()
        self.load_settings()

    def create_tabs(self):
        self.tabview = ctk.CTkTabview(self, width=980, height=780)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.tabview.add("Download")
        self.tabview.add("Analysis")
        self.tabview.add("Settings")
        
        self.create_download_tab()
        self.create_analysis_tab()
        self.create_settings_tab()

    def create_download_tab(self):
        download_tab = self.tabview.tab("Download")
        download_tab.grid_columnconfigure(0, weight=1)
        
        # URL Entry and Fetch
        url_frame = ctk.CTkFrame(download_tab)
        url_frame.pack(fill="x", pady=5, padx=10)
        
        self.url_entry = ctk.CTkEntry(url_frame, placeholder_text="Enter YouTube URL")
        self.url_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        paste_button = ctk.CTkButton(url_frame, text="Paste", width=60, command=self.paste_url)
        paste_button.pack(side="left")
        
        browse_button = ctk.CTkButton(url_frame, text="Browse", width=80, command=self.browse_download_directory)
        browse_button.pack(side="left", padx=(5, 0))
        
        # Display selected download directory
        self.download_dir_label = ctk.CTkLabel(download_tab, text=f"Download Folder: {self.download_directory}", anchor="w")
        self.download_dir_label.pack(fill="x", pady=5, padx=10)
        
        fetch_button = ctk.CTkButton(download_tab, text="Fetch Info", command=self.fetch_video_info)
        fetch_button.pack(fill="x", pady=5, padx=10)
        
        # Thumbnail Display
        self.thumbnail_label = ctk.CTkLabel(download_tab, text="", anchor="center")
        self.thumbnail_label.pack(pady=5, padx=10)
        
        # Quality Selection and Audio Only Option
        quality_frame = ctk.CTkFrame(download_tab)
        quality_frame.pack(fill="x", pady=5, padx=10)
        
        quality_label = ctk.CTkLabel(quality_frame, text="Quality:")
        quality_label.pack(side="left")
        
        self.quality_optionmenu = ctk.CTkOptionMenu(quality_frame, values=["Best", "4K", "2K", "1080p", "720p", "480p", "360p", "240p", "144p"])
        self.quality_optionmenu.pack(side="left", padx=10)
        self.quality_optionmenu.set("Best")
        
        # Audio Only Checkbox
        self.audio_checkbox = ctk.CTkCheckBox(quality_frame, text="Audio Only", command=self.toggle_audio_only)
        self.audio_checkbox.pack(side="left", padx=10)
        
        # Download Button and Progress
        download_button = ctk.CTkButton(download_tab, text="Download", command=self.download)
        download_button.pack(fill="x", pady=5, padx=10)
        self.download_button = download_button
        self.download_button.configure(state="disabled")
        
        progress_frame = ctk.CTkFrame(download_tab)
        progress_frame.pack(fill="x", pady=5, padx=10)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(progress_frame, text="0%")
        self.progress_label.pack(side="left")
        
        # Status Label
        self.download_status_var = StringVar(value="Idle")
        self.download_status_label = ctk.CTkLabel(download_tab, textvariable=self.download_status_var)
        self.download_status_label.pack(fill="x", pady=5, padx=10)
        
        # Log Textbox
        self.log_text = ctk.CTkTextbox(download_tab, height=200)
        self.log_text.pack(fill="both", expand=True, pady=5, padx=10)
        
        # Control Buttons
        control_frame = ctk.CTkFrame(download_tab)
        control_frame.pack(fill="x", pady=5, padx=10)
        
        stop_button = ctk.CTkButton(control_frame, text="Stop All Downloads", command=self.stop_all_downloads, state="disabled")
        stop_button.pack(side="left", padx=(0, 5))
        self.stop_button = stop_button
        
        clear_log_button = ctk.CTkButton(control_frame, text="Clear Log", command=self.clear_log)
        clear_log_button.pack(side="left", padx=5)

    def create_analysis_tab(self):
        analysis_tab = self.tabview.tab("Analysis")
        analysis_tab.grid_columnconfigure(0, weight=1)
        
        # Similarity Threshold
        threshold_frame = ctk.CTkFrame(analysis_tab)
        threshold_frame.pack(fill="x", pady=5, padx=10)
        
        similarity_label = ctk.CTkLabel(threshold_frame, text="Similarity Threshold:")
        similarity_label.pack(side="left")
        
        self.similarity_threshold_var = DoubleVar(value=0.8)
        self.similarity_slider = ctk.CTkSlider(threshold_frame, from_=0.5, to=1.0, number_of_steps=50,
                                               variable=self.similarity_threshold_var, command=self.update_similarity_label)
        self.similarity_slider.pack(side="left", fill="x", expand=True, padx=(10, 10))
        
        self.similarity_value_label = ctk.CTkLabel(threshold_frame, text=f"{self.similarity_threshold_var.get():.2f}")
        self.similarity_value_label.pack(side="left")
        
        # Analyze and Save Buttons
        analyze_frame = ctk.CTkFrame(analysis_tab)
        analyze_frame.pack(fill="x", pady=5, padx=10)
        
        analyze_button = ctk.CTkButton(analyze_frame, text="Find Similar Frames", command=self.analyze_frames)
        analyze_button.pack(side="left", fill="x", expand=True, padx=(0,5))
        self.analyze_button = analyze_button
        self.analyze_button.configure(state="disabled")
        
        save_button = ctk.CTkButton(analyze_frame, text="Save Selected Frame", command=self.save_selected_frame, state="disabled")
        save_button.pack(side="left", fill="x", expand=True, padx=5)
        self.save_frame_button = save_button
        
        download_all_button = ctk.CTkButton(analysis_tab, text="Download All Frames", command=self.download_all_frames, state="disabled")
        download_all_button.pack(fill="x", pady=5, padx=10)
        self.download_all_frames_button = download_all_button
        
        # Similar Frames Display
        self.similar_frames_canvas = ctk.CTkScrollableFrame(analysis_tab, width=850, height=400)
        self.similar_frames_canvas.pack(fill="both", expand=True, pady=5, padx=10)
        
        # Status Label
        self.analysis_status_var = StringVar(value="Idle")
        self.analysis_status_label = ctk.CTkLabel(analysis_tab, textvariable=self.analysis_status_var)
        self.analysis_status_label.pack(fill="x", pady=5, padx=10)

    def create_settings_tab(self):
        settings_tab = self.tabview.tab("Settings")
        settings_tab.grid_columnconfigure(0, weight=1)
        
        # Theme Switch
        theme_frame = ctk.CTkFrame(settings_tab)
        theme_frame.pack(fill="x", pady=10, padx=10)
        
        self.theme_switch_var = ctk.StringVar(value="dark")
        self.theme_switch = ctk.CTkSwitch(theme_frame, text="Light Mode", command=self.change_theme,
                                         variable=self.theme_switch_var, onvalue="light", offvalue="dark")
        self.theme_switch.pack(side="left")
        
        # GitHub and About Buttons
        info_frame = ctk.CTkFrame(settings_tab)
        info_frame.pack(fill="x", pady=10, padx=10)
        
        github_button = ctk.CTkButton(info_frame, text="GitHub", command=self.open_github)
        github_button.pack(side="left", padx=(0, 5))
        
        about_button = ctk.CTkButton(info_frame, text="About", command=self.show_about)
        about_button.pack(side="left", padx=5)
        
        # Open Download Folder
        folder_button = ctk.CTkButton(settings_tab, text="Open Download Folder", command=self.open_download_folder)
        folder_button.pack(fill="x", pady=5, padx=10)
        
        # Quit Button
        quit_button = ctk.CTkButton(settings_tab, text="Quit", command=self.quit_app)
        quit_button.pack(fill="x", pady=5, padx=10)

    def update_similarity_label(self, value):
        self.similarity_value_label.configure(text=f"{float(value):.2f}")

    def paste_url(self):
        self.url_entry.delete(0, 'end')
        self.url_entry.insert(0, pyperclip.paste())

    def browse_download_directory(self):
        directory = filedialog.askdirectory(title="Select Download Directory")
        if directory:
            self.download_directory = directory
            self.download_dir_label.configure(text=f"Download Folder: {self.download_directory}")

    def toggle_audio_only(self):
        self.audio_only = not self.audio_only
        if self.audio_only:
            self.quality_optionmenu.configure(state="disabled")
        else:
            self.quality_optionmenu.configure(state="normal")

    def fetch_video_info(self):
        url = self.url_entry.get()
        if not url:
            messagebox.showwarning("Error", "Please enter a YouTube URL.")
            return

        try:
            ydl_opts = {
                'noplaylist': False,  # Allow playlist extraction
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.video_info = ydl.extract_info(url, download=False)
                # Check if the URL is a playlist
                if 'entries' in self.video_info:
                    # It's a playlist
                    self.formats = self.video_info['entries'][0]['formats']
                    self.is_playlist = True
                    self.log(f"Playlist title: {self.video_info.get('title', 'N/A')}")
                    self.log(f"Number of videos in playlist: {len(self.video_info.get('entries', []))}")
                else:
                    # Single video
                    self.formats = self.video_info['formats']
                    self.is_playlist = False
                    self.log(f"Video title: {self.video_info.get('title', 'N/A')}")
                    self.log(f"Channel: {self.video_info.get('channel', 'N/A')}")
                    duration_seconds = self.video_info.get('duration', 0)
                    minutes, seconds = divmod(duration_seconds, 60)
                    self.log(f"Duration: {int(minutes)}:{int(seconds):02d}")
                    self.log(f"View count: {self.video_info.get('view_count', 'N/A')}")
                    upload_date = self.video_info.get('upload_date', 'N/A')
                    if upload_date != 'N/A':
                        upload_date_formatted = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                    else:
                        upload_date_formatted = 'N/A'
                    self.log(f"Upload date: {upload_date_formatted}")
                    self.display_thumbnail(self.video_info.get('thumbnail', ''))
                    # Enable the download button
                    self.download_button.configure(state="normal")
                    messagebox.showinfo("Success", "Video information fetched successfully!")

        except Exception as e:
            self.log(f"Error fetching video info: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def display_thumbnail(self, thumbnail_url):
        try:
            if thumbnail_url:
                response = requests.get(thumbnail_url)
                img = Image.open(BytesIO(response.content))
                img_resized = img.resize((200, 150), Image.LANCZOS)
                photo = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(200, 150))
                self.thumbnail_label.configure(image=photo)
                self.thumbnail_label.image = photo

                # Save the thumbnail image locally for comparison
                self.thumbnail_file_path = os.path.join(os.getcwd(), 'thumbnail.jpg')
                img.save(self.thumbnail_file_path)
            else:
                self.thumbnail_label.configure(image=None, text="No Thumbnail Available")
        except Exception as e:
            self.log(f"Error displaying thumbnail: {str(e)}")

    def analyze_frames(self):
        if not self.video_file_path or not os.path.exists(self.video_file_path):
            messagebox.showwarning("Error", "Video file not found. Please download the video first.")
            return

        if not self.thumbnail_file_path or not os.path.exists(self.thumbnail_file_path):
            messagebox.showwarning("Error", "Thumbnail image not found.")
            return

        self.analyze_button.configure(state="disabled")
        similarity_threshold = self.similarity_threshold_var.get()
        self.analysis_thread = FrameAnalysisThread(
            self.video_file_path,
            self.thumbnail_file_path,
            self.display_similar_frames,
            self.log,
            similarity_threshold,
            max_results=5  # Limit to 5 similar frames
        )
        self.analysis_thread.start()
        self.analysis_status_var.set("Analyzing frames...")

    def display_similar_frames(self, frames):
        # Clear previous frames
        for widget in self.similar_frames_canvas.winfo_children():
            widget.destroy()

        if not frames:
            self.log("No similar frames found.")
            self.analysis_status_var.set("No similar frames found.")
            self.analyze_button.configure(state="normal")
            return

        self.selected_frame = None  # Reset selected frame
        self.frames = frames  # Store frames for reference

        def on_frame_click(idx):
            self.selected_frame = idx
            self.save_frame_button.configure(state="normal")
            # Highlight selected frame
            for i, child in enumerate(self.similar_frames_canvas.winfo_children()):
                child.configure(border_color="transparent")
            selected_widget = self.similar_frames_canvas.winfo_children()[idx]
            selected_widget.configure(border_color="blue")

        for idx, (frame_pil, similarity, timestamp) in enumerate(frames):
            try:
                img = frame_pil.resize((160, 120), Image.LANCZOS)
                photo = ctk.CTkImage(light_image=img, dark_image=img, size=(160, 120))

                # Create a button for each frame
                frame_button = ctk.CTkButton(
                    self.similar_frames_canvas,
                    image=photo,
                    text=f"Sim: {similarity:.2f}\nTime: {timestamp:.1f}s",
                    compound="top",
                    command=lambda idx=idx: on_frame_click(idx)
                )
                frame_button.image = photo  # Keep a reference
                frame_button.grid(row=idx//2, column=idx%2, padx=10, pady=10, sticky="nsew")

            except Exception as e:
                self.log(f"Error displaying similar frame: {str(e)}")

        # Enable the "Download All Frames" button
        self.download_all_frames_button.configure(state="normal")
        self.analysis_status_var.set("Analysis completed.")
        self.analyze_button.configure(state="normal")

    def save_selected_frame(self):
        if self.selected_frame is None:
            messagebox.showwarning("No Selection", "Please select a frame first.")
            return

        selected_frame_pil, similarity, timestamp = self.frames[self.selected_frame]
        # Ask user where to save the image
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],
                                                 initialfile=f"frame_{timestamp:.1f}s.jpg")
        if save_path:
            selected_frame_pil.save(save_path)
            messagebox.showinfo("Saved", f"Frame saved to {save_path}")

    def download_all_frames(self):
        if not self.frames:
            messagebox.showwarning("No Frames", "No frames are available to download.")
            return

        # Ask user where to save the frames
        directory = filedialog.askdirectory(title="Select Directory to Save Frames")
        if directory:
            for idx, (frame_pil, similarity, timestamp) in enumerate(self.frames):
                filename = os.path.join(directory, f"frame_{timestamp:.1f}s.jpg")
                frame_pil.save(filename)
            messagebox.showinfo("Saved", f"All frames saved to {directory}")

    def download_frame(self, frame_pil, timestamp):
        # Download a single frame
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],
                                                 initialfile=f"frame_{timestamp:.1f}s.jpg")
        if save_path:
            frame_pil.save(save_path)
            messagebox.showinfo("Saved", f"Frame saved to {save_path}")

    def download(self):
        if not self.video_info:
            messagebox.showwarning("Error", "Please fetch video info first.")
            return

        url = self.url_entry.get()
        if not url:
            messagebox.showwarning("Error", "Please enter a YouTube URL.")
            return

        quality = self.quality_optionmenu.get()
        if self.audio_only:
            format_string = 'bestaudio[ext=m4a]/bestaudio'
        elif quality == "Best":
            format_string = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        elif quality == "4K":
            format_string = 'bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]/best[height<=2160][ext=mp4]/best'
        elif quality == "2K":
            format_string = 'bestvideo[height<=1440][ext=mp4]+bestaudio[ext=m4a]/best[height<=1440][ext=mp4]/best'
        else:
            format_string = f'bestvideo[height<={quality[:-1]}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality[:-1]}][ext=mp4]/best'

        ydl_opts = {
            'outtmpl': os.path.join(self.download_directory, '%(title)s.%(ext)s'),
            'format': format_string,
            'progress_hooks': [self.update_progress],
            'noplaylist': False if self.is_playlist else True,  # Allow playlist download
            'noprogress': True,
            'no_warnings': True,
            'quiet': True,
            'merge_output_format': 'mp4',
            'prefer_ffmpeg': True,
        }

        # Handle playlist
        if self.is_playlist:
            # For playlists, download each video separately
            self.download_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.download_status_var.set("Starting playlist download...")
            for entry in self.video_info.get('entries', []):
                video_url = entry.get('webpage_url')
                video_title = re.sub(r'[\\/*?:"<>|]', "", entry.get('title', 'video'))  # Remove invalid characters
                ydl_opts['outtmpl'] = os.path.join(self.download_directory, f"{video_title}.%(ext)s")
                download_thread = DownloadThread(
                    video_url,
                    self.download_directory,
                    self.update_progress,
                    self.download_finished,
                    self.download_error,
                    ydl_opts
                )
                self.download_threads.append(download_thread)
                download_thread.start()
                self.log(f"Started downloading: {video_title}")
        else:
            # Single video download
            self.download_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.download_status_var.set("Download started...")
            download_thread = DownloadThread(
                url,
                self.download_directory,
                self.update_progress,
                self.download_finished,
                self.download_error,
                ydl_opts
            )
            self.download_threads.append(download_thread)
            download_thread.start()

    def update_progress(self, d):
        if d['status'] == 'downloading':
            try:
                percent_str = d.get('_percent_str', '0%').strip()
                # Handle cases where percent_str might have multiple '%' symbols or unexpected formats
                percent_match = re.search(r'([\d.]+)%', percent_str)
                percent = float(percent_match.group(1)) if percent_match else 0.0
                self.progress_bar.set(percent / 100)
                self.progress_label.configure(text=f"{percent:.1f}%")
                self.download_status_var.set(f"Downloading: {d.get('filename', 'Unknown file')} - {percent:.1f}%")
                self.log(f"Downloading: {d.get('filename', 'Unknown file')} - {percent:.1f}%")
            except Exception as e:
                self.log(f"Error parsing progress: {str(e)}")
        elif d['status'] == 'finished':
            self.download_status_var.set("Download completed.")
            self.progress_bar.set(1.0)
            self.progress_label.configure(text="100%")

    def download_finished(self, message):
        self.log(message)
        self.download_status_var.set(message)
        self.reset_progress()
        # Re-enable download button if no other downloads are active
        if not any(thread.is_alive() for thread in self.download_threads):
            self.download_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            messagebox.showinfo("Success", message)
            self.analyze_button.configure(state="normal")

    def download_error(self, message):
        self.log(f"Error: {message}")
        self.download_status_var.set("Download failed. Check log for details.")
        self.reset_progress()
        # Re-enable download button if no other downloads are active
        if not any(thread.is_alive() for thread in self.download_threads):
            self.download_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            messagebox.showerror("Error", message)

    def reset_progress(self):
        self.progress_bar.set(0)
        self.progress_label.configure(text="0%")

    def stop_all_downloads(self):
        for thread in self.download_threads:
            if thread.is_alive():
                thread.stop()
        self.download_threads = []
        self.reset_progress()
        self.download_status_var.set("All downloads stopped by user.")
        self.download_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.log("All downloads stopped by user.")

    def log(self, message):
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        logging.info(message)

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def change_theme(self):
        if self.theme_switch_var.get() == "light":
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode("dark")
        self.save_settings()

    def open_github(self):
        webbrowser.open("https://github.com/your-github-repo")  # Replace with your actual GitHub URL

    def show_about(self):
        about_text = (
            "YouTube Downloader\n\n"
            "Version: 2.3\n"
            "Author: Your Name\n"
            "License: MIT\n\n"
            "This application allows you to download YouTube videos and playlists easily.\n"
            "Features:\n"
            "- Select video quality or download audio only\n"
            "- Download single videos or entire playlists\n"
            "- Find frames similar to the thumbnail\n"
            "- Display frames as buttons and download selected frames\n"
            "- Save selected frame or download all similar frames\n"
            "- Select custom download directory\n"
            "- Batch download management"
        )
        messagebox.showinfo("About", about_text)

    def open_download_folder(self):
        folder = self.download_directory
        if folder:
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            else:
                subprocess.Popen(['xdg-open', folder])

    def load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
                self.theme_switch_var.set(settings.get('theme', 'dark'))
                self.quality_optionmenu.set(settings.get('quality', 'Best'))
                self.similarity_threshold_var.set(settings.get('similarity_threshold', 0.8))
                self.download_directory = settings.get('download_directory', os.getcwd())
                self.download_dir_label.configure(text=f"Download Folder: {self.download_directory}")
                self.audio_only = settings.get('audio_only', False)
                self.audio_checkbox.configure(state="normal")
                self.audio_checkbox.select() if self.audio_only else self.audio_checkbox.deselect()
                self.update_similarity_label(self.similarity_threshold_var.get())
                self.change_theme()
        except FileNotFoundError:
            pass

    def save_settings(self):
        settings = {
            'theme': self.theme_switch_var.get(),
            'quality': self.quality_optionmenu.get(),
            'similarity_threshold': self.similarity_threshold_var.get(),
            'download_directory': self.download_directory,
            'audio_only': self.audio_only
        }
        with open('settings.json', 'w') as f:
            json.dump(settings, f)

    def quit_app(self):
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.save_settings()
            self.destroy()

    def on_closing(self):
        self.quit_app()

if __name__ == "__main__":
    # Ensure TensorFlow uses GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logging.info("GPU found and configured for TensorFlow.")
        except Exception as e:
            logging.error(f"Error configuring GPU for TensorFlow: {e}")
    
    app = YouTubeDownloader()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
