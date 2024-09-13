import sys
import os
import re
import yt_dlp
import customtkinter as ctk
from threading import Thread
from tkinter import messagebox, StringVar, DoubleVar, filedialog, ttk
import logging
from PIL import Image, ImageTk
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
    # Initialize the model once and reuse it
    model = None

    def __init__(self, video_path, thumbnail_path, display_callback, log_callback, similarity_threshold, max_results=10):
        super().__init__()
        self.video_path = video_path
        self.thumbnail_path = thumbnail_path
        self.display_callback = display_callback
        self.log_callback = log_callback
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results

        # Load model if not already loaded
        if FrameAnalysisThread.model is None:
            FrameAnalysisThread.model = self.load_model()

    def load_model(self):
        try:
            base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
            self.log_callback("Model loaded successfully.")
            return model
        except Exception as e:
            self.log_callback(f"Error loading model: {str(e)}")
            raise e

    def run(self):
        try:
            similar_frames = self.find_similar_frames()
            self.display_callback(similar_frames)
        except Exception as e:
            self.log_callback(f"Error during frame analysis: {str(e)}")

    def find_similar_frames(self):
        model = FrameAnalysisThread.model

        # Load and preprocess the thumbnail image
        thumbnail = keras_image.load_img(self.thumbnail_path, target_size=(224, 224))
        thumbnail_array = keras_image.img_to_array(thumbnail)
        thumbnail_array = np.expand_dims(thumbnail_array, axis=0)
        thumbnail_array = preprocess_input(thumbnail_array)

        # Extract features from the thumbnail
        thumbnail_features = model.predict(thumbnail_array)

        # Initialize video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log_callback("Could not open video file.")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps != 0 else 0

        similar_frames = []

        # Adjusted frame extraction interval
        total_frames_to_analyze = min(frame_count, 200)  # Limit total frames to analyze
        frame_indices = np.linspace(0, frame_count - 1, total_frames_to_analyze, dtype=np.int32)

        frames = []
        frame_timestamps = []

        # Collect frames to process
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)
            timestamp = idx / fps
            frame_timestamps.append(timestamp)

        cap.release()

        if not frames:
            self.log_callback("No frames extracted from the video.")
            return []

        # Preprocess frames in batch
        frame_arrays = np.array([
            preprocess_input(
                np.expand_dims(
                    cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224)),
                    axis=0
                )
            )[0]
            for frame in frames
        ])

        # Extract features from frames
        frame_features = model.predict(frame_arrays, batch_size=32, verbose=0)

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
        top_frames = all_frames[:self.max_results]

        # Prepare the final similar frames list
        similar_frames = []
        for frame, sim, timestamp in top_frames:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            similar_frames.append((frame_pil, sim, timestamp))
            self.log_callback(f"Timestamp {self.format_timestamp(timestamp)}: Similarity={sim:.4f}")

        return similar_frames

    @staticmethod
    def format_timestamp(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


class YouTubeDownloader(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YouTube Downloader")
        self.geometry("1200x800")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.create_tabs()
        self.download_thread = None
        self.analysis_thread = None
        self.formats = []
        self.video_info = None
        self.video_file_path = ""
        self.thumbnail_file_path = ""
        self.selected_frame = None  # To store the user-selected frame
        self.frames = []  # Store all frames for downloading
        self.load_settings()

    def create_tabs(self):
        self.tabview = ctk.CTkTabview(self, width=1180, height=780)
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
        url_frame.pack(fill="x", pady=10, padx=10)

        self.url_entry = ctk.CTkEntry(url_frame, placeholder_text="Enter YouTube URL")
        self.url_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        paste_button = ctk.CTkButton(url_frame, text="Paste", width=80, command=self.paste_url)
        paste_button.pack(side="left")

        fetch_button = ctk.CTkButton(download_tab, text="Fetch Info", command=self.fetch_video_info)
        fetch_button.pack(fill="x", pady=5, padx=10)

        # Thumbnail Display
        thumbnail_frame = ctk.CTkFrame(download_tab)
        thumbnail_frame.pack(fill="x", pady=5, padx=10)

        self.thumbnail_label = ctk.CTkLabel(thumbnail_frame, text="", anchor="center")
        self.thumbnail_label.pack(pady=5, padx=10)

        # Video Information Display
        info_frame = ctk.CTkFrame(download_tab)
        info_frame.pack(fill="x", pady=5, padx=10)

        self.info_text = ctk.CTkTextbox(info_frame, height=100, state="disabled")
        self.info_text.pack(fill="both", expand=True)

        # Quality Selection
        quality_frame = ctk.CTkFrame(download_tab)
        quality_frame.pack(fill="x", pady=5, padx=10)

        quality_label = ctk.CTkLabel(quality_frame, text="Quality:")
        quality_label.pack(side="left")

        self.quality_optionmenu = ctk.CTkOptionMenu(quality_frame, values=["Best", "4K", "2K", "1080p", "720p", "480p", "360p", "240p", "144p"])
        self.quality_optionmenu.pack(side="left", padx=10)
        self.quality_optionmenu.set("Best")

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
        self.log_text = ctk.CTkTextbox(download_tab, height=150, state="disabled")
        self.log_text.pack(fill="both", expand=True, pady=5, padx=10)

        # Control Buttons
        control_frame = ctk.CTkFrame(download_tab)
        control_frame.pack(fill="x", pady=5, padx=10)

        stop_button = ctk.CTkButton(control_frame, text="Stop", command=self.stop_download, state="disabled")
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
        analyze_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.analyze_button = analyze_button
        self.analyze_button.configure(state="disabled")

        save_button = ctk.CTkButton(analyze_frame, text="Save Selected Frame", command=self.save_selected_frame, state="disabled")
        save_button.pack(side="left", fill="x", expand=True, padx=5)
        self.save_frame_button = save_button

        export_button = ctk.CTkButton(analyze_frame, text="Export to CSV", command=self.export_to_csv, state="disabled")
        export_button.pack(side="left", fill="x", expand=True, padx=5)
        self.export_button = export_button

        download_all_button = ctk.CTkButton(analysis_tab, text="Download All Frames", command=self.download_all_frames, state="disabled")
        download_all_button.pack(fill="x", pady=5, padx=10)
        self.download_all_frames_button = download_all_button

        # Similar Frames Display with Treeview
        display_frame = ctk.CTkFrame(analysis_tab)
        display_frame.pack(fill="both", expand=True, pady=5, padx=10)

        # Adding a Treeview to display timestamps and similarity
        self.frames_tree = ttk.Treeview(display_frame, columns=("Timestamp", "Similarity"), show='headings', selectmode='browse')
        self.frames_tree.heading("Timestamp", text="Timestamp (MM:SS)")
        self.frames_tree.heading("Similarity", text="Similarity")
        self.frames_tree.column("Timestamp", anchor="center", width=120)
        self.frames_tree.column("Similarity", anchor="center", width=100)
        self.frames_tree.pack(side="left", fill="both", expand=True)

        self.frames_tree.bind("<Double-1>", self.on_tree_item_double_click)

        # Adding a scrollbar to the treeview
        scrollbar = ttk.Scrollbar(display_frame, orient="vertical", command=self.frames_tree.yview)
        self.frames_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # Frame Preview Section
        preview_frame = ctk.CTkFrame(analysis_tab)
        preview_frame.pack(fill="both", expand=True, pady=5, padx=10)

        self.preview_label = ctk.CTkLabel(preview_frame, text="Selected Frame Preview:", anchor="w")
        self.preview_label.pack(fill="x", pady=(0, 5), padx=10)

        self.frame_preview = ctk.CTkLabel(preview_frame, text="", anchor="center")
        self.frame_preview.pack(fill="both", expand=True, pady=(0, 5), padx=10)

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

    def fetch_video_info(self):
        url = self.url_entry.get()
        if not url:
            messagebox.showwarning("Error", "Please enter a YouTube URL.")
            return

        try:
            ydl_opts = {
                'noplaylist': True,
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.video_info = ydl.extract_info(url, download=False)
                if 'entries' in self.video_info:
                    messagebox.showwarning("Playlist Detected", "This URL corresponds to a playlist. Please enter a single video URL.")
                    return
                self.formats = self.video_info['formats']
                self.display_video_info()
                self.display_thumbnail(self.video_info.get('thumbnail', ''))

                # Enable the download button
                self.download_button.configure(state="normal")

                messagebox.showinfo("Success", "Video information fetched successfully!")

        except Exception as e:
            self.log(f"Error fetching video info: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def display_video_info(self):
        if not self.video_info:
            return
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        title = self.video_info.get('title', 'N/A')
        channel = self.video_info.get('uploader', 'N/A')
        duration_seconds = self.video_info.get('duration', 0)
        minutes, seconds = divmod(duration_seconds, 60)
        duration = f"{int(minutes)}:{int(seconds):02d}"
        view_count = self.video_info.get('view_count', 'N/A')
        upload_date = self.video_info.get('upload_date', 'N/A')
        if upload_date != 'N/A':
            upload_date_formatted = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        else:
            upload_date_formatted = 'N/A'

        info = (
            f"Title: {title}\n"
            f"Channel: {channel}\n"
            f"Duration: {duration}\n"
            f"View count: {view_count}\n"
            f"Upload date: {upload_date_formatted}\n"
        )
        self.info_text.insert("end", info)
        self.info_text.configure(state="disabled")
        self.log(info)

    def display_thumbnail(self, thumbnail_url):
        try:
            if thumbnail_url:
                response = requests.get(thumbnail_url)
                img = Image.open(BytesIO(response.content))
                img_resized = img.resize((300, 200), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img_resized)
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
            max_results=20  # Adjust as needed
        )
        self.analysis_thread.start()
        self.analysis_status_var.set("Analyzing frames...")

    def display_similar_frames(self, frames):
        # Clear previous frames
        for item in self.frames_tree.get_children():
            self.frames_tree.delete(item)

        if not frames:
            self.log("No similar frames found.")
            self.analysis_status_var.set("No similar frames found.")
            self.analyze_button.configure(state="normal")
            return

        self.selected_frame = None  # Reset selected frame
        self.frames = frames  # Store frames for reference

        for idx, (frame_pil, similarity, timestamp) in enumerate(frames):
            # Insert into treeview with formatted timestamp
            formatted_timestamp = self.format_timestamp(timestamp)
            self.frames_tree.insert("", "end", iid=idx, values=(formatted_timestamp, f"{similarity:.2f}"))

            # Optionally, save frame images for potential download
            frames[idx] = (frame_pil, similarity, timestamp)

        # Enable the "Download All Frames" and "Export to CSV" buttons
        self.download_all_frames_button.configure(state="normal")
        self.export_button.configure(state="normal")
        self.analysis_status_var.set("Analysis completed.")
        self.analyze_button.configure(state="normal")

    @staticmethod
    def format_timestamp(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def on_tree_item_double_click(self, event):
        selected_item = self.frames_tree.focus()
        if selected_item:
            idx = int(selected_item)
            frame_pil, similarity, timestamp = self.frames[idx]
            # Display the selected frame in the preview section
            img_resized = frame_pil.resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img_resized)
            self.frame_preview.configure(image=photo)
            self.frame_preview.image = photo
            self.selected_frame = idx
            self.save_frame_button.configure(state="normal")

    def save_selected_frame(self):
        if self.selected_frame is None:
            messagebox.showwarning("No Selection", "Please select a frame first.")
            return

        selected_frame_pil, similarity, timestamp = self.frames[self.selected_frame]
        # Ask user where to save the image
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")],
                                                 initialfile=f"frame_{self.format_timestamp(timestamp)}.jpg")
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
                filename = os.path.join(directory, f"frame_{self.format_timestamp(timestamp)}.jpg")
                frame_pil.save(filename)
            messagebox.showinfo("Saved", f"All frames saved to {directory}")

    def export_to_csv(self):
        if not self.frames:
            messagebox.showwarning("No Frames", "No frames to export.")
            return

        # Ask user where to save the CSV
        save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV File", "*.csv")],
                                                 initialfile="similar_frames.csv")
        if save_path:
            import csv
            try:
                with open(save_path, mode='w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(["Timestamp (MM:SS)", "Similarity"])
                    for frame_pil, similarity, timestamp in self.frames:
                        formatted_timestamp = self.format_timestamp(timestamp)
                        writer.writerow([formatted_timestamp, f"{similarity:.2f}"])
                messagebox.showinfo("Exported", f"Frame data exported to {save_path}")
            except Exception as e:
                self.log(f"Error exporting to CSV: {str(e)}")
                messagebox.showerror("Error", f"Failed to export CSV: {str(e)}")

    def download(self):
        if not self.video_info:
            messagebox.showwarning("Error", "Please fetch video info first.")
            return

        save_path = filedialog.askdirectory(title="Select Download Directory")
        if not save_path:
            return

        quality = self.quality_optionmenu.get()
        if quality == "Best":
            format_string = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        elif quality == "4K":
            format_string = 'bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]/best[height<=2160][ext=mp4]/best'
        elif quality == "2K":
            format_string = 'bestvideo[height<=1440][ext=mp4]+bestaudio[ext=m4a]/best[height<=1440][ext=mp4]/best'
        else:
            format_string = f'bestvideo[height<={quality[:-1]}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality[:-1]}][ext=mp4]/best'

        ydl_opts = {
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
            'format': format_string,
            'progress_hooks': [self.update_progress],
            'noplaylist': True,
            'noprogress': False,
            'no_warnings': True,
            'quiet': True,
            'merge_output_format': 'mp4',
            'prefer_ffmpeg': True,
        }

        # Get the output file path
        video_title = re.sub(r'[\\/*?:"<>|]', "", self.video_info.get('title', 'video'))  # Remove invalid characters
        self.video_file_path = os.path.join(save_path, f"{video_title}.mp4")

        self.download_thread = DownloadThread(
            self.url_entry.get(),
            save_path,
            self.update_progress,
            self.download_finished,
            self.download_error,
            ydl_opts
        )
        self.download_thread.start()
        self.download_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.download_status_var.set("Download started...")

    def update_progress(self, d):
        if d['status'] == 'downloading':
            try:
                percent_str = d.get('_percent_str', '0%').strip()
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
        messagebox.showinfo("Success", message)
        self.reset_progress()
        self.download_status_var.set("Download completed successfully!")
        self.analyze_button.configure(state="normal")

    def download_error(self, message):
        self.log(f"Error: {message}")
        messagebox.showerror("Error", message)
        self.reset_progress()
        self.download_status_var.set("Download failed. Check log for details.")

    def reset_progress(self):
        self.progress_bar.set(0)
        self.progress_label.configure(text="0%")
        self.download_button.configure(state="normal" if self.video_info else "disabled")
        self.stop_button.configure(state="disabled")

    def stop_download(self):
        if self.download_thread and self.download_thread.is_alive():
            self.download_thread.stop()
            self.log("Download stopped by user.")
            self.reset_progress()
            self.download_status_var.set("Download stopped by user.")

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        logging.info(message)

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

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
            "This application allows you to download YouTube videos easily.\n"
            "Features:\n"
            "- Select video quality\n"
            "- Download videos\n"
            "- Find frames similar to the thumbnail\n"
            "- Display similar frames with images and timestamps\n"
            "- Preview and save selected frames\n"
            "- Export similar frames data to CSV\n"
            "- Download all similar frames at once"
        )
        messagebox.showinfo("About", about_text)

    def open_download_folder(self):
        folder = os.getcwd()
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
                self.update_similarity_label(self.similarity_threshold_var.get())
                self.change_theme()
        except FileNotFoundError:
            pass

    def save_settings(self):
        settings = {
            'theme': self.theme_switch_var.get(),
            'quality': self.quality_optionmenu.get(),
            'similarity_threshold': self.similarity_threshold_var.get()
        }
        with open('settings.json', 'w') as f:
            json.dump(settings, f)

    def quit_app(self):
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.save_settings()
            self.destroy()

    def on_closing(self):
        self.quit_app()

    def format_timestamp(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"


if __name__ == "__main__":
    app = YouTubeDownloader()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
