
# YouTube Downloader

![YouTube Downloader Logo](path_to_logo_image) <!-- Replace with your actual logo path -->

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [Usage](#usage)
  - [Navigating the UI](#navigating-the-ui)
  - [Download Section](#download-section)
  - [Analysis Section](#analysis-section)
  - [Settings Section](#settings-section)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Introduction

YouTube Downloader is a feature-rich desktop application that allows users to effortlessly download YouTube videos in various qualities. Beyond downloading, the application offers advanced functionalities such as frame analysis to identify and save frames similar to the video's thumbnail. With a modern and intuitive sidebar-based UI, managing downloads and analyses becomes seamless.

## Features

- **Download YouTube Videos:**
  - Enter YouTube video URLs to fetch video information.
  - Choose from multiple video quality options including Best, 4K, 2K, 1080p, and more.
  - Monitor download progress with a visual progress bar and percentage indicator.
  - Pause or stop ongoing downloads as needed.

- **Frame Analysis:**
  - Analyze downloaded videos to find frames similar to the video's thumbnail.
  - Adjust similarity thresholds to refine frame selection.
  - Preview, save individual frames, or download all similar frames at once.
  - Export frame data (timestamps and similarity scores) to a CSV file for further analysis.

- **User-Friendly UI:**
  - Modern sidebar navigation to switch between Download, Analysis, and Settings sections.
  - Dark and Light mode themes for personalized user experience.
  - Detailed logging of all operations with the ability to view and clear logs.

- **Additional Utilities:**
  - Quick access to open the download folder.
  - Easy access to the application's GitHub repository and About section.
  - Persistent settings saved across sessions.

## Demo

![Download Section Screenshot](path_to_download_section_screenshot)
*Download Section*

![Analysis Section Screenshot](path_to_analysis_section_screenshot)
*Analysis Section*

![Settings Section Screenshot](path_to_settings_section_screenshot)
*Settings Section*

*Note: Replace `path_to_..._screenshot` with actual paths to your screenshot images.*

## Installation

### Prerequisites

Ensure you have **Python 3.7** or later installed on your system. You can download Python from the [official website](https://www.python.org/downloads/).

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-github-repo/YouTube-Downloader.git
   cd YouTube-Downloader
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   ```

   - **Activate the Virtual Environment:**
     - **Windows:**
       ```bash
       venv\Scripts\activate
       ```
     - **macOS/Linux:**
       ```bash
       source venv/bin/activate
       ```

3. **Install Required Dependencies:**

   Ensure you have `pip` updated:

   ```bash
   python -m pip install --upgrade pip
   ```

   Install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   *If you don't have a `requirements.txt`, you can install the packages manually:*

   ```bash
   pip install customtkinter yt-dlp numpy opencv-python tensorflow Pillow scikit-learn pyperclip
   ```

4. **Run the Application:**

   ```bash
   python youtube_downloader.py
   ```

   *Ensure that `youtube_downloader.py` is the name of your main Python file.*

## Usage

### Navigating the UI

Upon launching the application, you'll be greeted with a modern sidebar on the left, allowing you to navigate between the three main sections:

1. **Download**
2. **Analysis**
3. **Settings**

### Download Section

1. **Enter YouTube URL:**
   - Paste or manually enter the URL of the YouTube video you wish to download.
   - Use the "Paste" button for convenience if the URL is copied to your clipboard.

2. **Fetch Video Information:**
   - Click the "Fetch Info" button to retrieve video details such as title, channel, duration, view count, and upload date.
   - The video's thumbnail will be displayed for reference.

3. **Select Video Quality:**
   - Choose your desired video quality from the dropdown menu (e.g., Best, 4K, 2K, 1080p).

4. **Download the Video:**
   - Click the "Download" button to start downloading the video.
   - Monitor the download progress via the progress bar and percentage indicator.
   - Use the "Stop" button to cancel an ongoing download if necessary.

5. **Logs:**
   - View real-time logs of download activities in the log textbox.
   - Use the "Clear Log" button to clear the logs.

### Analysis Section

1. **Set Similarity Threshold:**
   - Adjust the slider to set the similarity threshold for frame analysis. Higher values mean stricter similarity requirements.

2. **Find Similar Frames:**
   - Click the "Find Similar Frames" button to analyze the downloaded video and identify frames similar to the thumbnail.
   - The analysis results will be displayed in a tree view listing timestamps and similarity scores.

3. **Preview and Save Frames:**
   - Double-click on any frame entry to preview the selected frame.
   - Use the "Save Selected Frame" button to save the previewed frame as an image file.

4. **Download All Frames:**
   - Click the "Download All Frames" button to save all identified similar frames to a chosen directory.

5. **Export to CSV:**
   - Export the list of similar frames, including their timestamps and similarity scores, to a CSV file for further analysis.

### Settings Section

1. **Theme Switch:**
   - Toggle between Light and Dark modes to suit your preference.

2. **GitHub Repository:**
   - Click the "GitHub" button to open the application's GitHub repository in your default web browser.

3. **About:**
   - View detailed information about the application, including version, author, and features.

4. **Open Download Folder:**
   - Quickly access the folder where downloaded videos and frames are saved.

5. **Quit Application:**
   - Exit the application safely using the "Quit" button.

## Configuration

### Settings Persistence

User preferences such as theme, selected video quality, and similarity threshold are saved to a `settings.json` file in the application's directory. These settings are loaded automatically when the application starts, ensuring a personalized experience across sessions.

### Log Files

All operational logs are saved to a timestamped log file (e.g., `youtube_downloader_20230914_153045.log`) in the application's directory. These logs provide detailed information about downloads, analyses, and any errors encountered.

## Contributing

Contributions are welcome! If you'd like to enhance the application, please follow these steps:

1. **Fork the Repository:**

   Click the "Fork" button on the repository page to create your own copy.

2. **Clone Your Fork:**

   ```bash
   git clone https://github.com/your-username/YouTube-Downloader.git
   cd YouTube-Downloader
   ```

3. **Create a New Branch:**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Make Your Changes:**

   Implement your feature or bug fix.

5. **Commit Your Changes:**

   ```bash
   git commit -m "Add your descriptive commit message here"
   ```

6. **Push to Your Fork:**

   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Create a Pull Request:**

   Go to the original repository and click "Compare & pull request" to submit your changes for review.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software as per the terms of the license.

## Acknowledgements

- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for providing a modern and customizable UI framework.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for enabling robust video downloading capabilities.
- [TensorFlow](https://www.tensorflow.org/) and [MobileNetV2](https://keras.io/api/applications/mobilenet/) for powerful image processing and analysis.
- [OpenCV](https://opencv.org/) for efficient video and image handling.
- [Pillow](https://python-pillow.org/) for image processing tasks.
- [Scikit-learn](https://scikit-learn.org/) for implementing machine learning functionalities.

## Contact

For any questions, suggestions, or feedback, please feel free to reach out:

- **Author:** Your Name
- **Email:** your.email@example.com
- **GitHub:** [https://github.com/your-github-repo](https://github.com/your-github-repo)

---

*Feel free to customize this `README.md` further to better fit your project's specifics and personal preferences.*


---

## Additional Notes:

1. **Replace Placeholder Texts:**
   - **Logo Image:** Replace `path_to_logo_image` with the actual path to your application's logo.
   - **Screenshots:** Replace `path_to_download_section_screenshot`, `path_to_analysis_section_screenshot`, and `path_to_settings_section_screenshot` with actual paths to your screenshot images.
   - **GitHub Repository URL:** Update the `git clone` command and GitHub buttons with your actual repository URL.
   - **Author Information:** Update the author name, email, and GitHub link with your actual details.

2. **`requirements.txt`:**
   - It's a good practice to include a `requirements.txt` file in your repository for easy installation of dependencies. Here's an example based on your project's dependencies:

     ```txt
     customtkinter
     yt-dlp
     numpy
     opencv-python
     tensorflow
     Pillow
     scikit-learn
     pyperclip
     ```

3. **License File:**
   - Ensure you include a `LICENSE` file in your repository. If you choose the MIT License as indicated, you can generate one from [Choose a License](https://choosealicense.com/licenses/mit/).

4. **Enhancements:**
   - **Screenshots and Logo:** Adding visual elements like screenshots and a logo can significantly enhance the README's appeal.
   - **Badges:** Consider adding badges (e.g., build status, license) for a professional touch.

5. **File Structure Example:**

   ```
   YouTube-Downloader/
   ├── screenshots/
   │   ├── download_section.png
   │   ├── analysis_section.png
   │   └── settings_section.png
   ├── logo.png
   ├── youtube_downloader.py
   ├── requirements.txt
   ├── LICENSE
   └── README.md
   ```

6. **Testing:**
   - Before finalizing the README, test the installation and usage steps to ensure they work seamlessly for users.

7. **Future Updates:**
   - Keep the README updated with new features, changes, or improvements to maintain accurate documentation.
