from pathlib import Path
import subprocess

def convert_images_to_video(image_dir, output_file, fps=30, pattern="frame_%d.png", codec="libx264", pix_fmt="yuv420p"):
    """
    Convert a series of images in image_dir to a video saved at output_file.

    Parameters:
    - image_dir (str or Path): Directory containing image files.
    - output_file (str or Path): Path to save the output video.
    - fps (int): Frames per second for the video.
    - pattern (str): Pattern for input image filenames.
    - codec (str): Video codec to use.
    - pix_fmt (str): Pixel format.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist.")

    # Check for matching image files
    image_files = list(image_dir.glob(pattern.replace("%d", "*")))
    if not image_files:
        print(f"No images found in {image_dir} matching pattern '{pattern}'.")
        return

    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = image_dir / output_path

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if exists
        "-framerate", str(fps),
        "-i", str(image_dir / pattern),
        "-c:v", codec,
        "-pix_fmt", pix_fmt,
        str(output_path)
    ]

    print(f"Converting images in {image_dir} to video at {output_path}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Conversion successful. ffmpeg output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during video conversion:")
        print(e.stderr)

# # Example usage - inside a class method:
# def convert_external_images_to_video(self, output_file="external_video.mp4", fps=30):
#     convert_images_to_video(self.IMG_PATH, output_file, fps=fps, pattern="frame_%d.png")
#
# def convert_onboard_images_to_video(self, output_file="onboard_video.mp4", fps=30):
#     convert_images_to_video(self.ONBOARD_IMG_PATH, output_file, fps=fps, pattern="frame_%d.png")
