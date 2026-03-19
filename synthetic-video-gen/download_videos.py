import os
import argparse
import subprocess
import sys

def download_drive_file(file_id, output_path):
    print(f"Attempting to download {file_id} to {output_path}...")
    try:
        import gdown
        gdown.download(id=file_id, output=output_path, quiet=False)
    except ImportError:
        print("gdown not found. Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        gdown.download(id=file_id, output=output_path, quiet=False)

def download_video_dataset(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # This is the Google Drive link from the GitHub repo. It's a zip file.
    file_id = "1KOUzy07viQzpmpcBqydUA043VQZ4nmRv"
    zip_path = os.path.join(output_dir, "VNBench_videos.zip")
    
    if not os.path.exists(zip_path):
        download_drive_file(file_id, zip_path)
    else:
        print(f"File {zip_path} already exists. Skipping download.")
        
    # Extract the zip file
    extract_dir = os.path.join(output_dir, "extracted")
    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_path} to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file. It might be due to Google Drive rate limits.")
            print("Please try manually downloading from: https://drive.google.com/file/d/1KOUzy07viQzpmpcBqydUA043VQZ4nmRv/view")
            sys.exit(1)
    else:
        print("Videos are already extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="source_videos")
    args = parser.parse_args()
    
    download_video_dataset(args.output_dir)
