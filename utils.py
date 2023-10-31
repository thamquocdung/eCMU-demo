import yt_dlp
import os

def _yt_dlp_download(url, quality, dst, ext='.mp4'):
    num_thread = 16
    ydl_opts = {'format': 'bv*[ext=mp4][height<={0}]+ba/b[ext=mp4][height<={0}]'.format(quality),
                'outtmpl': os.path.join(dst, '%(id)s.%(ext)s'),
                'merge_output_format': 'mp4', 
                'concurrent_fragment_downloads': num_thread}

    print(f'using num_thread = {num_thread} for youtube downloading ...')
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)

        try:
            info_dict = info_dict['entries'][0]
        except Exception as e:
            info_dict['ext'] = 'mp4'

        fn = ydl.prepare_filename(info_dict)
        if os.path.exists(fn):
            return fn, True
        else:
            parent_dir = os.path.dirname(fn)
            basename = os.path.basename(fn)[:-4]
            for file in os.listdir(parent_dir):
                if file.startswith(basename) and file.endswith(ext):
                    fn = os.path.join(parent_dir, file)
                    return fn, True
            return '', False

def download_audio_file(url, dst="data/scriptdownloaded_audio"):
    ydl_opts = {'format': 'bestaudio/best',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192', }],
                'outtmpl': os.path.join(dst, "%(id)s.%(ext)s")}
    print("Downloading Audio ....")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        try:
            info_dict = info_dict['entries'][0]
        except Exception as e:
            info_dict['ext'] = 'wav'

        fn = ydl.prepare_filename(info_dict)
        if os.path.exists(fn):
            return fn, True
        else:
            parent_dir = os.path.dirname(fn)
            basename = os.path.basename(fn)[:-4]
            for file in os.listdir(parent_dir):
                if file.startswith(basename) and file.endswith('.wav'):
                    fn = os.path.join(parent_dir, file)
                    return fn, True
            return '', False

def cut_video(videofile, from_second, to_second):
    dir_name, base_name = os.path.dirname(videofile), os.path.basename(videofile)
    out_a = os.path.join(dir_name.replace("videos", "audio"), base_name.replace(".mp4", ".wav"))
    out_v = os.path.join(dir_name, f"cut_{base_name}")
    cmd = ["ffmpeg", "-y", 
           "-ss", str(from_second), 
           "-t", str(to_second-from_second), 
           "-i", videofile, 
           "-map 0 -c copy", out_v,
           "-map 0:a -acodec pcm_s16le", out_a
    ] 
    print(cmd)
    cmd = " ".join(cmd)
    os.system(cmd)
    return out_v, out_a

if __name__ == "__main__":
    # _yt_dlp_download("https://www.youtube.com/watch?v=G5ERdrjBe40", "720", "static/videos")
    # download_audio_file("https://www.youtube.com/watch?v=G5ERdrjBe40", "static/audio")
    cut_video("static/videos/G5ERdrjBe40.mp4", 10, 20)
