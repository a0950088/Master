from pytube import YouTube
from pytube import Playlist
import subprocess
import os

path = "./piano_accompaniment/"
all_file_name = os.listdir(path)
playlist_url = 'https://youtube.com/playlist?list=PLVzcEcYVk_Bu-az-UE9YCoVFsfgq3idku&si=t8iMh7ID1hyDZy3z'
single_url = 'https://www.youtube.com/watch?v=XwiNJxxO0II'
def DownloadSingleYtSource(url):
    yt = YouTube(url)
    file_name = (yt.title).replace('/',' ').replace('"',' ')
    try:
        yt.streams.filter().get_audio_only().download(filename=f'url.mp3')
        subprocess.call(['ffmpeg', '-i', f'url.mp3', f'ok.wav'])
        print("Done")
    except:
        print(f"try again ...")
        
def DownloadYtPlaylist(url):
    playlist = Playlist(url)
    urls = playlist.video_urls

    for i in range(len(urls)):
        yt = YouTube(urls[i])
        print(f"download {i}: {yt.title} ...")
        file_name = (yt.title).replace('/',' ').replace('"',' ')
        if file_name+'.wav' in all_file_name:
            print("already in the path")
        else:
            try:
                yt.streams.filter().get_audio_only().download(filename=f'url.mp3')
                subprocess.call(['ffmpeg', '-i', f'url.mp3', f'./piano_accompaniment/{file_name}.wav'])
            except:
                print(f"try again ... {i}")
                i-=1
                print(i)
    print("ok!")

# DownloadYtPlaylist(playlist_url)
DownloadSingleYtSource(single_url)

