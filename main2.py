from AudioAnalyzer import *
import random
import colorsys
import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import os
from natsort import natsorted

song_list = natsorted(os.listdir("songs/playlist2"))
print(song_list)

for song in tqdm(song_list):
    filename = "songs/playlist2/" + song

# filename = "songs/운이좋았지.mp3"

    y, sr = librosa.load(filename, sr=22050)

    song_length = int(y.shape[0]/sr)
    print(f'song length: {song_length}')

    analyzer = AudioAnalyzer()
    analyzer.load(filename)

    # print(analyzer.spectrogram.shape)
    # print(y.shape[0] / 512)

    spectogram = np.swapaxes(analyzer.spectrogram, 0, 1)

    print(spectogram.shape)

    WIDTH = 1920
    HEIGHT = 1080
    x, y = int(WIDTH/2)-230, int(2*HEIGHT/3) - 70

    # bg_img = cv2.imread('pics/autumn.jpg')
    bg_img = cv2.imread('pics/seoul_pic.jpg')
    resized = cv2.resize(bg_img, dsize=(WIDTH,HEIGHT))
    rect_width = 4
    bar_gap = 5

    radius =10
    color = (230,230,230)

    # radio_img = cv2.imread('pics/radio2.png')

    # # make transparent radio image
    # tmp = cv2.cvtColor(radio_img, cv2.COLOR_BGR2GRAY)

    # _,alpha = cv2.threshold(tmp,20,255,cv2.THRESH_BINARY)
    # b, g, r = cv2.split(radio_img)
    # rgba = [b,g,r, alpha]
    # dst = cv2.merge(rgba,4)
    # cv2.imwrite("pics/radio_trans.jpg", dst)

    # radio_h, radio_w, _ = radio_img.shape
    # print(radio_w, radio_h)
    # radio_x, radio_y = x - 100, y - 100
    # resized[radio_y:radio_y+radio_h,radio_x:radio_x+radio_w,:] = radio_img

    fps = int(spectogram.shape[0] / song_length)
    print(f'fps: {fps}')
    frames = []
    save_name = os.path.basename(filename[:-4])
    print(save_name)
    writer = cv2.VideoWriter(f'videos/2/{save_name}.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (WIDTH,HEIGHT))
    for frequency in tqdm(spectogram):
        frequency += 80
        bg = resized.copy()
        rect_x = x
        for idx, f in enumerate(frequency[:int(2*len(frequency)/3)]):
            # cv2.circle(bg, (int((2*rect_x + rect_width)/2), y+int(0.8*f/2+100)), radius, color, -1)
            cv2.rectangle(bg, (rect_x, y+int(0.8*f/2)), (rect_x + rect_width, y-int(0.8*f/2)), color, 2)
            rect_x += rect_width + bar_gap
        # frames.append(bg)
        writer.write(bg)
        # cv2.imshow('frame', bg)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    # cv2.destroyAllWindows()

    writer.release()