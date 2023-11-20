import os
import json
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from decord import VideoReader
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type = str, default = 'tempo json')
parser.add_argument('--input_video_dir', type = str, default = 'video directory')
parser.add_argument('--output_csv', type = str, default = 'output csv')
parser.add_argument('--output_dir', type = str, default = 'output dir')

args = parser.parse_args()

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(path, num_frames=4):
    vr = VideoReader(path, height=224, width=224)
    total_frames = len(vr)
    frame_indices = get_index(total_frames, num_frames)
    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        images_group.append(img)
    return images_group

def main():

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    count = 0
    res = defaultdict(list)

    for j in tqdm(range(len(data))):
        instance = data[j]
        videoname = instance['video']
        videopath = os.path.join(args.input_video_dir, videoname)
        annotation_id = instance['annotation_id']
        if os.path.exists(videopath):
            if "before_" in annotation_id or 'after_' in annotation_id or 'then_' in annotation_id:
                context = instance['context']
                train_time = instance['train_times'][0]
                if context[0] <= train_time[0]:
                    a, b = context[0], context[1]
                    c, d = train_time[0], train_time[1]
                else:
                    c, d = context[0], context[1]
                    a, b = train_time[0], train_time[1]

                a = a * 5
                c = c * 5
                b = (b + 1) * 5
                d = (d + 1) * 5
                print(a, b, c, d)
                extn = videopath.split('.')[-1]
                targetvidpath = os.path.join(args.output_dir, f"{count}.{extn}")
                if c <= b:
                    try:
                        count += 1
                        load_video(videopath, num_frames= 4)
                        ffmpeg_extract_subclip(videopath, a, d, targetname = targetvidpath)
                        res['caption'].append(instance['description'])
                        res['original_videopath'].append(videoname)
                        res['videopath'].append(targetvidpath)
                    except:
                        print('broken video or extractor')
                else:
                    continue
                    ### tried making it work but not working for now
                    # clip = VideoFileClip(videopath)
                    # clip1 = clip.subclip(a, b)
                    # clip2 = clip.subclip(c, d)
                    # final = concatenate_videoclips([clip1, clip2])
                    # print(videopath, targetvidpath)
                    # final.write_videofile(targetvidpath)
        
    print(count)
    df = pd.DataFrame(res)
    df.to_csv(args.output_csv, index = False)         

if __name__ == "__main__":
    main()