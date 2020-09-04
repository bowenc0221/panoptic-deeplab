import cv2
import glob
import numpy as np
from moviepy.editor import *


class LoadMovieOrImages:
    """
    video or images
    Video supports frame interval and time interval, select 1 from 2 and time_step takes precedence when used at the same time
    @param frame_step:frame interval (long)
    @param time_step:time interval (second float)
    @param start_time:video start time (second float)
    @param end_time:video end time (second float)
    """

    def __init__(self, path, img_size: int = 640, frame_step: int = None, time_step: float = None,
                 start_time: float = None, end_time: float = None):
        img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
        vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']
        path = str(path)
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        self.frame_step = frame_step
        self.time_step = time_step
        self.start_time = start_time
        self.end_time = end_time
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.vfc = None
        assert self.nF > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (path, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # video
            self.mode = 'video'
            # img0 = self.iter_frame.__next__()
            frame_loc = self._frame_c * self.item_time
            if frame_loc > (self.nframes / self.fps):
                raise StopIteration
            # It's the end time
            if self.end_time is not None and frame_loc >= self.end_time:
                raise StopIteration
            img0 = self.vfc.get_frame(frame_loc)  # The parameter is seconds as the unit can be a decimal
            if img0 is None:  # Video is over
                self.count += 1
                self.vfc.close()
                if self.count == self.nF:  # The files are all processed
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    # img0 = self.iter_frame.__next__()
                    img0 = self.vfc.get_frame(frame_loc)

            if self.frame_step is not None and self.frame_step > 0:
                self.frame = self._frame_c * self.frame_step
            else:
                self.frame += 1
            if self.time_step is not None and self.time_step > 0:
                self.frame = self.fps * frame_loc

            self._frame_c += 1
            img0 = np.ascontiguousarray(img0.copy())  # RGB
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # pictures
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            img0 = np.ascontiguousarray(img0[:, :, ::-1])  # BGR to RGB
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img = self.letterbox(img0, new_shape=self.img_size)[0]

        # Format conversion
        img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.vfc

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def new_video(self, path, audio=False):
        self.frame = 0  # Current frame
        self._frame_c = 0  # How many frames were processed
        self.vfc = VideoFileClip(path, audio=audio)
        # self.iter_frame = self.vfc.iter_frames()
        self.nframes = max(int(self.vfc.duration * self.vfc.fps), 0)
        self.fps = self.vfc.fps
        self.item_time = 1 / self.fps
        # frame interval
        if self.frame_step is not None and self.frame_step > 0:
            self.item_time = self.item_time * self.frame_step
        # time interval (second)
        if self.time_step is not None and self.time_step > 0:
            self.item_time = self.time_step
        # Start time
        if self.start_time is not None and self.start_time > 0:
            self._frame_c = self.start_time / self.item_time
            if self.frame_step is not None and self.frame_step > 0:
                self.frame = self._frame_c * self.frame_step
            else:
                self.frame = self._frame_c
            if self.time_step is not None and self.time_step > 0:
                self.frame = self.fps * self._frame_c * self.item_time
        # End Time
        if self.end_time is not None and self.end_time > 0:
            if self.start_time is not None and self.start_time > 0:
                if self.end_time < self.start_time:
                    self.end_time = None  # The time is wrong, go straight to the end of the video
                    print("end_time must be greater than start_time")

    def __len__(self):
        return self.nF  # number of files
