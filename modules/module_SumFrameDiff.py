import cv2
import numpy as np

class sumFrameDiff:
    def __init__(self):
        pass
    
    def SumFrameDiff(self, path):
        """ 
        Perform Sum of Frame Differences on a video file
        
        Parameter
        -----------
        path: file path of video
        
        Return
        -----------
        flattened array of sum of frame differences
        """

        self.path = path
        cap = cv2.VideoCapture(self.path)
        n_frames, height, width = get_video_properties(cap)
        frame_int = 1
        [prev_frame, diff_frame, sum_frame] = frame_placeholders(3, height, width)
        
        for frame in range(n_frames):
            ret, img = cap.read()
            if ret == False:
                break

            if frame % frame_int == 0:
                curr_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if frame == 0:
                    prev_frame = curr_frame
                else:
                    # frame difference
                    diff_frame = cv2.subtract(curr_frame, prev_frame)

                    # reduce noise and save temporal features
                    diff_frame = reduce_noise(diff_frame, upperthresh=200, lowerthresh=55)
                    diff_frame = save_temporal_features(diff_frame, upperthresh=255, frame_idx=frame)

                    # save in sum frame
                    sum_frame = cv2.add(sum_frame, diff_frame)
                    prev_frame = curr_frame

        cap.release
        return sum_frame.flatten()

def get_video_properties(cap):
    """ Returns number of frames, height, and width of a video """
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    if ret:
        return n_frames, frame.shape[0], frame.shape[1]

def frame_placeholders(n, height, width):
    """ Returns a list of length n with entries np.zeros((height, width), dtype.uint8) """
    pholder = []
    for i in range(n):
        pholder.append(np.zeros((height, width), dtype=np.uint8))
    return pholder

def reduce_noise(frame, upperthresh, lowerthresh):
    """ Changes pixel value to 0 if value > upperthresh or value < lowerthresh """
    frame[frame < lowerthresh] = 0
    frame[frame > upperthresh] = 0
    return frame

def save_temporal_features(frame, upperthresh, frame_idx):
    """ Saves the temporal features of the frame difference """
    frame[frame!=0] = np.floor(upperthresh/frame_idx)
    return frame