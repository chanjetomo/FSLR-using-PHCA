import cv2
import numpy as np

class sumFrameDiff:
    def __init__(self):
        pass
    
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

    def SumFrameDiff(self, path):
        # importing video file from path
        cap = cv2.VideoCapture(path)

        # video properties
        n_frames, height, width = self.get_video_properties(cap)

        # frame interval
        frame_int = 1

        # create placeholders
        [prev_frame, diff_frame, sum_frame] = self.frame_placeholders(3, height, width)

        for frame in range(n_frames):
            ret, img = cap.read()
            if ret == False:
                break

            if frame % frame_int == 0:
                curr_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # save and pass the first frame
                if frame == 0:
                    prev_frame = curr_frame
                else:
                    # perform frame difference
                    diff_frame = cv2.subtract(curr_frame, prev_frame)

                    # reduce noise
                    diff_frame = self.reduce_noise(diff_frame, upperthresh=200, lowerthresh=55)

                    # save temporal features
                    diff_frame = self.save_temporal_features(diff_frame, upperthresh=255, frame_idx=frame)

                    # save in sum frame
                    sum_frame = cv2.add(sum_frame, diff_frame)
                    prev_frame = curr_frame
        
        cap.release()
        return sum_frame.flatten()