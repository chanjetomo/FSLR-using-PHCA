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

    # importing video from path
    cap = cv2.VideoCapture(self.path)

    # video properties
    n_frames, height, weight = get_video_properties(cap)

    # frame interval
    frame_int = 1

    # create placeholders
    [prev_frame, diff_frame, sum_frame] = frame_placeholders(3, height, weight)

    for frame in range(n_frames):
      ret, img = cap.read()
      if ret == False:
        break
      if frame % frame_int == 0:
        
        curr_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # save and pass first frame
        if frame == 0:
          prev_frame = curr_frame
        else:
          # perform frame difference
          diff_frame = cv2.subtract(curr_frame, prev_frame)

          # reduce noise
          diff_frame = reduce_noise(diff_frame, upperthresh=200, lowerthresh=50)

          # save temporal features
          diff_frame = save_temporal_features(diff_frame, upperthresh=255, frame_idx=frame)

          sum_frame = cv2.add(sum_frame, diff_frame)
          prev_frame = curr_frame
    
    cap.release()
    return sum_frame.flatten()

def get_video_properties(cap):
  ''' Returns number of frames, height, and weight of a video'''
  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  ret, frame = cap.read()
  if ret:
    return n_frames, frame.shape[0], frame.shape[1]

def frame_placeholders(n, height, width):
  """ Returns a list of length n with entries np.zeros((height, width), dtype.uint8) """
  pholder = []
  for i in range(n):
    pholder.append(np.zeros((height, width, 3), dtype=np.uint8))
  return pholder

def frame_to_rgb(frame):
  """ Splits the frame into arrays of R, G, and B """
  r, g, b = cv2.split(frame)
  return np.asarray([r, g, b])

def reduce_noise(frame, upperthresh, lowerthresh):
  """ Changes pixel value to 0 if value > upperthresh or value < lowerthresh """
  colors = frame_to_rgb(frame)
  for color in colors:
    color[color < lowerthresh] = 0
    color[color > upperthresh] = 0
  return cv2.merge([colors[0],colors[1], colors[2]])

def save_temporal_features(frame, upperthresh, frame_idx):
  """ Saves the temporal features of the frame difference """
  colors = frame_to_rgb(frame)
  for color in colors:
    color[color!=0] = np.floor(upperthresh/frame_idx)
  return cv2.merge([colors[0],colors[1], colors[2]])