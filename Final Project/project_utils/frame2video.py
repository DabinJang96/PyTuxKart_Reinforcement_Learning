import cv2 

def frame2video(frames, fname):
    '''
    Create an output video with ".avi extension"
    from frames of equal shape

    :param frames: array of images with 1 or 3 channels
    :param fname: string with ".avi" extension
    
    :return: void
    '''
    writer = cv2.VideoWriter(fname,
        cv2.VideoWriter_fourcc('M','J','P','G'), 
        10, frames[0].shape[:2])

    for i in range(len(frames)):
        writer.write(frames[i])

    writer.release()