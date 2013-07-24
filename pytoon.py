import cv
import cv2
import numpy as np

def array2cv(a):
  dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
  try:
    nChannels = a.shape[2]
  except:
    nChannels = 1
  cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
          dtype2depth[str(a.dtype)],
          nChannels)
  cv.SetData(cv_im, a.tostring(),
             a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im

def blur(frame):
    out = cv.CloneImage(frame)
    cv.Smooth(frame, out, cv.CV_MEDIAN, 7)
    return out

def laplace_edge_detection(frame):
    l = cv.CloneImage(frame)
    cv.Laplace(l, l)
    return l

def haar_detection(frame):
    image = cv.CloneImage(frame)

    haars = (
        cv.Load('cascades/haarcascade_eye.xml'),
        cv.Load('cascades/haarcascade_frontalface_default.xml'),
    )
    
    detected = False
    for detection in haars:
        storage = cv.CreateMemStorage()
        detectedBody = cv.HaarDetectObjects(image, detection, storage, flags=cv.CV_HAAR_DO_CANNY_PRUNING)

        if detectedBody:
            detected = True
            for body in detectedBody:
                cv.Rectangle(image,(body[0][0],body[0][1]),
                    (body[0][0]+body[0][2],body[0][1]+body[0][3]),
                    cv.RGB(255, 0, 0),2)

    return (image, detected)


def hog_detection(frame):
    img = cv.CloneImage(frame)

    from glob import glob
    import itertools as it
    
    def inside(r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    def draw_detections(img, rects, thickness = 1):
        for x, y, w, h in rects:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


    hog = cv2.HOGDescriptor()
    detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
    # detector = cv2.HOGDescriptor_getDaimlerPeopleDetector()

    hog.setSVMDetector(detector)

    if type(img) == cv2.cv.iplimage:
        img = np.asarray(img[:,:])        # convert from IplImage to Numpy array

    hog_params = {
        # 'winStride' : (8, 8),
        'padding' : (32, 32),
        'scale' : 1.05,
    }

    found, w = hog.detectMultiScale(img, **hog_params)
    found_filtered = []

    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
            else:
                found_filtered.append(r)
    draw_detections(img, found)
    draw_detections(img, found_filtered, 3)

    detected = len(found) > 0
    return img, detected


def process_frames(capture):

    # # get frame 
    frame = cv.QueryFrame(capture)
    size = cv.GetSize(frame)
    orig = cv.CreateImage((size[1] / 2, size[0] / 4), frame.depth, frame.channels)

    # resize so its faster
    cv.Resize(frame, orig)

    # blur the frame
    blurred_out = blur(orig)

    # laplace edge detection
    edged_out = laplace_edge_detection(orig)

    # haar cascades - detection
    haars_out, detected = haar_detection(orig)

    # hog descriptor - detection
    hog_out, detected = hog_detection(orig)


    # display images
    cv.ShowImage("HOG Descriptor", array2cv(hog_out))
    cv.ShowImage("Haar Cascades", haars_out)
    cv.ShowImage("Canny Edge Detection", edged_out)
    cv.ShowImage("Median Blur", blurred_out)
    cv.ShowImage("Original", orig)

def main():

    capture = cv.CaptureFromCAM(0)
    cv.NamedWindow("Original", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("Median Blur", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("Canny Edge Detection", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("Haar Cascades", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("HOG Descriptor", cv.CV_WINDOW_AUTOSIZE)


    while True:
        process_frames(capture)

        if cv.WaitKey(33) == 27:
            break

    cv.DestroyAllWindows()

if __name__ == "__main__":
    main()

