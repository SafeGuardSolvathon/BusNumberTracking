
# import cv2 to helo load our image
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
import easyocr
model=YOLO('YOLOFin/best.pt')
import threading
import requests
cp = cv2.VideoCapture(0)
import sys
import time
def draw_box(frame, top_left, bottom_right, color=(0, 255, 0), thickness=2):
  """
  Draws a rectangle on a cv2 frame.

  Args:
      frame: The cv2 frame as a numpy array.
      top_left: A tuple (x, y) representing the top-left corner of the box.
      bottom_right: A tuple (x, y) representing the bottom-right corner of the box.
      color: The color of the box in BGR format (default: green).
      thickness: The thickness of the box lines in pixels (default: 2).
  """
  cv2.rectangle(frame, top_left, bottom_right, color, thickness)
def read_license_plate(cropped_input):
    reader = easyocr.Reader(['en'])  # Replace 'en' with your desired language code

  # Read text from the image
    results = reader.readtext(cropped_input)

  # Return the list of detected text elements
    return results
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

global images_rec
images_rec=[]
api_key = "e6a7b91ab288957"

def ocr_space_file(filename, overlay=False, api_key='helloworld', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r.content.decode()

def check_ocr(images_rec):
    def create_and_store_temp_image(image):
        """
        Creates a temporary image file, stores the OpenCV image data, and returns the file path.

        Args:
            image: The OpenCV image data (numpy array).

        Returns:
            The path to the temporary image file.
        """

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, image)
            return temp_file_path

    while(True):
        time.sleep(2)
        if len(images_rec)>0:
            print("\n\n\n\nStarted\n\n\n\n\n")
            p=create_and_store_temp_image(images_rec[0])
            try:
                text=ocr_space_file(p,api_key=api_key)
                print(f"\n\n\n\n{text}\n\n\n\n")
                os.remove(p)
                images_rec=[]
            except Exception as e:
                print(f"\n\n\n{e}\n\n\n")
            sys.exit()




t1 = threading.Thread(target=check_ocr,args=(images_rec,))
def everthying():
    
    while(True):
        camera, frame = cp.read()
        if frame is not None:
            
            results=model(frame)[0]
            for detection in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
                
                if(score>0.6):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    license_plate_crop = frame[y1:y2, x1: x2, :]
                    print(type(license_plate_crop))
                        # process license plate
                    images_rec.append(license_plate_crop)
                    '''
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY),
                    print(type(license_plate_crop_gray))
                    if type(license_plate_crop_gray) == tuple:
                        license_plate_crop_gray = np.array(license_plate_crop_gray)
                    license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    '''
                        # read license plate number
                    
                    #license_plate_text= read_license_plate(license_plate_crop)
                    #print(license_plate_text)
                    img=license_plate_crop
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
                    edges = auto_canny(thresh_inv)
                    ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
                    img_area = img.shape[0]*img.shape[1]
                    for i, ctr in enumerate(sorted_ctrs):
                        x, y, w, h = cv2.boundingRect(ctr)
                        roi_area = w*h
                        roi_ratio = roi_area/img_area
                        if((roi_ratio >= 0.015) and (roi_ratio < 0.09)):
                            if ((h>1.2*w) and (3*w>=h)):
                                cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
                    
                    #license_plate_text= read_license_plate(license_plate_crop)
                    #print(license_plate_text)
            
            cv2.imshow("Frame", frame)

                
                    # read license plate number
                

        q = cv2.waitKey(1)
        if q==ord("q"):
            t1.join()
            
            break
    cv2.destroyAllWindows()
    sys.exit()


t2=threading.Thread(target=everthying,args=())
t2.start()
t1.start()


