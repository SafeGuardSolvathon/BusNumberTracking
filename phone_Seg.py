import numpy as np
import pandas as pd
import os
import cv2
import time
import random
import matplotlib.pyplot as plt
import requests
import json
import easyocr
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
import threading
labels = open('./input/yolo-weights-for-licence-plate-detector/classes.names').read()
weights_path = './input/yolo-weights-for-licence-plate-detector/lapi.weights'
configuration_path = './input/yolo-weights-for-licence-plate-detector/darknet-yolov3.cfg'

probability_minimum = 0.5
threshold = 0.3
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
layers_names_all = network.getLayerNames()
print(network.getUnconnectedOutLayers())
#layers_names_output = [layers_names_all[i[0]-1] for i in network.getUnconnectedOutLayers()]
#layers_names_output = [layers_names_all[network.getUnconnectedOutLayers() - 1]]


# Get the list of unconnected out layer indices
unconnected_out_layer_indices = network.getUnconnectedOutLayers()
# Extract layer names using the correct indices
layers_names_output = [layers_names_all[i - 1] for i in unconnected_out_layer_indices]
print(layers_names_output)
api_key = "e6a7b91ab288957"

def cv2_to_multipart(image):
  """
  Converts a cv2 image to a multipart file representation (without base64).

  Args:
      image: A cv2 image object.

  Returns:
      A tuple containing the filename and the image data as a multipart file.
  """
  # Encode image to PNG (you can choose other formats if needed)
  _, image_buffer = cv2.imencode('.png', image)
  # Get image data as bytes
  image_data = image_buffer.tobytes()
  # Define filename (optional, adjust as needed)
  filename = "image.png"
  return (filename, image_data)


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
    filename,image_data=cv2_to_multipart(filename)
    #with open(filename, 'rb') as f:
    r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: image_data},
                          data=payload,
                          )
    print(type(r.content.decode()))

    return json.loads(r.content.decode())

import re

def extract_number_plate(json_string):
  """
  Extracts the number plate from a JSON string response from an OCR API.

  Args:
      json_string: A string containing the JSON response data.

  Returns:
      A string containing the extracted number plate or None if not found.
  """
  try:
    # Load JSON data
    #data = json.loads(json_string)
    # Access the parsed text
    data=json_string
    parsed_text = data["ParsedResults"][0]["ParsedText"]
    # Extract the number plate (assuming it's on the first line)
    number_plate = parsed_text.strip().split("\r\n")[0]
    return number_plate
  except (json.JSONDecodeError, KeyError):
    print("Error: Invalid JSON data or missing key")
    return None


# Example usage

reader = easyocr.Reader(['en'], gpu=True)
def perform_ocr_on_image(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]

    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)

    text = ""
    for res in results:
        if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
            text = res[1]

    return str(text)
url = "http://100.79.67.94:4747/video"
cp = cv2.VideoCapture(0)
img=0
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
while(True):
    boxes = []
    confidences = []
    class_ids = []
    camera, frame = cp.read()
    if frame is not None:
        #image_input=cv2.imread()
        cv2.imshow("Frame", frame)
        img_copy=frame.copy()
        image_input = cv2.resize(frame,dsize=None,fx=0.2,fy=0.2)

        blob = cv2.dnn.blobFromImage(image_input, 1/255.0, (416,416), swapRB=True, crop=False)
        blob_to_show = blob[0,:,:,:].transpose(1,2,0)
        network.setInput(blob)
        output_from_network = network.forward(layers_names_output)
        np.random.seed(42)
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        bounding_boxes = []
        confidences = []
        class_numbers = []
        h,w = image_input.shape[:2]
        flag=0
        #print(output_from_network)
        for result in output_from_network:
            for detection in result:
                scores = detection[5:]
                class_current = np.argmax(scores)
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                confidence_current = scores[class_current]
                
                if confidence_current > probability_minimum:
                    flag=confidence_current
                    print(detection)
                    center_x = int(detection[0]* img_copy.shape[1])
                    center_y = int(detection[1]* img_copy.shape[0])
                    w = int(detection[2] * img_copy.shape[1])
                    h = int(detection[3] * img_copy.shape[0])
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    tl=(center_x-w//2,center_y-h//2)
                    br=(center_x+w//2,center_y+h//2)
                    draw_box(frame,tl,br,thickness=7)
                    class_ids.append(class_id)
                    
                    break
            if flag!=0:
                #cv2.imwrite(f"/images/{img}.jpg", frame)
                #print(extract_number_plate(ocr_space_file(frame)))
                print()
                break
    indices = cv2.dnn.NMSBoxes(boxes, confidences, probability_minimum, 0.0)

  # Check if any number plates were detected
    if len(indices) > 0:
        # Extract only the first detected plate (assuming one plate per image)
        for i in indices.flatten():
            (x, y, w, h) = boxes[i][:]
            plate_img = img_copy[y:y+h, x:x+w]

            # Use EasyOCR to recognize characters
            result = reader.readtext(plate_img)

            # Extract recognized text if available
            plate_text = None
            if result:
                plate_text = result[0][1]

            # Draw bounding box and display recognized text (if any)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if plate_text:
                cv2.putText(img_copy, plate_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            print(plate_text)
   

        #print(flag)
 
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cv2.destroyAllWindows()