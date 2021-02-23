import cv2
import imutils
import easyocr
import numpy as np
import time
import os
import random
import sys


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture("video3.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0
count = 0

#recoding video
violations = 0
frameArray = []
frame_count=0
    
  
while True:
    #ret, frame = cap.read()
    #start_time = time.time()
    _, frame = cap.read()
    frame_id += 1
  
    
    #violation count-
    height,width=frame.shape[0:2]
    frame[0:70,0:width]=[0,0,255]
    cv2.putText(frame,'VIOLATION COUNT: ',(10,50),font,1.5,(255,255,255),2)

    #cross line
    cv2.line(frame,(0,height-200),(width,height-200),(0,255,265),2)
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    allowed_vehicles=['car','bus','','motorbike','truck']

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            if label in allowed_vehicles:
                
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y-10),font, 0.7, color, 2)

                #count
                vehiCy=int(y+h/2)
                lineCy=height-200
                if(vehiCy<lineCy+2 and vehiCy>lineCy-2):
                    #saving frames into array
                    violations +=1
                    if (len(frameArray) < violations):
                        frameArray.append([])
            
                    count = count+1
                    path = 'E:\\openCV\\ViolDetectwithANPR-master\\ViolDetectwithANPR-master\\save_img'
                    path2 = 'E:\\openCV\\ViolDetectwithANPR-master\\ViolDetectwithANPR-master\\save_rec'
                    roi = frame[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(path,'vehicle_'+label+'_'+str(count)+'.jpg'),roi)
                    cv2.line(frame,(0,height-200),(width,height-200),(0,0,255),2)
                    
                    #ANPR BEGIN----------------------------------------------------------------------------
                    img = cv2.imread(os.path.join(path,'vehicle_'+label+'_'+str(count)+'.jpg'))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #cv2.imshow('original img',gray)

                    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
                    edged = cv2.Canny(bfilter, 30, 200) #Edge detection
                    #cv2.imshow('Noise reduction img',edged)

                    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = imutils.grab_contours(keypoints)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

                    location = None
                    for contour in contours:
                        approx = cv2.approxPolyDP(contour, 10, True)
                        if len(approx) == 4:
                            location = approx
                            break

                    #print(location)

                    mask = np.zeros(gray.shape, np.uint8)
                    new_image = cv2.drawContours(mask, [location], 0,255, -1)
                    new_image = cv2.bitwise_and(img, img, mask=mask)
                    #cv2.imshow('mask img',new_image)

                    (x,y) = np.where(mask==255)
                    (x1, y1) = (np.min(x), np.min(y))
                    (x2, y2) = (np.max(x), np.max(y))
                    cropped_image = gray[x1:x2+1, y1:y2+1]
                    #cv2.imshow('cropped img',cropped_image)

                    reader = easyocr.Reader(['en'])
                    result = reader.readtext(cropped_image)
                    print(result)
                    #ANPR END------------------------------------------------------------------------------
                
                cv2.putText(frame,str(count),(500,50),font ,1.5,(255,255,255),2)
                
            #recoding video--------------------------------------------------------------------------------------   
            if violations > 0:
                if (len(frameArray[violations -1]) <180):#and (frame_count % 20 == 0)
                    frameArray[violations -1].append(frame)
                    #cv2.imwrite(os.path.join(path2,'vehicle_'+label+'_'+str(violations)+'_'+str(len(frameArray[violations -1]))+'.jpg'),frame)
                    if ( '_'+str(violations)+'.mp4' not in os.listdir(path2)):
                        writer= cv2.VideoWriter(os.path.join(path2,'_'+str(violations)+'.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
                    writer.write(frame)       

            frame_count+=1

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 100),font, 0.9, (255, 255, 255), 2)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
