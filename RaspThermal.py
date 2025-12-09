#!/usr/bin/env
import cv2
import numpy as np
import argparse
import time
import io
import telepot
chat_id = 547
bot = telepot.Bot('7972098389:A8_28tYCndJyrVdwMM')

def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception: pass
    return False
def alert():
    bot.sendMessage(chat_id, 'start')
    return()


isPi = is_raspberrypi()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
args = parser.parse_args()

if args.device:
    dev = args.device
else:
    dev = 0
    
#init video
cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)
#cap = cv2.VideoCapture(0)
#pull in the video but do NOT automatically convert to RGB, else it breaks the temperature data!
#https://stackoverflow.com/questions/63108721/opencv-setting-videocap-property-to-cap-prop-convert-rgb-generates-weird-boolean
if isPi == True:
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
else:
    cap.set(cv2.CAP_PROP_CONVERT_RGB, False)

#256x192 General settings
width = 256 #Sensor width
height = 192 #sensor height
scale = 3 #scale multiplier
newWidth = width*scale 
newHeight = height*scale
alpha = 1.0 # Contrast control (1.0-3.0)
colormap = 0
font=cv2.FONT_HERSHEY_SIMPLEX
dispFullscreen = False
cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Thermal', newWidth,newHeight)
rad = 0 #blur radius
threshold = 2
hud = True
recording = False
elapsed = "00:00:00"
snaptime = "None"
alert()

def rec():
    now = time.strftime("%Y%m%d--%H%M%S")
    #do NOT use mp4 here, it is flakey!
    videoOut = cv2.VideoWriter(now+'output.avi', cv2.VideoWriter_fourcc(*'XVID'),25, (newWidth,newHeight))
    return(videoOut)

def snapshot(heatmap):
    #I would put colons in here, but it Win throws a fit if you try and open them!
    now = time.strftime("%Y%m%d-%H%M%S") 
    snaptime = time.strftime("%H:%M:%S")
    cv2.imwrite("TC001"+now+".png", heatmap)
    return snaptime
 

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #alert()
    if ret == True:
        imdata,thdata = np.array_split(frame, 2)
        hi = thdata[96][128][0]
        lo = thdata[96][128][1]
        lo = lo*256
        rawtemp = hi+lo
        temp = (rawtemp/64)-273.15
        temp = round(temp,2)

        #find the max temperature in the frame
        lomax = thdata[...,1].max()
        posmax = thdata[...,1].argmax()
        mcol,mrow = divmod(posmax,width)
        himax = thdata[mcol][mrow][0]
        lomax=lomax*256
        maxtemp = himax+lomax
        maxtemp = (maxtemp/64)-273.15
        maxtemp = round(maxtemp,2)
        
        #find the lowest temperature in the frame
        lomin = thdata[...,1].min()
        posmin = thdata[...,1].argmin()
        lcol,lrow = divmod(posmin,width)
        himin = thdata[lcol][lrow][0]
        lomin=lomin*256
        mintemp = himin+lomin
        mintemp = (mintemp/64)-273.15
        mintemp = round(mintemp,2)

        #find the average temperature in the frame
        loavg = thdata[...,1].mean()
        hiavg = thdata[...,0].mean()
        loavg=loavg*256
        avgtemp = loavg+hiavg
        avgtemp = (avgtemp/64)-273.15
        avgtemp = round(avgtemp,2)

        bgr = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.convertScaleAbs(bgr, alpha=alpha)#Contrast
        bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)#Scale up!
        if rad>0:
            bgr = cv2.blur(bgr,(rad,rad))

        #apply colormap
        if colormap == 0:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
            cmapText = 'Jet'
        # draw crosshairs
        cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
        (int(newWidth/2),int(newHeight/2)-20),(255,255,255),2) #vline
        cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
        (int(newWidth/2)-20,int(newHeight/2)),(255,255,255),2) #hline

        cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
        (int(newWidth/2),int(newHeight/2)-20),(0,0,0),1) #vline
        cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
        (int(newWidth/2)-20,int(newHeight/2)),(0,0,0),1) #hline
        #show temp
        cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

        if hud==True:
            if recording == False:
                cv2.putText(heatmap,'Recording: '+elapsed, (10, 112),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,(200, 200, 200), 1, cv2.LINE_AA)
            if recording == True:
                cv2.putText(heatmap,'Recording: '+elapsed, (10, 112),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,(40, 40, 255), 1, cv2.LINE_AA)
        
        #Yeah, this looks like we can probably do this next bit more efficiently!
        #display floating max temp
        if maxtemp > avgtemp+threshold:
            cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,0), 2)
            cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,255), -1)
            cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
            cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)
            
        if maxtemp > 30:
            snaptime=snapshot(heatmap)

        #display floating min temp
        if mintemp < avgtemp-threshold:
            cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (0,0,0), 2)
            cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (255,0,0), -1)
            cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
            cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

        #display image
        cv2.imshow('Thermal',heatmap)

        if recording == True:
            elapsed = (time.time() - start)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed)) 
            #print(elapsed)
            videoOut.write(heatmap)
        
        keyPress = cv2.waitKey(1)
        if keyPress == ord('a'): #Increase blur radius
            rad += 1
        if keyPress == ord('z'): #Decrease blur radius
            rad -= 1
            if rad <= 0:
                rad = 0

        if keyPress == ord('s'): #Increase threshold
            threshold += 1
        if keyPress == ord('x'): #Decrease threashold
            threshold -= 1
            if threshold <= 0:
                threshold = 0

        if keyPress == ord('d'): #Increase scale
            scale += 1
            if scale >=5:
                scale = 5
            newWidth = width*scale
            newHeight = height*scale
            if dispFullscreen == False and isPi == False:
                cv2.resizeWindow('Thermal', newWidth,newHeight)
        if keyPress == ord('c'): #Decrease scale
            scale -= 1
            if scale <= 1:
                scale = 1
            newWidth = width*scale
            newHeight = height*scale
            if dispFullscreen == False and isPi == False:
                cv2.resizeWindow('Thermal', newWidth,newHeight)

        if keyPress == ord('q'):
            break
            capture.release()
            cv2.destroyAllWindows()
        
