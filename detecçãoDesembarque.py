import numpy as np
import cv2
import time


#funcao para calcular o centro do contorno
def center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy

#Opcoes das linhas
posL = 125
offSet = 30
limiteVerticalEsquerdo = 90
limiteVerticalDireito = 420
xy1 = (limiteVerticalEsquerdo, posL)
xy2 = (limiteVerticalDireito, posL)
scale = 46
tempoSeguranca = 3
tempo = time.time()
porta = 1

######
#variaveis
detects = []
#contagemq
up = 0
detectionCounter = 0
cap = cv2.VideoCapture('/home/pi/compartilhamento/9.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()
person_cascade = cv2.CascadeClassifier('/home/pi/compartilhamento/opencv/Treino7.xml')
while 1:
    
    ret, frame = cap.read()
    frame = cv2.resize(frame,(480,320)) # Downscale to improve frame rate
    #get the webcam size
    heightCam, widthCam, channels = frame.shape

    #prepare the crop
    centerX,centerY=int(heightCam/2),int(widthCam/2)
    radiusX,radiusY= int(scale*heightCam/100),int(scale*widthCam/100)

    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    cropped = frame[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (widthCam, heightCam))
    #converter imagem para cinza
    gray = cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2GRAY)
    rects = person_cascade.detectMultiScale(gray,1.1,3)

    i = 0 #id da contagem

    for (x, y, w, h) in rects:
        centro=center(x,y,w,h)
        #gerar contorno e circulo
        if centro[0] > limiteVerticalEsquerdo and centro[0] < limiteVerticalDireito:
            cv2.circle(resized_cropped, centro, 1, (0,0,255), -1)
            cv2.rectangle(resized_cropped, (x,y), (x+w, y+h), (0,255,0),2)
            #gerar codigo do objeto
            cv2.putText(resized_cropped, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        #incrementar id
        if len(detects) <= i:
            detects.append([])
        if centro[1]> posL - offSet and centro[1] < posL + offSet and centro[0] > limiteVerticalEsquerdo and centro[0] < limiteVerticalDireito:
            detects[i].append(centro)
        else:
            detects[i].clear()

        i += 1
        tempo = time.time()
    if len(rects) == 0:
        detects.clear()
        tempoAtual = time.time()
        if((tempoAtual-tempoSeguranca >= tempo) & porta == 1):
            cv2.putText(resized_cropped, "Desembarque Finalizado", (40,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    else:
        tempo = time.time()
        if(porta == 1):
            cv2.putText(resized_cropped, "Desembarcando", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        for detect in detects:
            for (c,l) in enumerate(detect):
                #verificar se o objeto subiu
                if detect[c-1][1] < posL and l[1] > posL :
                    detect.clear()
                    up+=1
                    cv2.line(resized_cropped,xy1,xy2,(0,255,0),5)
                    continue
                #desenha a linha que segue o objeto
                if c > 0:
                    cv2.line(resized_cropped,detect[c-1],l,(0,0,255),1)


    #retirar contorno

    #gerar linhas
    cv2.line(resized_cropped, xy1, xy2,(255,0,0),3)
    cv2.line(resized_cropped,(xy1[0], posL - offSet), (xy2[0], posL - offSet), (255,255,0),2)
    cv2.line(resized_cropped,(xy1[0], posL + offSet), (xy2[0], posL + offSet), (255,255,0),2)
    cv2.line(resized_cropped,(limiteVerticalEsquerdo,0),(limiteVerticalEsquerdo,480),(0,0,255),2)
    cv2.line(resized_cropped,(limiteVerticalDireito,0),(limiteVerticalDireito,480),(0,0,255),2)


    #####

    #exibicoes
    #cv2.putText(resized_cropped, "Descendo: " + str(up), (120,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
    #cv2.putText(resized_cropped, "Detect: " + str(detectionCounter), (120,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
    #video
    cv2.imshow("frame", resized_cropped)
    #cv2.imshow("dilation", dilation)
    #cv2.imshow("blurVideo", gray)
    #cv2.imshow("mask", fgmask)
    
    if cv2.waitKey(90) & 0xFF == ord('1'):
       if porta == 1:
           porta = 0
       else:
           porta = 1
       print(porta)

    #if cv2.waitKey(90) & 0xFF == ord('q'):
     #   break

cap.release()
cv2.destroyAllWindows()
