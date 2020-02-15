智慧小巴 AI帶你PAPAGO!
--
![Alt text](/Image/圖片1.png)

What we have done in this project:
--
<ol>
* Autopilot with obstacle avoidance
* Pedestrian detection
* Hardware and software integration



Hardware:
--
* Raspberry Pi 3 Model B
* Raspberry Pi Camera Module
* PCA 9685
* Movidius Neural Compute Stick 
* PS4 joystick
* Arduino
* Ultrasonic Distance Sensor

<a href=http://docs.donkeycar.com/>Quick start :</a>
--
>#### <ul>1. Setup your environment and necessary packages :</ul>

>#### <ul>2. Calibrate steering and throttle:</ul>
>> ##### <u1>1. 確認channel 0和1與轉向和油門間的對應關係(端看接線方式決定)，更改並記錄於myconfig.py內</u1>
>> ##### <u1>2. 運行donkey 套件內的calibrate</u1>
>> ##### <u1>3. 當校正轉向時，由360開始向上增加或向下遞減，找到轉向最左及最右數值，更改並記錄於myconfig.py內</u1>
>> ##### <u1>3. 當校正油門時，由370開始向上增加或向下遞減，找到適當速度之油門數值，更改並記錄於myconfig.py內</u1>

>#### <ul>3. Get driving:</ul>
>> ##### <u1>1. 於命令視窗切換到專案資料夾內</u1>
>> ##### <u1>2. 與手把連線後，運行以下py檔</u1>
>>><pre>python manage_ori.py drive --js</pre>

>#### <ul>4. Collect data and train an autopilot :</ul>
>>##### <ul>使用手把驅動自走車時，會自動將pi camera所視存為圖片，以及此刻轉向及油門存為json格式</ul>

![Alt text](/Image/圖片2.JPG)

Add on your own function parts :
--
### <pb>定義一個功能類別函式</pb>
### <pb>每個獨立part須具有以下幾個function : </pb>
><pre> __init__ , update(選配) , run or run_threaded , shutdown </pre>

<pre>__init__: 定義並初始化part參數</pre>
<pre>shutdown: 當主程序被停止時，該Part會被呼叫使用的function</pre>

### <pb>接下來分為兩種情形</pb>
>#### <pb>1.如該Part所需要的運算時間可能大於所設定之主程序迴圈週期時 - 使用run\_threaded搭配update</pb>

>><pre>update : 
該function內命令會被啟用，並在背景執行，使得主程序迴圈不會因為其所需較長的運算時間而拖垮整個迴圈的運行速度</pre>

>><pre>run_threaded :
主程序呼叫到該Part時，必會運行其內的命令，其內容不宜繁複耗時，通常用為將update所更新的最新part參數回傳給主程序使用
</pre>

>#### <pb>2.如該Part所需要的運算時間可能小於所設定之主程序迴圈週期時 - 使用run</pb>
>><pre>run :
主程序呼叫到該Part時，必會運行其內的命令，其內容不宜繁複耗時
</pre>

#### <pb>Sample Code : </pb>
<pre>
class MobileSSD(object):

    def __init__(self):
        self.on=True
        self.CLASSES=["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        self.COLORS= np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
        print("[INFO] loading success")
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        self.frame=None
        self.recent=None
        self.detect=False
        self.station=False



    def update(self):
        while self.on:
            try:
                if self.frame is not None:
                    #self.recent=cv2.resize(self.frame,(60,60),cv2.INTER_AREA)
                    person=False
                    bottle=False
                    (h, w) = self.frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(self.frame, 0.007843, (300, 300), 127.5)
                    self.net.setInput(blob)
                    detections = self.net.forward()
                    for i in np.arange(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.55:
                            idx = int(detections[0, 0, i, 1])
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            label = "{}: {:.2f}%".format(self.CLASSES[idx],confidence * 100)
                            if "person" in label:
                                #area=(endX-startX)*(endY-startY)
                                #if area>15000:
                                person=True
                            if "bottle" in label:
                                bottle=True
                            cv2.rectangle(self.frame, (startX, startY), (endX, endY),self.COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(self.frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
                    #cv2.putText(self.frame,"Just for test",(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.2,(255,0,0))
                    self.recent=self.frame
                    self.detect=person
                    self.station=bottle
            except:
                #print('serial.serialutil.SerialException from Lidar. common when shutting down.')
                print("something wrong")
                time.sleep(10)
                

    def run_threaded(self,frame=None):
        self.frame=copy.copy(frame)
        #cv2.imwrite("Frame.jpg",self.frame)
        if self.recent is not None:
            pic=self.recent
            status=self.detect
            pos=self.station
        else:
            pic=self.frame
            status=self.detect
            pos=self.station
        return pic,status,pos

    def shutdown(self):
        self.on = False
        time.sleep(2)
</pre>


Get start with our pre-trained model :
--
>#### <ul>Requirement:</ul>
>>##### <ul><a href=https://www.pyimagesearch.com/2019/04/08/openvino-opencv-and-movidius-ncs-on-the-raspberry-pi/>OpenVINO</a></ul>
>>##### <ul>Python 3.7.2</ul>
>>##### <ul>OpenCV 4.1.2</ul>
>>##### <ul>Tensorflow 1.13.1</ul>


>#### <ul>Script shell:</ul>
>>##### <ul>manage_04.py : 專案主程序</ul>
>>##### <ul>arduino.py : 透過Serial Port傳輸超音波測距結果</ul>
>>##### <ul>MobileSSD.py : Mobile_SSD物件偵測，回傳是否偵測行人</ul>
>>##### <ul>HoldOnMode.py : 當偵測到行人時，由自走模式切換到載客模式，並維持該模式</ul>
>>##### <ul>Resize.py : 由於自走車模型input為160*120之image，若更改myconfig內照片解析度時，則須運行該前處理</ul>
>>##### <ul>keras.py : 自走車keras模型，使用前覆蓋於donkey/utils內的keras.py</ul>

>#### <ul>Usage:</ul>
>> ##### <u1>1. 於命令視窗切換到專案資料夾內</u1>
>> ##### <u1>2. 執行以下命令</u1>
>>><pre>python manage_4.py drive --model 2020Final_T1.h5</pre>

>> ##### <ul>3. 開啟瀏覽器，前往[your car's ip address:8887]，切換模式為Autopilot即可</u1>

Maintainers :
--
* <b>中龍鋼鐵股份有限公司: 蔡天翔</b>
* <b>台灣積體電路製造股份有限公司: 羅世璋</b>
* <b>漢鼎智慧科技股份有限公司: 漳彥皓</b>
* <b>漢鼎智慧科技股份有限公司: 林瑋鑫</b>




