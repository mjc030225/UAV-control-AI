import sys
import time
import os
from collections import deque

import asyncio
import cv2
import numpy as np
import torchvision.transforms.transforms as transforms
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
from threading import Thread, Event

from main_window_ui import Ui_MainWindow    # example这里是你的命名文件
from menu_ui import Ui_register_2
from register_ui import Ui_register_3
##########import face dlib module #############
from val_recognize.face_dlib.face_reco_from_camera import Face_Recognizer
from val_recognize.face_dlib.features_extraction_to_csv import main_tranfer
import HandTrackingModule as htm
from downstream_task.detection.model import build_yolo_people_detection
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityNedYaw)


class CameraStream:
    """Helper class for managing OpenCV capture and FPS tracking."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.frame_start_time = None
        self.start_time = None
        self.fps = 0.0
        self.fps_show = 0.0
        self.frame_time = 0.0
        self.ensure_camera()

    def ensure_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            now = time.time()
            self.frame_start_time = now
            self.start_time = now

    def read(self):
        self.ensure_camera()
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def update_fps(self) -> float:
        now = time.time()
        if self.frame_start_time is None:
            self.frame_start_time = now
        if self.start_time is None:
            self.start_time = now
        if int(self.start_time) != int(now):
            self.fps_show = self.fps
        self.frame_time = now - self.frame_start_time if self.frame_start_time else 0.0
        if self.frame_time > 0:
            self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        self.start_time = now
        return self.fps

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None


class thread_auto_start_and_stop(Thread):
    def __init__(self,target=None,daemon=None):
        super(thread_auto_start_and_stop,self).__init__()
        self.flag=True
        # self.event.set()
    def run(self):
        while self.flag:
            time.sleep(1)
        print("Exit!")
    def stop(self):
        self.flag=False
        
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.face_rec=Face_Recognizer()
        self.register=self.pushButton
        self.login=self.pushButton_2
        self.login_camera_button=self.pushButton_3
        self.frameToAnalyze=deque(maxlen=1)
       
        #########实时处理###########
        self.is_get=0
        self.login_timer_check=QTimer(self)
        self.login_timer_check.timeout.connect(self.update_camera)
        self.login_if=False
        self.login_detect_res=self.label_5
        self.user_name=self.textEdit
        self.menu=Menu()
        self.reg=Register()
        self.password=self.textEdit_2
        self.login_camera=self.label
        self.info_path=r"val_recognize/user_info"
        self.flag=0
        self.camera_stream = CameraStream()
        self.name=''
        # 启动处理视频帧独立线程

        Thread(target=self.frameAnalyzeThreadFunc,daemon=True).start()
        self.register.clicked.connect(self.open_register)
        self.login_camera_button.clicked.connect(self.login_button_open_camera_click)
        self.login.clicked.connect(self.check_is_login)
        self.reg.account_Register.clicked.connect(self.close_register)
        self.create_folder()
    def create_folder(self):
        if not os.path.exists(self.info_path):
            os.makedirs(self.info_path)

    def close_register(self):
        self.reg.close()
        self.reg.timer_reg.stop()
        self.Open()

    def get_frame(self):
        return self.camera_stream.read()
        

    def change_status(self):
        self.flag=1-self.flag


    def frameAnalyzeThreadFunc(self):
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue
            else:
                login_frame = self.frameToAnalyze.popleft()
                img,self.is_get,results=self.face_rec.process(login_frame)
                # print(results)
                if self.is_get==1:
                    # return results
                    # img=results 
                    qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                        QtGui.QImage.Format_RGB888)  # 变成QImage形式
                    self.login_detect_res.setPixmap(QtGui.QPixmap.fromImage(qImage))  # 往显示Label里 显示QImage
                    time.sleep(0.1)
                    self.name=results[0]

    def login_button_open_camera_click(self):
        self.change_status()
        if self.flag==1:
            self.login_camera_button.setText("开始识别")
            self.login_timer_check.start(20)
        if self.flag==0:
            # self.change_status()
            self.login_timer_check.stop()
            self.login_camera_button.setText("关闭识别")
            self.login_camera.clear()
            self.camera_stream.release()
            self.login_camera.setText("")


    def update_camera(self):
        ret, login_frame=self.get_frame()
        if ret:
            result=login_frame
            fps = self.camera_stream.update_fps()
            cv2.putText(result, f'FPS:{int(fps)}', (270, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255, 255), 2)
            result = QtGui.QImage(result.data, result.shape[1], result.shape[0], QtGui.QImage.Format_RGB888)
            self.login_camera.setPixmap(QtGui.QPixmap.fromImage(result))
            if not self.frameToAnalyze:
                self.frameToAnalyze.append(login_frame)
            if self.name and self.name!='unknown':
                time.sleep(1)
                self.login_fun()

    def check_is_login(self):
        file_list=os.listdir(self.info_path)
        file_list=[i.split('.')[0] for i in file_list]
        #########没检测出人脸或者无法识别##########
        usr_name,pwd=self.user_name.toPlainText(),self.password.toPlainText()
        if usr_name and pwd:
            usrtxt=usr_name+'.txt'
            usr_info_path=os.path.join(self.info_path,usrtxt)
            with open(usr_info_path,'r') as f_val:
                content=f_val.read()
                content=content.split('\n')[1]
                content=content.split(':')[-1]
                if content==pwd:
                    self.login_if=False
                    self.login_fun()
                else:
                    print('login failed')
                    self.user_name.clear()
                    self.password.clear()
            return
        else:
            self.login_if=False
            print('please input full information!!')
            return
        
    def login_fun(self):
        print('login success')
        self.login_timer_check.stop()
        self.user_name.clear()
        self.password.clear()
        self.camera_stream.release()
        self.menu.Open()
        self.close()
            
    def open_register(self):
        self.reg.Open()
        self.close()

    def Open(self):
        self.show()
        
# 注册界面
class Register(QMainWindow, Ui_register_3):
    def __init__(self, parent=None):
        super(Register, self).__init__(parent)
        self.setupUi(self)
        self.face_rec=Face_Recognizer()
        self.account_Register=self.pushButton
        self.users_name=self.textEdit
        self.passwords=self.textEdit_2
        self.camera=self.label
        self.detect_res=self.label_5
        self.camera_button=self.pushButton_3
        self.flag=0
        self.camera_button.clicked.connect(self.button_open_camera_click)
        self.account_Register.clicked.connect(self.load_info)
        self.pic=np.ndarray
        self.is_detect=False
        self.camera_stream = CameraStream()
        self.timer_reg = QTimer(self)
        self.register_frame=deque(maxlen=1)
        self.timer_reg.timeout.connect(self.recognize_face)
        Thread(target=self.frameAnalyzeThreadFunc,daemon=True).start()
        self.dir_name=''
        self.info_path=r"val_recognize/user_info"
        self.img_path=r"val_recognize/face_dlib/data/data_faces_from_camera"

    def frameAnalyzeThreadFunc(self):
        while True:
            if not self.register_frame:
                time.sleep(0.01)
                continue
            else:
                login_frame = self.register_frame.popleft()
                faces,result=self.face_rec.recoginition(login_frame)
                print('faces',len(faces))
                if len(faces)==1:
                    self.pic=cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    result = QtGui.QImage(result.data, result.shape[1], result.shape[0], QtGui.QImage.Format_RGB888)
                    self.detect_res.setPixmap(QtGui.QPixmap.fromImage(result))
                    print('检测到了人脸')
                    self.is_detect=True
                    break
                else:
                    self.is_detect=False
                    continue
                
    def get_frame(self):
        return self.camera_stream.read()
       

    def recognize_face(self):
        ret,register_frame=self.get_frame()
        if ret:
            fps = self.camera_stream.update_fps()
            result=register_frame
            cv2.putText(result, f'FPS:{int(fps)}', (270, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255, 255), 2)
            result = QtGui.QImage(result.data, result.shape[1], result.shape[0], QtGui.QImage.Format_RGB888)
            self.camera.setPixmap(QtGui.QPixmap.fromImage(result))
            if not self.register_frame:
                self.register_frame.append(register_frame)



    def change_status(self):
        self.flag=1-self.flag

    def button_open_camera_click(self):
        self.change_status()
        if self.flag==1:
            self.camera_button.setText("开始检测")
            self.timer_reg.start(20)
        if self.flag==0:
            # self.change_status()
            self.timer_reg.stop()
            self.camera_button.setText("关闭检测")
            self.camera.clear()
            self.camera_stream.release()
            self.camera.setText("")
    
    
    def get_len(self,path):
        file=os.listdir(path)
        return len(file)
    
    def save_picture(self,path):
        ids=self.get_len(path)
        file_name=f'img_face_{ids+1}.jpg'
        path=os.path.join(path,file_name)
        cv2.imwrite(path,self.pic)

    def load_info(self):
        name,passwords=self.users_name.toPlainText(),self.passwords.toPlainText()
        nametxt=name+'.txt'
        if name and passwords:
            path=os.path.join(self.info_path,nametxt)
            with open(path,'w') as f:
                f.write(f'usr_name:{name}\n')
                f.write(f'usr_pwd:{passwords}')
                self.dir_name=f'picture_{name}'
                dir_path=os.path.join(self.img_path,f'picture_{name}')
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                # print(self.is_detect)
                if self.is_detect:
                    self.save_picture(dir_path)
                    main_tranfer()
            self.users_name.clear()
            self.passwords.clear()
            # self.close()
        else:
            print('please take full information!!')
            self.users_name.clear()
            self.passwords.clear()

    def Open(self):
        self.show()

# 菜单界面
class Menu(QMainWindow, Ui_register_2):
    def __init__(self, parent=None):
        super(Menu, self).__init__(parent)
        self.setupUi(self)
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512))])
        self.camera_stream = CameraStream()
        self.time_start_flag=0
        self.battery = self.lcdNumber_7
        self.speed_x = self.lcdNumber_9
        self.speed_y = self.lcdNumber_8
        self.speed_z = self.lcdNumber_6
        #init data
        self.volt=0
        self.sx=0
        self.sy=0
        self.sz=0
        self.manual_inputs = {
            "hover": (0.0, 0.0, 0.5, 0.0),
            "yaw_left": (0.0, 0.0, 0.5, -1.0),
            "yaw_right": (0.0, 0.0, 0.5, 1.0),
        }
        #打开定时器，对于每次定时器读取命令做出对应的抉择。
        self.keyboard_control = self.pushButton_12
        self.hands_control = self.pushButton_11
        self.voice_control = self.pushButton_8
        self.keyboard_signal = self.textBrowser
        self.voice_signal = self.textBrowser_2
        self.hands_signal = self.textBrowser_4
        self.human_identify = self.pushButton_13
        self.license_identify = self.pushButton_15
        self.reconstruction = self.pushButton_14
        self.takeoff_button=self.pushButton_9
        self.exit_button=self.pushButton_10
        self.ground_view = self.label
        self.air_view = self.label_4
        self.textwindow=self.textBrowser_3
        self.is_connect=self.checkBox_2
        self.stop_event=Event()
        self.drone=System()
        self.command_queue=deque()
        self.velocity=2.0
        # self.thread_command=Thread(target=self.display_command,daemon=True)
        self.object_model=build_yolo_people_detection()
        self.is_connect.stateChanged.connect(self.init_sitl)
        self.license_identify.clicked.connect(self.object_detection)
        # self.hands_control.clicked.connect()  #手势控制
        self.hands_control.clicked.connect(self.label_to_start_timer)  #手势控制
        self.takeoff_button.clicked.connect(self.takeoff_offer)
        self.timer_tsk=QTimer()
        self.process_it=None
        self.timer_tsk.timeout.connect(self.update_camera)
        self.timer_update_info=QTimer()
        self.timer_update_info.timeout.connect(self.update_flight_info)
        self.frame_to_tsk=deque(maxlen=1)
        self.t=thread_auto_start_and_stop()
        Thread(target=self.object_detection,daemon=True).start()
        self.command_handlers = {
            "向右": lambda: self._handle_velocity(0.0, -self.velocity, 0.0, (0.0, self.velocity, 0.0)),
            "向左": lambda: self._handle_velocity(0.0, self.velocity, 0.0, (0.0, -self.velocity, 0.0)),
            "向前": lambda: self._handle_velocity(self.velocity, 0.0, 0.0, (self.velocity, 0.0, 0.0)),
            "向后": lambda: self._handle_velocity(-self.velocity, 0.0, 0.0, (-self.velocity, 0.0, 0.0)),
            "上升": lambda: self._handle_velocity(0.0, 0.0, -self.velocity, (0.0, 0.0, -self.velocity)),
            "下降": lambda: self._handle_velocity(0.0, 0.0, self.velocity, (0.0, 0.0, self.velocity)),
            "向左转向": lambda: self._set_manual_control(self.manual_inputs["yaw_left"], (0.0, 0.0, 0.0)),
            "向右转向": lambda: self._set_manual_control(self.manual_inputs["yaw_right"], (0.0, 0.0, 0.0)),
            "悬停": lambda: self._set_manual_control(self.manual_inputs["hover"], (0.0, 0.0, 0.0)),
            "起飞": self.takeoff,
            "降落": self.land,
        }
    def loop_in_thread(self,_new_loop):  # 一个将被丢进线程的函数
        asyncio.set_event_loop(_new_loop)  # 调用loop需要使用set_event_loop方法指定loop
        _new_loop.run_forever()  # run_forever() 会永远阻塞当前线程，直到有人停止了该loop为止。

    def label_to_start_timer(self):
        new_loop = asyncio.new_event_loop()
        self.process_it = Thread(target=self.loop_in_thread, args=(new_loop,))  # 创建线程
        if not self.timer_update_info.isActive():
            self.timer_update_info.start(100)
        if not self.time_start_flag:
            self.process_it.start()
            even = asyncio.run_coroutine_threadsafe(self.display_command(), new_loop)
            self.hand_control()
        else:
            self.stop_event.set()
            print("have been stopped!")
        self.time_start_flag=1-self.time_start_flag
    def update_camera(self,tsk):
        """
        tsk=[obj_detection,sfm]
        检测（包括行人和车辆）
        """
        ret,frame=self.get_frame()
    def get_frame(self):
        return self.camera_stream.read()
    def change_status(self):
        self.flag=1-self.flag
    def update_flight_info(self):
        self.battery.display(int(self.volt))
        self.speed_x.display(int(self.sx))
        self.speed_y.display(int(self.sy))
        self.speed_z.display(int(self.sz))
    def takeoff_offer(self):
        def offer_command_takeoff():
            asyncio.run(self.takeoff())
        def offer_command_land():
            asyncio.run(self.land())
        if self.takeoff_button.text()=='起飞':
            # asyncio.run(self.takeoff())
            if self.t or self.t.is_alive():
                self.t.stop()
            self.t=thread_auto_start_and_stop(target=offer_command_takeoff)
            self.t.start()
            self.takeoff_button.setText('降落')
        if self.takeoff_button.text()=='降落':
            if self.t or self.t.is_alive():
                self.t.stop()
            self.t=thread_auto_start_and_stop(target=offer_command_land)
            self.t.start()
            self.takeoff_button.setText('起飞')
    async def display_command(self):
        await self.get_command_and_play()
        
    async def get_command_and_play(self):
        print("wait for connection…………")
        await self.drone.connect(system_address="udp://:14540")
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("----connected!")
                break

        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                break
        await self.drone.action.arm()
        print("-- Setting initial setpoint")
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))

        print("-- Starting offboard")
        try:
            await self.drone.offboard.start()
        except OffboardError as error:
            print(f"Starting offboard mode failed with error code: \
                {error._result.result}")
            print("-- Disarming")
            await self.drone.action.disarm()
            return
        async for battery in self.drone.telemetry.battery():
            print(f"电池状态: {battery}")
            self.volt=battery.voltage_v
            break
        while True:
            if self.command_queue:
                command=self.command_queue.popleft()
                print(command)
                handler=self.command_handlers.get(command, self._hover)
                result = handler()
                if asyncio.iscoroutine(result):
                    await result
                else:
                    await asyncio.sleep(0.1)
            else:
                await self._hover()

    async def _handle_velocity(self, north: float, east: float, down: float, display_speeds):
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(north, east, down, 0.0))
        self.sx, self.sy, self.sz = display_speeds
        await asyncio.sleep(0.1)

    async def _set_manual_control(self, inputs, display_speeds):
        roll, pitch, throttle, yaw = inputs
        await self.drone.manual_control.set_manual_control_input(
            float(roll), float(pitch), float(throttle), float(yaw)
        )
        self.sx, self.sy, self.sz = display_speeds
        await asyncio.sleep(0.1)

    async def _hover(self):
        await self._set_manual_control(self.manual_inputs["hover"], (0.0, 0.0, 0.0))

        
    def button_open_camera_click(self):
        self.change_status()
        if self.flag==1:
            self.license_identify.setText("开始行人检测")
            self.timer_tsk.start(20)
        if self.flag==0:
            # self.change_status()
            self.timer_tsk.stop()
            self.license_identify.setText("关闭行人检测")
            self.air_view.clear()
            self.camera_stream.release()
            self.air_view.setText("无人机端")
    def hand_control(self):
        if self.hands_control.text()=="开启手势控制":
            self.hands_control.setText("关闭手势控制")
            self.textwindow.setText("手势控制模式已开启")
            #电脑摄像头初始化
            wCam, hCam = 640, 480
            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
            cap.set(4, hCam)
            pTime = 0
            # 屏幕遥控器初始化：
            LeftPx, LeftPy = int(3 * wCam / 4), int(0.66 * hCam)
            RighPx, RighPy = int(wCam / 4), int(0.66 * hCam)
            PointRange = 60
            # 起降按钮大小：
            tlPw, tlPh = 64, 48
            # 起飞按钮(以左上角为初始点）
            tfP1x, tfP1y = 600, 30
            tfP2x, tfP2y = tfP1x - tlPw, tfP1y + tlPh
            takeoffcx, takeoffcy = 0.5 * (tfP1x + tfP2x), 0.5 * (tfP1y + tfP2y)
            # 降落按钮(以左上角为初始点）
            lP1x, lP1y = 104, 30
            lP2x, lP2y = lP1x - tlPw, lP1y + tlPh
            launchcx, launchcy = 0.5 * (lP1x + lP2x), 0.5 * (lP1y + lP2y)
            # 标志位
            takeoff = 1
            #print('aaaaaaaaaaaaaaaaaaaaa',takeoff)
            flag = 0
            # 速度命令初始化
            lr, fb, ud, yv = 40, 40, 25, 30
            lmList = []
            detector = htm.handDetector(detectionCon=0.7)  # 定义一个手部跟踪对象,置信度0.7
            while self.hands_control.text()=="关闭手势控制":
                success, img = cap.read()
                img = cv2.flip(img, 1)  # 水平镜像
                img = detector.findHands(img)  # 调用方法；将拍摄图像转换成带有手部标记的图像
                #telloimg = self.me.get_frame_read().frame  # 调用tello摄像头
                #telloimg = cv2.resize(telloimg, (360, 240))  # 调整图像大小
                lmList = detector.findPosition(img, draw=False)
                # 显示手势控制按钮

                # 起飞按钮
                if takeoff == 1:
                    cv2.rectangle(img, (tfP1x + 4, tfP1y - 3), (tfP2x - 4, tfP2y + 3), (0, 255, 0), cv2.FILLED)
                    cv2.rectangle(img, (lP1x, lP1y), (lP2x, lP2y), (128, 128, 128), cv2.FILLED)
                    cv2.putText(img, f'takeoff', (500, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(img, f'land', (25, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 1)
                    # 左右遥杆：
                    cv2.circle(img, (LeftPx, LeftPy), 10, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (LeftPx, LeftPy), PointRange, (0, 255, 0), 3)
                    cv2.circle(img, (RighPx, RighPy), 10, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (RighPx, RighPy), PointRange, (0, 255, 0), 3)
                else:
                    cv2.rectangle(img, (tfP1x, tfP1y), (tfP2x, tfP2y), (128, 128, 128), cv2.FILLED)
                    cv2.rectangle(img, (lP1x + 4, lP1y - 3), (lP2x - 4, lP2y + 3), (255, 0, 0), cv2.FILLED)
                    cv2.putText(img, f'takeoff', (500, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 1)
                    cv2.putText(img, f'land', (25, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

                    # 左右遥杆：
                    cv2.circle(img, (LeftPx, LeftPy), 10, (128, 128, 128), cv2.FILLED)
                    cv2.circle(img, (LeftPx, LeftPy), PointRange, (128, 128, 128), 3)
                    cv2.circle(img, (RighPx, RighPy), 10, (128, 128, 128), cv2.FILLED)
                    cv2.circle(img, (RighPx, RighPy), PointRange, (128, 128, 128), 3)

                if len(lmList) != 0:
                    cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (0, 0, 255), cv2.FILLED)  # 显示手指
                    # print(lmList[8][1], lmList[8][2])
                    val = [0, 0, 0, 0]
                    # 点击起飞降落按钮
                    if (takeoffcy - lmList[8][2]) ** 2 + (takeoffcx - lmList[8][1]) ** 2 < 13 ** 2:
                        takeoff = 1
                        if flag == 0:
                            self.command_queue.append("起飞")
                            self.hands_signal.setText("起飞")
                            flag = 1
                    if (launchcy - lmList[8][2]) ** 2 + (launchcx - lmList[8][1]) ** 2 < 13 ** 2:
                        takeoff = 0
                        self.command_queue.append("降落")
                        self.hands_signal.setText("降落")
                        flag = 0
                        # print("降落")

                    # 控制遥杆
                    if (LeftPy - lmList[8][2]) ** 2 + (LeftPx - lmList[8][1]) ** 2 < PointRange ** 2:
                        cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (255, 255, 0), cv2.FILLED)  # 指变色
                        cv2.circle(img, (LeftPx, LeftPy), 10, (255, 255, 0), cv2.FILLED)
                        cv2.circle(img, (LeftPx, LeftPy), PointRange, (255, 255, 0), 3)
                        cv2.line(img, (LeftPx, LeftPy), (lmList[8][1], lmList[8][2]), (255, 255, 0), 3)  # 杆指间画线
                        direction = abs((lmList[8][2] - LeftPy + 0.001) / (lmList[8][1] - LeftPx + 0.001))  # 计算指杆斜率绝对值

                        if lmList[8][2] < LeftPy:  # 手指在上方
                            if lmList[8][1] < LeftPx:  # 手指在第一象限
                                if direction < 1:
                                    # print("向左")
                                    self.command_queue.append("向左")
                                    self.hands_signal.setText("向左")
                                    val[0] = -lr
                                    # self.me.send_rc_control(-lr, 0, 0, 0)
                                else:
                                    # print("向前")
                                    self.command_queue.append("向前")
                                    self.hands_signal.setText("向前")
                                    val[1] = fb
                                    # self.me.send_rc_control(0, fb, 0, 0)
                            else:  # 手指在第四象限
                                if direction < 1:
                                    # print("向右")
                                    self.command_queue.append("向右")
                                    self.hands_signal.setText("向右")
                                    val[0] = lr
                                    # self.me.send_rc_control(lr, 0, 0, 0)
                                else:
                                    # print("向前")
                                    self.command_queue.append("向前")
                                    self.hands_signal.setText("向前")
                                    val[1] = fb
                                    # self.me.send_rc_control(0, fb, 0, 0)
                        else:  # 手指在下方
                            if lmList[8][1] < LeftPx:  # 手指在第二象限
                                if direction < 1:
                                    # print("向左")
                                    self.command_queue.append("向左")
                                    self.hands_signal.setText("向左")
                                    val[0] = -lr
                                    # self.me.send_rc_control(-lr, 0, 0, 0)
                                else:
                                    # print("向后")
                                    self.command_queue.append("向后")
                                    self.hands_signal.setText("向后")
                                    val[1] = -fb
                                    # self.me.send_rc_control(0, -fb, 0, 0)
                            else:  # 手指在第四象限
                                if direction < 1:
                                    # print("向右")
                                    self.command_queue.append("向右")
                                    self.hands_signal.setText("向右")
                                    val[0] = lr
                                    # self.me.send_rc_control(lr, 0, 0, 0)
                                else:
                                    # print("向后")
                                    self.command_queue.append("向后")
                                    self.hands_signal.setText("向后")
                                    val[1] = -fb
                                    # self.me.send_rc_control(0, -fb, 0, 0)
                    elif (RighPy - lmList[8][2]) ** 2 + (RighPx - lmList[8][1]) ** 2 < PointRange ** 2:
                        cv2.circle(img, (lmList[8][1], lmList[8][2]), 10, (255, 255, 0), cv2.FILLED)  # 指变色
                        cv2.circle(img, (RighPx, RighPy), 10, (255, 255, 0), cv2.FILLED)
                        cv2.circle(img, (RighPx, RighPy), PointRange, (255, 255, 0), 3)
                        cv2.line(img, (RighPx, RighPy), (lmList[8][1], lmList[8][2]), (255, 255, 0), 3)  # 杆指间画线
                        # 操作右遥杆发出命令：
                        direction = abs((lmList[8][2] - RighPy + 0.001) / (lmList[8][1] - RighPx + 0.001))  # 计算指杆斜率绝对值
                        if lmList[8][2] < RighPy:  # 手指在上方
                            if lmList[8][1] < RighPx:  # 手指在第一象限
                                if direction < 1:
                                    # print("向左转向")
                                    self.command_queue.append("向左转向")
                                    self.hands_signal.setText("向左转向")
                                    val[3] = yv
                                    # self.me.send_rc_control(0, 0, 0, yv)
                                else:
                                    # print("上升")
                                    self.command_queue.append("上升")
                                    self.hands_signal.setText("上升")
                                    val[2] = ud
                                    # self.me.send_rc_control(0, 0, ud, 0)
                            else:  # 手指在第四象限
                                if direction < 1:
                                    # print("向右转向")
                                    self.command_queue.append("向右转向")
                                    self.hands_signal.setText("向右转向")
                                    val[3] = -yv
                                    # self.me.send_rc_control(0, 0, 0, -yv)
                                else:
                                    # print("上升")
                                    self.command_queue.append("上升")
                                    self.hands_signal.setText("上升")
                                    val[2] = ud
                                    # self.me.send_rc_control(0, 0, ud, 0)
                        else:  # 手指在下方
                            if lmList[8][1] < RighPx:  # 手指在第二象限
                                if direction < 1:
                                    # print("向左转向")
                                    self.command_queue.append("向左转向")
                                    self.hands_signal.setText("向左转向")
                                    val[3] = yv
                                    # self.me.send_rc_control(0, 0, 0, yv)
                                else:
                                    # print("下降")
                                    self.command_queue.append("下降")
                                    self.hands_signal.setText("下降")
                                    val[2] = -ud
                                    # self.me.send_rc_control(0, 0, -ud, 0)
                            else:  # 手指在第四象限
                                if direction < 1:
                                    # print("向右转向")
                                    self.command_queue.append("向右转向")
                                    self.hands_signal.setText("向右转向")
                                    val[3] = -yv
                                    # self.me.send_rc_control(0, 0, 0, -yv)
                                else:
                                    # print("下降")
                                    self.command_queue.append("下降")
                                    self.hands_signal.setText("下降")
                                    val[2] = -ud
                                    # self.me.send_rc_control(0, 0, -ud, 0)
                    else:
                        # print("悬停")
                        self.command_queue.append("悬停")
                        self.hands_signal.setText("悬停")
                
                # print(1)
                # 帧率显示
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime

                # 在img上实时显示帧率：坐标：（10,70），字体，比例，颜色，粗细
                cv2.putText(img, f'FPS:{int(fps)}', (270, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                #show1 = cv2.resize(self.image, (400, 300))
                show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                self.ground_view.setPixmap(QtGui.QPixmap.fromImage(showImage))

                #cv2.imshow("手势控制模式", img)
                ##cv2.imshow("Tello", telloimg)

                cv2.waitKey(1)
        if self.hands_control.text()=="关闭手势控制":
            self.hands_control.setText("开启手势控制")
            self.textwindow.setText("手势控制模式已关闭")
            self.ground_view.clear()
            self.command_queue.clear()
            self.ground_view.setText("地面端")

    def object_detection(self):
        while True:
            if not self.frame_to_tsk:
                time.sleep(0.01)
                continue
            login_frame = self.frame_to_tsk.popleft()
            result_cls,result_conf,bboxs = self.object_model.predict(login_frame)
            if result_cls.numel()>0:
                for cls, conf, bbox in zip(result_cls.tolist(), result_conf.tolist(), bboxs.tolist()):
                    # TODO: integrate detection results with UI display.
                    pass
    def init_sitl(self):
        state=self.is_connect.isChecked()
        if state:
            self.timer_update_info.start(100)
            Thread(target=self.get_changed,daemon=True).start()
        else:
            self.timer_update_info.stop()
            print("The drone has been cancelled.")
    def get_changed(self,target):
        asyncio.run(self.set_thread_get_info()) 
    async def set_thread_get_info(self):
        await self.drone.connect(system_address="udp://:14540")
        print("Waiting for drone to connect...")
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to drone!")
                break
        async for gps_info in self.drone.telemetry.gps_info():
            print(f"GPS 信息: {gps_info}")
            break

        async for battery in self.drone.telemetry.battery():
            print(f"电池状态: {battery}")
            self.volt=battery.voltage_v
            break
    async def takeoff(self):
        await self.drone.connect(system_address="udp://:14540")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to drone!")
                break
                # 使飞行器解锁
        print("-- 解锁飞行器")
        await self.drone.action.arm()

        # 使飞行器起飞
        print("-- 起飞")
        await self.drone.action.takeoff()

        await asyncio.sleep(10)
    async def land(self):
        await self.drone.connect(system_address="udp://:14540")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to drone!")
                break
                # 使飞行器解锁
        print("-- 解锁飞行器")
        await self.drone.action.arm()

        # 使飞行器起飞
        print("-- 降落")
        await self.drone.action.land()
        await asyncio.sleep(10)
    def Open(self):
        self.show()                      
def close_all_windows():
    for window in QApplication.topLevelWidgets():
        window.close()
    sys.exit(QApplication.exec_())
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # main_app=app()
    main = MainWindow()
    # 实例化注册页面
    main.Open()
    main.menu.exit_button.clicked.connect(close_all_windows)
    sys.exit(app.exec_())
