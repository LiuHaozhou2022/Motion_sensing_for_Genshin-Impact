"""
future work: add explain for this project
"""
import json
import os
import time
import subprocess
import threading
import numpy as np
import pyautogui
import pydirectinput
import math
import cv2


from pynput.keyboard import Controller


base_dir = os.path.dirname(__file__)
keyboard = Controller()
threshold = 0.5
mouse_move_flag = 0.0

def camera_thread():
    """ 
    该函数为启动相机与姿态识别线程
    """
    infer_path = os.path.join(base_dir, "python", "det_keypoint_unite_infer.py")
    det_model_dir = os.path.join(base_dir, "models", "picodet_v2_s_320_pedestrian")
    kpt_model_dir = os.path.join(base_dir, "models", "tinypose_128x96")
    shell = "python %s --det_model_dir=%s --keypoint_model_dir=%s --camera_id=0 --device=GPU --save_res=True" % \
            (infer_path, det_model_dir, kpt_model_dir)
    subprocess.run(shell)
    
def mouse_move_thread():
    """ 
    视角丝滑移动线程
    """
    while(1):
        if mouse_move_flag<1.0 and mouse_move_flag>-1.0:
            continue
        elif mouse_move_flag<-1.0:
            pydirectinput.moveRel(xOffset=-50,yOffset=0,duration=1.4,relative=True)
        elif mouse_move_flag>1.0:
            pydirectinput.moveRel(xOffset=50,yOffset=0,duration=1.4,relative=True)
    
def processAndKeyboard():
    """ 
    处理AI检测的结果，根据动作实现按键控制
    """
    while True:
        camera_res = None
        
        while camera_res is None:
            try:
                camera_res = json.load(open("temp.json", "r"))
            except:
                pass
        camera_KPT = camera_res[2][0][0]
        camera_box = camera_res[1][0]
        camera_vec = np.array(camera_KPT)[0:17, 0:3]
        # print(camera_vec[0][0])
        check_view_turn(camera_vec[1],camera_vec[0],camera_vec[2])
        # check_turn(camera_vec[5],camera_vec[6],camera_vec[11],camera_vec[12])
        check_run(camera_vec[13],camera_vec[14],camera_vec[11],camera_vec[12])
        check_jump(camera_vec[11],camera_vec[12],camera_vec[9],camera_vec[10],camera_vec[0])
        check_q(camera_vec[5],camera_vec[6],camera_vec[7],camera_vec[8],camera_vec[9],camera_vec[10])
        check_acc(camera_vec[5],camera_vec[6],camera_vec[7],camera_vec[8],camera_vec[9],camera_vec[10])
        check_e(camera_vec[5],camera_vec[6],camera_vec[7],camera_vec[8],camera_vec[9],camera_vec[10])
        check_attack(camera_vec[5],camera_vec[6],camera_vec[7],camera_vec[8],camera_vec[9],camera_vec[10])
        
        
        # time.sleep(0.01)
    
def check_view_turn(leftShoulderP, noseP, rightShoulderP):
    """ 
    转身：通过左右眼睛与鼻子的相对距离检测
    """
    if leftShoulderP[2]<threshold or noseP[2]<threshold or rightShoulderP[2]<threshold: 
        return  #判断置信度
    leftDis =  abs(leftShoulderP[0] -  noseP[0])
    rightDis = abs(rightShoulderP[0] - noseP[0])
    leftDisNorm = leftDis/(leftDis+rightDis)
    rightDisNorm = rightDis/(leftDis+rightDis)
    threshold_view_turn = 0.2
    error_norm = leftDisNorm-rightDisNorm
    global mouse_move_flag
    if leftDisNorm-rightDisNorm<-threshold_view_turn: 
        mouse_move_flag = -2.0
        # pyautogui.dragRel(-20, 0)
    elif leftDisNorm-rightDisNorm> threshold_view_turn:
        mouse_move_flag = 2.0
        # pyautogui.dragRel(20, 0)
    else:
        mouse_move_flag = 0.0
    
def check_turn(leftShoulderP, rightShoulderP, leftHipP, rightHipP):
    """ 
    转身：通过左右肩膀与左右髋关节相对距离检测
    """
    if leftShoulderP[2]<0.5 or leftHipP[2]<0.5 or rightShoulderP[2]<0.5 or rightHipP[2]<0.5: 
        return  #判断置信度
    shoulder_center_x = (leftShoulderP[0] + rightShoulderP[0]) / 2
    hip_lenth = abs(leftHipP[0] - rightHipP[0])
    shoulder_center_proj = abs(leftHipP[0] - shoulder_center_x) / hip_lenth
    threshold_turn = 0.15
    if shoulder_center_proj < threshold_turn:
        pydirectinput.moveRel(xOffset=-200,yOffset=0,duration=5.4,relative=True)
    if shoulder_center_proj > (1-threshold_turn):
        pydirectinput.moveRel(xOffset=200,yOffset=0,duration=5.4,relative=True)
        
def check_run(leftKnee, rightKnee, leftHipP, rightHipP):
    """ 
    跑步：通过左右膝盖与左右髋关节相对距离检测
    """
    if leftKnee[2]<threshold or leftHipP[2]<threshold or rightKnee[2]<threshold or rightHipP[2]<threshold: 
        return  #判断置信度
    knee_center = (leftKnee[1] + rightKnee[1])/2
    hip_center = (leftHipP[1] + rightHipP[1])/2
    left_norm_dis = abs(hip_center - leftKnee[1])
    right_norm_dis = abs(hip_center - rightKnee[1])
    relative_position = left_norm_dis / right_norm_dis
    threshold_run = 0.1
    if relative_position > (1+threshold_run):
        # keyboard.type("w")
        pydirectinput.keyDown("w")
        # Controller.press("w")
    elif relative_position < (1-threshold_run):
        pydirectinput.keyDown("w")
    else:
        pydirectinput.keyUp("w")
        
def check_jump(leftHipP, rightHipP, leftWrist, rightWrist, nose):
    """ 
    跳跃：通过判定双手与鼻子的竖直距离
    """        
    if leftHipP[2]<threshold or rightHipP[2]<threshold or leftWrist[2]<threshold or rightWrist[2]<threshold or nose[2]<threshold: 
        return  #判断置信度
    norm_dis = abs(nose[1]-(leftHipP[1]+rightHipP[1])/2)
    left_norm = abs(nose[1]-leftWrist[1])
    right_norm = abs(nose[1]-rightWrist[1])
    threshold_run = 0.1
    if leftWrist[1]<nose[1] and rightWrist[1]<nose[1]:
        pydirectinput.keyDown(" ")
    else:
        pydirectinput.keyUp(" ")
        
def check_acc(leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist):
    """ 
    加速：通过判定双手平举
    """     
    if leftShoulder[2]<threshold or rightShoulder[2]<threshold or leftElbow[2]<threshold or  \
    rightElbow[2]<threshold or leftWrist[2]<threshold or rightWrist[2]<threshold :
        return
    left_big_arm_norm =  math.sqrt((leftShoulder[0]-leftElbow[0])**2 + (leftShoulder[1]-leftElbow[1])**2)
    left_small_arm_norm= math.sqrt((leftWrist[0]-leftElbow[0])**2 + (leftWrist[1]-leftElbow[1])**2)
    right_big_arm_norm=  math.sqrt((rightShoulder[0]-rightElbow[0])**2 + (rightShoulder[1]-rightElbow[1])**2)
    right_small_arm_norm=math.sqrt((leftWrist[0]-leftElbow[0])**2 + (leftWrist[1]-leftElbow[1])**2)
    left_arm_norm = left_big_arm_norm + left_small_arm_norm
    right_arm_norm= right_big_arm_norm+ right_small_arm_norm
    left_arm_x_dis = abs(leftShoulder[0]-leftWrist[0])
    right_arm_x_dis= abs(rightShoulder[0]-rightWrist[0])
    acc_threshold = 0.9
    if left_arm_x_dis/left_arm_norm >acc_threshold and right_arm_x_dis/right_arm_norm >acc_threshold:
        pydirectinput.mouseDown(button='right')
        # keyboard.type("q")
    else:
        pydirectinput.mouseUp(button='right')
    
def check_e(leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist):
    """ 
    左键：左手放左肩膀
    """  
    if leftShoulder[2]<threshold or rightShoulder[2]<threshold or leftElbow[2]<threshold or  \
    rightElbow[2]<threshold or leftWrist[2]<threshold or rightWrist[2]<threshold :
        return
    norm_length = abs(leftShoulder[0] - rightShoulder[0])
    left_elbow2shoulder = math.sqrt((leftElbow[0]-leftShoulder[0])**2 + (leftElbow[1]-leftShoulder[1])**2)
    right_elbow2shoulder= math.sqrt((rightElbow[0]-rightShoulder[0])**2 + (rightElbow[1]-rightShoulder[1])**2)
    left_wrist2shoulder = math.sqrt((leftWrist[0]-leftShoulder[0])**2 + (leftWrist[1]-leftShoulder[1])**2)
    right_wrist2shoulder = math.sqrt((rightWrist[0]-rightShoulder[0])**2 + (rightWrist[1]-rightShoulder[1])**2)
    e_threshold = 0.25
    left_elbow2shoulder_norm = left_elbow2shoulder/norm_length
    right_elbow2shoulder_norm= right_elbow2shoulder/norm_length
    left_wrist2shoulder_norm = left_wrist2shoulder/norm_length
    right_wrist2shoulder_norm= right_wrist2shoulder/norm_length
    if left_wrist2shoulder_norm<e_threshold and rightWrist[1]>leftElbow[1]:
        pydirectinput.keyDown("e")
    else:
        pydirectinput.keyUp("e")
    
def check_attack(leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist):
    """ 
    左键：左手放左肩膀
    """  
    if leftShoulder[2]<threshold or rightShoulder[2]<threshold or leftElbow[2]<threshold or  \
    rightElbow[2]<threshold or leftWrist[2]<threshold or rightWrist[2]<threshold :
        return
    norm_length = abs(leftShoulder[0] - rightShoulder[0])
    left_elbow2shoulder = math.sqrt((leftElbow[0]-leftShoulder[0])**2 + (leftElbow[1]-leftShoulder[1])**2)
    right_elbow2shoulder= math.sqrt((rightElbow[0]-rightShoulder[0])**2 + (rightElbow[1]-rightShoulder[1])**2)
    left_wrist2shoulder = math.sqrt((leftWrist[0]-leftShoulder[0])**2 + (leftWrist[1]-leftShoulder[1])**2)
    right_wrist2shoulder = math.sqrt((rightWrist[0]-rightShoulder[0])**2 + (rightWrist[1]-rightShoulder[1])**2)
    e_threshold = 0.25
    left_elbow2shoulder_norm = left_elbow2shoulder/norm_length
    right_elbow2shoulder_norm= right_elbow2shoulder/norm_length
    left_wrist2shoulder_norm = left_wrist2shoulder/norm_length
    right_wrist2shoulder_norm= right_wrist2shoulder/norm_length
    if right_wrist2shoulder_norm<e_threshold and leftWrist[1]>rightElbow[1]:
        pydirectinput.click(button='left')
    
def check_q(leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist):
    """ 
    q技能：右手--|
    """  
    if leftShoulder[2]<threshold or rightShoulder[2]<threshold or leftElbow[2]<threshold or  \
    rightElbow[2]<threshold or leftWrist[2]<threshold or rightWrist[2]<threshold :
        return
    # norm_length = abs(leftShoulder[0] - rightShoulder[0])
    # left_elbow2right_wrist_x = abs(leftElbow[0]-rightWrist[0])
    # right_elbow2left_wrist_x=  abs(rightElbow[0]-rightWrist[0])
    # e_threshold = 0.3
    # shoulder_dis = abs(leftShoulder[0]-rightShoulder[0])
    # left_elbow2right_wrist_norm = left_elbow2right_wrist_x/shoulder_dis
    # right_elbow2left_wrist_norm = right_elbow2left_wrist_x/shoulder_dis
    # if left_elbow2right_wrist_norm<e_threshold and right_elbow2left_wrist_norm<e_threshold:
    right_elbow_dis_x=abs(rightElbow[0]-rightShoulder[0])
    right_elbow_dis  =math.sqrt((rightElbow[0]-rightShoulder[0])**2 + (rightElbow[1]-rightShoulder[1])**2)
    right_wrist_dis_x=abs(rightWrist[1]-rightElbow[1])
    right_wrist_dis  =math.sqrt((rightWrist[1]-rightElbow[1])**2 + (rightWrist[0]-rightElbow[0])**2)
    if right_elbow_dis_x/right_elbow_dis > 0.8 and right_wrist_dis_x/right_wrist_dis >0.8 \
    and leftWrist[1]>leftElbow[1] and leftElbow[1]>leftShoulder[1]:
        pydirectinput.keyDown("q")
    else:
        pydirectinput.keyUp("q")
    
    
    
    
    
    
if __name__ == "__main__":
    
    t = threading.Thread(target=camera_thread, args=())
    # t.setDaemon(True)
    t.start()
    if os.path.isfile("temp.json"):
        os.remove("temp.json")
    t2 = threading.Thread(target=mouse_move_thread, args=())
    t2.start()
    print("15秒以后开启姿态检测")
    cv2.waitKey(15000)
    processAndKeyboard()
    
    
   