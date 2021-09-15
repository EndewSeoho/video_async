from torch.autograd import Variable
from torchvision import transforms
from .preprocessing import faceAlignment
from PIL import Image
from .apps import *
# from moviepy.editor import *
# import librosa
import numpy as np
import mediapipe as mp
import pandas as pd
import csv

# device 설정. CUDA 사용가능하면 CUDA 모드로, 못 쓰면 CPU 모드로 동작
# 단 cpu로 연산 할 경우 인식 함수 내 코드 수정 필요.
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cuda"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 얼굴 정보를 담을 class
class Face:
    def __init__(self):
        # class init
        self.rt = [-1, -1, -1, -1]
        self.sID = ""
        self.fvScore = -1.
        self.ptLE = [-1, -1]
        self.ptRE = [-1, -1]
        self.ptLM = [-1, -1]
        self.ptRM = [-1, -1]

        self.ptLED = [-1, -1]
        self.ptRED = [-1, -1]

        self.fYaw = -1.
        self.fPitch = -1.
        self.fRoll = -1.

        self.nEmotion = -1
        self.fEmotionScore = [-1, -1, -1, -1, -1, -1, -1]



def get_state_dict(origin_dict):
    old_keys = origin_dict.keys()
    new_dict = {}

    for ii in old_keys:
        temp_key = str(ii)
        if temp_key[0:7] == "module.":
            new_key = temp_key[7:]
        else:
            new_key = temp_key

        new_dict[new_key] = origin_dict[temp_key]
    return new_dict



# 초기화
def Initialization():
    # face detector. OpenCV SSD
    # FD_Net = cv2.dnn.readNetFromCaffe("C:/Users/withmind/Desktop/models/opencv_ssd.prototxt", "C:/Users/withmind/Desktop/models/opencv_ssd.caffemodel")
    FD_Net = ImConfig.FD_Net

    # Landmark 모델
    # Landmark_Net = LandmarkNet(3, 3)
    # # Landmark_Net = torch.nn.DataParallel(Landmark_Net).to(device)
    # Landmark_Net = Landmark_Net.to(device)
    # Landmark_Net.load_state_dict(torch.load("C:/Users/withmind/Desktop/models/ETRI_LANDMARK_68pt.pth.tar", map_location=device)['state_dict'])
    # Landmark_Net.load_state_dict(torch.load("/home/ubuntu/projects/withmind_video/im_video/file/ETRI_LANDMARK_68pt.pth.tar", map_location=device)['state_dict'])
    ImConfig.Landmark_Net


    # Headpose 모델
    # Headpose_Net = HeadposeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    # Headpose_Net = Headpose_Net.to(device)
    # Headpose_Net.load_state_dict(torch.load("C:/Users/withmind/Desktop/models/ETRI_HEAD_POSE.pth.tar"))
    # Headpose_Net.load_state_dict(torch.load("/home/ubuntu/projects/withmind_video/im_video/file/ETRI_HEAD_POSE.pth.tar"))
    ImConfig.Headpose_Net

    # Emotion classifier
    # Emotion_Net = EmotionNet(num_classes=7).to(device)
    # new_dict = get_state_dict(torch.load("C:/Users/withmind/Desktop/models/ETRI_Emotion.pth.tar")['state_dict'])
    # # new_dict = get_state_dict(torch.load("/home/ubuntu/projects/withmind_video/im_video/file/ETRI_EMOTION.pth.tar")['state_dict'])
    # Emotion_Net.load_state_dict(new_dict)
    ImConfig.Emotion_Net


    # 각 모델 evaluation 모드로 설정
    ImConfig.Landmark_Net.eval()
    ImConfig.Headpose_Net.eval()
    ImConfig.Emotion_Net.eval()

    return FD_Net, ImConfig.Landmark_Net, ImConfig.Headpose_Net, ImConfig.Emotion_Net


# 얼굴검출
# OpenCV 기본 예제 적용
def Face_Detection(FD_Net, cvImg, list_Face):
    del list_Face[:]
    img = cvImg.copy()
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    FD_Net.setInput(blob)
    detections = FD_Net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.95:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ETRIFace 클래스에 입력하여 리스트에 저장.
            # 정방 사이즈로 조절하여 저장 함.
            ef = Face()
            difX = endX - startX
            difY = endY - startY
            if difX > difY:
                offset = int((difX - difY) / 2)
                new_startY = max(startY - offset, 0)
                new_endY = min(endY + offset, h - 1)
                new_startX = max(startX, 0)
                new_endX = min(endX, w - 1)
                ef.rt = [new_startX, new_startY, new_endX, new_endY]
            else:
                offset = int((difY - difX) / 2)
                new_startX = max(startX - offset, 0)
                new_endX = min(endX + offset, w - 1)
                new_startY = max(startY, 0)
                new_endY = min(endY, h - 1)
                ef.rt = [new_startX, new_startY, new_endX, new_endY]

            list_Face.append(ef)

    # torch.cuda.empty_cache()

    return len(list_Face)

# landmark 검출
def Landmark_Detection(Landmark_Net, cvImg, list_Face, nIndex):
    h, w, _ = cvImg.shape
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yLeftBottom_in = list_Face[nIndex].rt[1]
    yRightTop_in = list_Face[nIndex].rt[3]
    xLeftBottom_in = list_Face[nIndex].rt[0]
    xRightTop_in = list_Face[nIndex].rt[2]

    n15 = (yRightTop_in - yLeftBottom_in) * 0.2
    xLeftBottom_in = max(xLeftBottom_in - n15, 0)
    xRightTop_in = min(xRightTop_in + n15, w-1)
    yLeftBottom_in = max(yLeftBottom_in - n15, 0)
    yRightTop_in = min(yRightTop_in + n15, h-1)

    INPUT = cvImg[(int(yLeftBottom_in)):(int(yRightTop_in)), (int(xLeftBottom_in)): (int(xRightTop_in))]
    # print("erreere", INPUT)
    # 인식 좌표 정보에 얼굴 위치 보정하기 위한 값
    # offsetX = list_ETRIFace[nIndex].rt[0]
    # offsetY = list_ETRIFace[nIndex].rt[1]
    offsetX = xLeftBottom_in
    offsetY = yLeftBottom_in

    # if len(INPUT) != 0:

    # preprocessing
    w = xRightTop_in - xLeftBottom_in
    INPUT = cv2.resize(INPUT, (256, 256))
    INPUT = INPUT / 255
    ratio = w / 256
    INPUT = np.transpose(INPUT, axes=[2, 0, 1])
    INPUT = np.array(INPUT, dtype=np.float32)
    INPUT = torch.from_numpy(INPUT)
    INPUT = torch.unsqueeze(INPUT, 0)
    INPUT = INPUT.to(device)
    OUTPUT = Landmark_Net(INPUT)
    OUTPUT = torch.squeeze(OUTPUT)
    output_np = OUTPUT.cpu().detach().numpy()
    output_np = output_np * 1.1 * 256
    output_np = output_np * ratio

    # 좌표 보정
    for ii in range(68):
        output_np[ii * 2 + 0] = output_np[ii * 2 + 0] + offsetX
        output_np[ii * 2 + 1] = output_np[ii * 2 + 1] + offsetY

    leX = leY = reX = reY = lmX = lmY = rmX = rmY = nX = nY = 0
    for ii in range(36, 42, 1):
        leX = leX + output_np[ii * 2 + 0]
        leY = leY + output_np[ii * 2 + 1]

    for ii in range(42, 48, 1):
        reX = reX + output_np[ii * 2 + 0]
        reY = reY + output_np[ii * 2 + 1]

    # 눈, 입 양 끝점 저장
    list_Face[nIndex].ptLE = [int(leX / 6), int(leY / 6)]
    list_Face[nIndex].ptRE = [int(reX / 6), int(reY / 6)]
    list_Face[nIndex].ptLM = [int(output_np[48 * 2 + 0]), int(output_np[48 * 2 + 1])]
    list_Face[nIndex].ptRM = [int(output_np[54 * 2 + 0]), int(output_np[54 * 2 + 1])]
    list_Face[nIndex].ptN = [int(output_np[30 * 2 + 0]), int(output_np[30 * 2 + 1])]

    torch.cuda.empty_cache()

    return output_np

    # else:
    #     return []


transformations_emotionnet = transforms.Compose(
    [transforms.Grayscale(),
     transforms.CenterCrop(128),
     transforms.ToTensor()]
)

transformations_headposenet = transforms.Compose(
    [transforms.Scale(224),
     transforms.CenterCrop(224), transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def HeadPose_Estimation(HeadPose_Net, cvImg, list_Face, nIndex):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oImg = cvImg[list_Face[nIndex].rt[1]:list_Face[nIndex].rt[3], \
          list_Face[nIndex].rt[0]:list_Face[nIndex].rt[2]].copy()

    # cv2.imshow("ZZ", oImg)
    # cv2.waitKey(0)

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    PILImg = Image.fromarray(oImg)

    img = transformations_headposenet(PILImg)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).to(device)

    yaw, pitch, roll = HeadPose_Net(img)

    yaw_predicted = F.softmax(yaw)
    pitch_predicted = F.softmax(pitch)
    roll_predicted = F.softmax(roll)

    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    list_Face[nIndex].fYaw = yaw_predicted
    list_Face[nIndex].fPitch = pitch_predicted
    list_Face[nIndex].fRoll = roll_predicted

    torch.cuda.empty_cache()

    return (yaw_predicted, pitch_predicted, roll_predicted)



def list2SoftList(srcList):
    tmpList = srcList.copy()

    fSum = 0.

    for ii in range(len(srcList)):
        fExp = np.exp(srcList[ii])
        fSum = fSum + fExp
        tmpList[ii] = fExp
    for ii in range(len(srcList)):
        srcList[ii] = tmpList[ii] / fSum;

    return srcList

def Emotion_Classification(Emotion_Net, cvImg, list_Face, nIndex):
    if list_Face[nIndex].ptLE == [-1, -1]:
        return -1

    # 정규화 얼굴영역 이미지 생성
    img = cvImg.copy()
    ROIImg = faceAlignment(img, list_Face[nIndex].ptLE, list_Face[nIndex].ptRE
                           , list_Face[nIndex].ptLM, list_Face[nIndex].ptLM)

    # image preprocessing
    PILROI = Image.fromarray(ROIImg)
    transformedImg = transformations_emotionnet(PILROI)
    transformedImg = torch.unsqueeze(transformedImg, 0)
    transformedImg = transformedImg.to(device)

    # emotion-net feedforward processing
    output_glasses = Emotion_Net(transformedImg)

    # 출력 결과 확률로 변환
    output_cpu = output_glasses.cpu().detach().numpy().squeeze()
    output = list2SoftList(output_cpu).tolist()

    # emotion label
    # surprise, fear, disgust, happy, sadness, angry, neutral
    list_Face[nIndex].nEmotion = output.index(max(output))
    for ii in range(7):
        list_Face[nIndex].fEmotionScore[ii] = output[ii]

    # torch.cuda.empty_cache()

def Gaze_Regression(list_Face, nIndex):
    if list_Face[nIndex].ptLE == [-1, -1] or list_Face[nIndex].fYaw == -1:
        return -1

    d2r = 3.141592 / 180.0
    #fDist = math.sqrt(pow(list_Face[nIndex].ptRE[0] - list_Face[nIndex].ptLE[0], 2) + pow(list_Face[nIndex].ptRE[1] - list_Face[nIndex].ptLE[1], 2))
    fDist = 20 * ((list_Face[nIndex].fPitch + list_Face[nIndex].fYaw)/2)

    normX = -1 * math.sin(d2r * list_Face[nIndex].fYaw) * fDist
    normY = -1 * math.sin(d2r * list_Face[nIndex].fPitch) * math.cos(d2r * list_Face[nIndex].fYaw) * fDist

    list_Face[nIndex].ptLED = [list_Face[nIndex].ptLE[0] + normX, list_Face[nIndex].ptLE[1] + normY]
    list_Face[nIndex].ptRED = [list_Face[nIndex].ptRE[0] + normX, list_Face[nIndex].ptRE[1] + normY]

    torch.cuda.empty_cache()

    return list_Face[nIndex].ptLED, list_Face[nIndex].ptRED

class pose_Detector():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.7, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # print(results.pose_landmarks)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def gaze_Detector(self, gaze, img_show):
        if gaze != None:
            # print("gaze", gaze)
            center_gaze_x = (gaze[0][0] + gaze[1][0]) / 2
            center_gaze_y = (gaze[0][1] + gaze[1][1]) / 2
            cv2.circle(img_show, (int(center_gaze_x), int(center_gaze_y)), 8, (0, 0, 255), -1)
            center_gaze = (int(center_gaze_x), int(center_gaze_y))
            return center_gaze


    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


# def soundcheck(self):
#
#     sound = AudioFileClip(self)  # self = .mp4
#
#     shortsound = sound.subclip("00:00:01", "00:00:10")  # audio from 1 to 10 seconds
#     # fileroute = 'C:/Users/withmind/Desktop/'
#     fileroute = '/home/ubuntu/project/'
#     filename = self + '.wav'
#     shortsound.write_audiofile(fileroute + filename, 44100, 2, 2000, "pcm_s32le")
#
#     y, sr = librosa.load(fileroute + filename)
#     sound_result = 0
#     for i in y:
#         if y[-0] == 0.00:
#             print('음성확인 > ', False)
#             sound_result = 1
#             break
#         else:
#             if i == 0.00:
#                 continue
#             else:
#                 print('음성확인 > ', True)
#                 sound_result = 0
#                 break
#
#     os.remove(fileroute + filename)
#
#     return sound_result

class shoulder_movement:
    def shoulder_vertically(shoulder, Landmark_list):
        landmark_no7_y = Landmark_list[13]
        shoulder_move_list = []
        shoulder_move_count = 0

        if shoulder[1] >= landmark_no7_y:
            shoulder_move_list.append(1)
        else:
            if len(shoulder_move_list) > 3:
                shoulder_move_count = 1
            shoulder_move_list = []

        return shoulder_move_count

    def shoulder_horizontally(shoulder, Landmark_list):
        landmark_no5_x = Landmark_list[8]
        landmark_no13_x = Landmark_list[24]
        shoulder_left_move_list = []
        shoulder_left_move_count = 0
        shoulder_right_move_list = []
        shoulder_right_move_count = 0

        if shoulder[0] <= landmark_no5_x:
            shoulder_left_move_list.append(1)

        elif shoulder[0] >= landmark_no13_x:
            shoulder_right_move_list.append(1)

        else:
            if len(shoulder_left_move_list) > 3:
                shoulder_left_move_count = 1
            if len(shoulder_right_move_list) > 3:
                shoulder_right_move_count = 1

        return (shoulder_left_move_count, shoulder_right_move_count)

    def shoulder_vertical_low_high(shoulder_list):
        shoulder_low_y = max(t[1] for t in shoulder_list)
        shoulder_high_y = min(t[1] for t in shoulder_list)
        for x, y in enumerate(shoulder_list):
            if shoulder_low_y in y:
                shoulder_low_point = y
            if shoulder_high_y in y:
                shoulder_high_point = y

        return (shoulder_low_point , shoulder_high_point)

    def shoulder_horizon_left_right(shoulder_list):
        shoulder_left_x = min(t[0] for t in shoulder_list)
        shoulder_right_x = max(t[0] for t in shoulder_list)

        for x, y in enumerate(shoulder_list):
            if shoulder_left_x in y:
                shoulder_extreme_left_point = y
            if shoulder_right_x in y:
                shoulder_extreme_right_point = y

        return (shoulder_extreme_left_point, shoulder_extreme_right_point)


class Average:
    # Gaze 표준편차 / Gaze_Avg = [x_std, y_std] 리스트안 튜플
    def Gaze_Avg(self):
        df = pd.DataFrame(self, columns=['x', 'y'])
        x_std = round(np.std(df['x']))
        y_std = round(np.std(df['y']))

        GazeAvg_x = x_std
        GazeAvg_y = y_std

        return GazeAvg_x,  GazeAvg_y

    # 어깨 상하 최고 길이 도출
    def vertically_Avg(Left_shoulder_high, Left_shoulder_low, Right_shoulder_high, Right_shoulder_low):
        Left_shoulder = Left_shoulder_high[1] - Left_shoulder_low[1]
        Right_shoulder = Right_shoulder_high[1] - Right_shoulder_low[1]

        if Left_shoulder > Right_shoulder:
            return Left_shoulder
        else:
            return Right_shoulder

    # 어깨 좌우 최고 길이 도출
    def horizontally_Avg(Center_shoulder_extreme_right, Center_shoulder_extreme_left):

        if Center_shoulder_extreme_right[0] >= Center_shoulder_extreme_left[0]:
            horizontally = Center_shoulder_extreme_right[0] - Center_shoulder_extreme_left[0]
        else:
            horizontally = Center_shoulder_extreme_left[0] - Center_shoulder_extreme_right[0]

        return horizontally

    def shoulder_left_count(self):
        count = 0
        for i in self:
            if i != 0:
                count += 1
            else:
                count += 0
        return count

    def shoulder_right_count(self):
        count = 0
        for i in self:
            if i != None:
                count += 0
            else:
                count += 1
        return count

    # 제스처 왼.오 큰값 도출
    def GestureTIME(Left_Hand_time, Right_Hand_time):
        Left_Hand = Left_Hand_time
        Right_Hand = Right_Hand_time

        if Left_Hand > Right_Hand:
            return Left_Hand
        else:
            return Right_Hand


    # def Average_csv(Gaze_value, Roll_value, Shoulder_value,vertically_value, horizontally_value, GestureTIME_value):
    #     # ftp = FTP()
    #     #
    #     # ftp.connect('withmind.cache.smilecdn.com', 21)
    #     # ftp.login('withmind', 'dnlemakdlsem1!')
    #     # ftp.cwd('./analy_result')
    #     filename = '/Average.csv'
    #     # fileroute = 'C:/Users/withmind/Desktop'
    #     fileroute = '/home/ubuntu/project'
    #
    #     # CSV 누적
    #     headersCSV = ['Gaze', 'Roll', 'Shoulder', 'vertically', 'horizontally', 'GestureTIME']
    #     dict = {'Gaze': Gaze_value,
    #             'Roll': Roll_value,
    #             'Shoulder': Shoulder_value,
    #             'vertically': vertically_value,
    #             'horizontally': horizontally_value,
    #             'GestureTIME': GestureTIME_value
    #             }
    #
    #     with open(fileroute + filename, mode='a', encoding='utf-8', newline='') as csvfile:
    #         wr = csv.DictWriter(csvfile, fieldnames=headersCSV)
    #         wr.writerow(dict)
    #
    #     os.remove(fileroute + filename)



class scoring:

    '''
    구간별 점수 : 100점, 80점, 60점, 40점, 20점
    <시선 x>                            <얼굴 각도 >
    1구간 1~5%  = ~3               1구간 1~5%  = ~1
    2구간 6~25%  = 4~5             2구간 6~25%  =  2~7
    3구간 26~75%  = 6~18           3구간 26~75%  = 8~28
    4구간 76~95%  = 19~33          4구간 76~95%  = 29~46
    5구간 95~100% = 34~            5구간 95~100% = 47~

    <시선 y>                          <어깨 각도>
    1구간 1~5%  = ~2                1구간 1~5%  =  ~0
    2구간 6~25%  = 3~5              2구간 6~25%  =  1~1
    3구간 26~75%  = 6~17            3구간 26~75%  = 2~4
    4구간 76~95%  =  18~30          4구간 76~95%  = 5~6
    5구간 95~100% = 31~             5구간 95~100% = 7~

    <어깨 상하>                       <어깨 좌우>
    1구간 1~5%  = ~13              1구간 1~5%  = ~10
    2구간 6~25%  = 14~26           2구간 6~25%  =  11~20
    3구간 26~75%  = 27~94          3구간 36~75%  = 21~69
    4구간 76~95%  = 95~175         4구간 76~95%  =  70~143
    5구간 95~100% = 176~           5구간 95~100% = 144~
    '''

    def GAZE_X_scoring(self):
        if self >= 34:
            return 20
        elif self >= 19:
            return 40
        elif self >= 6:
            return 60
        elif self >= 4:
            return 80
        else:
            return 100

    def GAZE_Y_scoring(self):
        if self >= 31:
            return 20
        elif self >= 18:
            return 40
        elif self >= 6:
            return 60
        elif self >= 3:
            return 80
        else:
            return 100

    def SHOULDER_VERTICAL_scoring(self):
        if self >= 176:
            return 20
        elif self >= 95:
            return 40
        elif self >= 27:
            return 60
        elif self >= 14:
            return 80
        else:
            return 100

    def SHOULDER_HORIZON_scoring(self):
        if self >= 144:
            return 20
        elif self >= 70:
            return 40
        elif self >= 21:
            return 60
        elif self >= 11:
            return 80
        else:
            return 100

    def FACE_ANGLE_scoring(self):
        if self >= 47:
            return 20
        elif self >= 29:
            return 40
        elif self >= 8:
            return 60
        elif self >= 2:
            return 80
        else:
            return 100
    #
    # def GESTURE_scoring(self):
    #     if self >= 2:
    #         return 60
    #     elif self >= 1:
    #         return 80
    #     else:
    #         return 100

    def SHOULDER_ANGLE_scoring(self):
        if self >= 7:
            return 20
        elif self >= 5:
            return 40
        elif self >= 2:
            return 60
        elif self >= 1:
            return 80
        else:
            return 100