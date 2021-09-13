from __future__ import absolute_import, unicode_literals
from celery.utils.log import get_task_logger
from .models import ImQzAnalysis, ImQzAnalysisJobword
import json
from .function import *
from konlpy.tag import Komoran
import numpy as np
import pandas as pd

logger = get_task_logger(__name__)


def video(userKey, qzGroup, groupCode, qzNum, fileKey, fileUrl, zqCode, stt, qzTts, documentSentimentScore,
          documentSentimentMagnitude, voiceDb, voiceDbScore, voiceTone, voiceToneScore, voiceSpeed, voiceSpeedScore, watchfullnessType):
    komoran = Komoran()
    stt = str(stt)
    input_pos = komoran.pos(stt)

    input_noun_list = []
    for i in range(0, len(input_pos)):
        if input_pos[i][1] == 'NNG' or input_pos[i][1] == 'NNP' or input_pos[i][1] == 'NR':
            input_noun_list.append(input_pos[i][0])

    # job_noun_file = pd.read_csv('C:/Users/withmind/Desktop/models/total_Noun_df.csv', encoding='UTF8')
    job_noun_file = pd.read_csv('/home/ubuntu/project/models/total_Noun_df.csv', encoding='UTF8')
    del job_noun_file['Unnamed: 0']
    zqCode = zqCode - 1
    code_noun = job_noun_file.loc[zqCode]
    code_noun = code_noun.values.tolist()
    code_noun_set = set(code_noun)
    input_data_set = set(input_noun_list)

    intersection_noun = input_data_set.intersection(code_noun_set)
    for i in intersection_noun:
        t = 0
        for ii in input_noun_list:
            if i in i == ii:
                t += 1
        noun_result = ImQzAnalysisJobword(file_key=fileKey, qz_group=qzGroup, word=str(i), count=t)
        noun_result.save


    if len(input_data_set) != 0 :
        if watchfullnessType == 1 :
            if qzNum == 1:
                Similarity = 0
            else:
                Similarity = round((len(intersection_noun) * 40) / len(input_data_set), 1)
        else:
            Similarity = 0
    else :
        Similarity = 0

    # if watchfullness_type == 1:
    #     if qzNum == 1:
    #         Similarity = 0
    #     else:
    #         if len(input_data_set) != 0:
    #             Similarity = round(len(intersection_noun) * 40 / len(input_data_set), 1)
    #         else:
    #             Similarity = 0
    # else:
    #     Similarity = 0

    watchfullness = Similarity

    FD_Net, Landmark_Net, Headpose_Net, Emotion_Net = Initialization()

    pose_detector = pose_Detector()
    vc = cv2.VideoCapture(fileUrl)
    FPS = vc.get(cv2.CAP_PROP_FPS)
    video_width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # sound_confirm = soundcheck(fileUrl)
    sound_confirm = 0

    # 데이터 담을 리스트
    Face_count_list = []
    Roll_list = []
    Emotion_list = []
    Gaze_list = []
    left_hand_point_list = []
    Left_hand_count = 0
    Left_Hand_point_result = []
    right_hand_point_list = []
    Right_hand_count = 0
    Right_Hand_point_result = []
    Left_shoulder_point_list = []
    Right_shoulder_point_list = []
    Center_shoulder_point_list = []
    left_shoulder_vertically_move_count = 0
    right_shoulder_vertically_move_count = 0
    center_shoulder_horizontally_left_move_count = 0
    center_shoulder_horizontally_right_move_count = 0
    Shoulder_slope_list = []



    bReady = True
    if vc.isOpened() == False:
        bReady = False
        return 1

    frame_num = 1
    while(bReady):
        ret, frame = vc.read()

        if ret:
            # if (frame_num == 1):
            #     standard_frame = cv2.flip(frame, 1)
            #     standard_img = standard_frame
            #
            #     standard_list_Face = []
            #     standard_face_detection = Face_Detection(FD_Net, standard_img, standard_list_Face)
            #     if len(standard_list_Face) > 0:
            #         standard_Landmark_list = Landmark_Detection(Landmark_Net, standard_img, standard_list_Face, 0)
            #     # frame_num += 1
            #     # print('000000000000000000', frame_num)

            if (frame_num % 5 == 0):

                frame = cv2.flip(frame, 1)
                img = frame
                list_Face = []
                # cv2.imwrite('frame%d.png' % frame_num, frame)
                Face_Detection(FD_Net, img, list_Face)
                Face_count_list.append(len(list_Face))
                # print('111111111111111111111', frame_num)

                # print("asdasada", len(list_Face))
                if len(list_Face) > 0:

                    #랜드마크 분석
                    Landmark_list = Landmark_Detection(Landmark_Net, img, list_Face, 0)
                    if len(Landmark_list) > 0:

                        #머리 각도 분석
                        Headpose = HeadPose_Estimation(Headpose_Net, img, list_Face, 0)
                        roll = Headpose[2].item()
                        Roll_list.append(roll)
                        #감정분석
                        Emotion_Classification(Emotion_Net, img, list_Face, 0)
                        # sEmotionLabel = ["surprise", "fear", "disgust", "happy", "sadness", "angry", "neutral"]
                        # sEmotionResult = "Emotion : %s" % sEmotionLabel[list_Face[0].nEmotion]
                        EmotionResult = list_Face[0].fEmotionScore
                        # print("감정>>>>>>>>>>>>>>>", EmotionResult, frame_num)

                        Emotion_list.append(EmotionResult)
                        #시선분석
                        gaze = Gaze_Regression(list_Face, 0)
                        gaze_value = pose_detector.gaze_Detector(gaze, img)
                        # if gaze_value[0] <= video_width and gaze_value <= video_height:
                        Gaze_list.append(gaze_value)
                        pose_detector.findPose(img)

                        lmList_pose = pose_detector.findPosition(img)
                        if len(lmList_pose) != 0:
                            # 왼손
                            if lmList_pose[16][1] < video_width and lmList_pose[16][2] < video_height:
                                left_hand = [lmList_pose[16][1], lmList_pose[16][2]]
                                left_hand_point_list.append(left_hand)
                            else:
                                if len(left_hand_point_list) > 5:
                                    Left_hand_count += 1
                                    Left_Hand_point_result.extend(left_hand_point_list)
                                left_hand_point_list = []

                            # 오른손
                            if lmList_pose[15][1] < video_width and lmList_pose[15][2] < video_height:
                                right_hand = [lmList_pose[15][1], lmList_pose[15][2]]
                                right_hand_point_list.append(right_hand)
                            else:
                                if len(right_hand_point_list) > 5:
                                    Right_hand_count += 1
                                    Right_Hand_point_result.extend(right_hand_point_list)
                                right_hand_point_list = []

                            # 어깨 위치
                            left_shoulder_point = (int(lmList_pose[12][1]), int(lmList_pose[12][2]))
                            right_shoulder_point = (int(lmList_pose[11][1]), int(lmList_pose[11][2]))
                            center_shoulder_x = int((lmList_pose[11][1] + lmList_pose[12][1]) / 2)
                            center_shoulder_y = int((lmList_pose[11][2] + lmList_pose[12][2]) / 2)
                            center_shoulder_point = (center_shoulder_x, center_shoulder_y)

                            Left_shoulder_point_list.append(left_shoulder_point)
                            Right_shoulder_point_list.append(right_shoulder_point)
                            Center_shoulder_point_list.append(center_shoulder_point)

                            # 어깨 움직임
                            if len(Landmark_list) != 0:
                                left_shoulder_vertically_move_count += shoulder_movement.shoulder_vertically(
                                    left_shoulder_point, Landmark_list)
                                right_shoulder_vertically_move_count += shoulder_movement.shoulder_vertically(
                                    right_shoulder_point, Landmark_list)


                                center_shoulder_horizontally_move_count = shoulder_movement.shoulder_horizontally(center_shoulder_point, Landmark_list)
                                center_shoulder_horizontally_left_move_count += center_shoulder_horizontally_move_count[0]
                                center_shoulder_horizontally_right_move_count += center_shoulder_horizontally_move_count[1]

                                # 어깨 기울기
                                shoulder_slope = (right_shoulder_point[1] - left_shoulder_point[1]) / (right_shoulder_point[0] - left_shoulder_point[0])
                                Shoulder_slope_list.append(shoulder_slope)
                            # if len(Landmark_list) != 0:
                            #     left_shoulder_vertically_move_count += shoulder_movement.shoulder_vertically(
                            #         left_shoulder_point, Landmark_list)
                            #     right_shoulder_vertically_move_count += shoulder_movement.shoulder_vertically(
                            #         right_shoulder_point, Landmark_list)
                            #
                            #
                            #     center_shoulder_horizontally_move_count = shoulder_movement.shoulder_horizontally(center_shoulder_point, Landmark_list)
                            #     center_shoulder_horizontally_left_move_count += center_shoulder_horizontally_move_count[0]
                            #     center_shoulder_horizontally_right_move_count += center_shoulder_horizontally_move_count[1]
                            #
                            #     # 어깨 기울기
                            #     shoulder_slope = (right_shoulder_point[1] - left_shoulder_point[1]) / (right_shoulder_point[0] - left_shoulder_point[0])
                            #     Shoulder_slope_list.append(shoulder_slope)
                    else :
                        continue
                else:
                    continue

            frame_num += 1
        else:
            break

    Face_count_exception = len(Face_count_list) - Face_count_list.count(1)
    # print(Face_count_list)
    if Face_count_exception * 6 >= (FPS * 7) :
        Face_analy_result = 1
    else:
        Face_analy_result = 0

    Emotion_surprise = 0
    Emotion_fear = 0
    Emotion_aversion = 0
    Emotion_happy = 0
    Emotion_sadness = 0
    Emotion_angry = 0
    Emotion_neutral = 0

    for i in range(len(Emotion_list)):
        Emotion_surprise = Emotion_surprise + Emotion_list[i][0]
        Emotion_fear = Emotion_fear + Emotion_list[i][1]
        Emotion_aversion = Emotion_aversion + Emotion_list[i][2]
        Emotion_happy = Emotion_happy + Emotion_list[i][3]
        Emotion_sadness = Emotion_sadness + Emotion_list[i][4]
        Emotion_angry = Emotion_angry + Emotion_list[i][5]
        Emotion_neutral = Emotion_neutral + Emotion_list[i][6]
        # Emotion_surprise += Emotion_list[i][0]
        # Emotion_fear += Emotion_list[i][1]
        # Emotion_aversion += Emotion_list[i][2]
        # Emotion_happy += Emotion_list[i][3]
        # Emotion_sadness += Emotion_list[i][4]
        # Emotion_angry += Emotion_list[i][5]
        # Emotion_neutral += Emotion_list[i][6]

    if len(Emotion_list) != 0:
        if Emotion_surprise != 0:
            Emotion_surprise_mean = Emotion_surprise * 100 / len(Emotion_list)
        else:
            Emotion_surprise_mean = 0
        if Emotion_fear != 0:
            Emotion_fear_mean = Emotion_fear * 100 / len(Emotion_list)
        else:
            Emotion_fear_mean = 0
        if Emotion_aversion != 0:
            Emotion_aversion_mean = Emotion_aversion * 100 / len(Emotion_list)
        else:
            Emotion_aversion_mean = 0
        if Emotion_happy != 0:
            Emotion_happy_mean = Emotion_happy * 100 / len(Emotion_list)
        else:
            Emotion_happy_mean = 0
        if Emotion_sadness != 0:
            Emotion_sadness_mean = Emotion_sadness * 100 / len(Emotion_list)
        else:
            Emotion_sadness_mean = 0
        if Emotion_angry != 0:
            Emotion_angry_mean = Emotion_angry * 100 / len(Emotion_list)
        else:
            Emotion_angry_mean = 0
        if Emotion_neutral != 0:
            Emotion_neutral_mean = Emotion_neutral * 100 / len(Emotion_list)
        else:
            Emotion_neutral_mean = 0
    else:
        Emotion_surprise_mean = 0
        Emotion_fear_mean = 0
        Emotion_aversion_mean = 0
        Emotion_happy_mean = 0
        Emotion_sadness_mean = 0
        Emotion_angry_mean = 0
        Emotion_neutral_mean = 0

    if len(Roll_list) != 0:
        roll = np.array(Roll_list)
        Roll_mean_value = np.mean(roll)
    else :
        Roll_mean_value = 0

    if len(Shoulder_slope_list) != 0:
        slope = np.array(Shoulder_slope_list)
        Shoulder_slope_mean_value = np.mean(slope)

    else:
        Shoulder_slope_mean_value = 0
    # print("왼쪽어깨", Left_shoulder_point_list)
    # print("오른쪽어깨", Right_shoulder_point_list)
    # print("어깨가운데", Center_shoulder_point_list)
    if len(Left_shoulder_point_list) != 0:
        left_shoulder_cal = shoulder_movement.shoulder_vertical_low_high(Left_shoulder_point_list)
        Left_shoulder_low = left_shoulder_cal[0]
        Left_shoulder_high = left_shoulder_cal[1]

    else :
        Left_shoulder_low = (0, 0)
        Left_shoulder_high = (0, 0)

    if len(Right_shoulder_point_list) != 0:
        right_shoulder_cal = shoulder_movement.shoulder_vertical_low_high(Right_shoulder_point_list)
        Right_shoulder_low = right_shoulder_cal[0]
        Right_shoulder_high = right_shoulder_cal[1]

    else:
        Right_shoulder_low = (0, 0)
        Right_shoulder_high = (0, 0)

    if len(Center_shoulder_point_list) != 0:
        center_shoulder_cal = shoulder_movement.shoulder_horizon_left_right(Center_shoulder_point_list)
        Center_shoulder_extreme_left = center_shoulder_cal[0]
        Center_shoulder_extreme_right = center_shoulder_cal[1]

    else:
        Center_shoulder_extreme_left = (0, 0)
        Center_shoulder_extreme_right = (0, 0)

    Left_Hand_time = float(len(Left_Hand_point_result) / 6)
    Right_Hand_time = float(len(Right_Hand_point_result) / 6)

    # 점수화_표준편차
    if len(Gaze_list) > 0:

        Gaze_std_value = Average.Gaze_Avg(Gaze_list)

    else:
        Gaze_std_value = 0
    Roll_mean_value
    Shoulder_slope_mean_value
    shouleder_vertically_max_length_value = Average.vertically_Avg(Left_shoulder_high,
                                                                   Left_shoulder_low,
                                                                   Right_shoulder_high,
                                                                   Right_shoulder_low)
    # print("asd", Center_shoulder_extreme_right)
    # print("Asdasd", Center_shoulder_extreme_left)
    shoulder_horizontally_max_length_value = Average.horizontally_Avg(Center_shoulder_extreme_right, Center_shoulder_extreme_left)
    # GestureTIME_value = Average.GestureTIME(Left_Hand_time, Right_Hand_time)

    gaze_dict = {"point": Gaze_list}
    left_shoulder_dict = {"low_spot": {"x": Left_shoulder_low[0], "y": Left_shoulder_low[1]},
                          "high_spot": {"x": Left_shoulder_high[0], "y": Left_shoulder_high[1]}}
    right_shoulder_dict = {"low_spot": {"x": Right_shoulder_low[0], "y": Right_shoulder_low[1]},
                           "high_spot": {"x": Right_shoulder_high[0], "y": Right_shoulder_high[1]}}
    center_shoulder_dict = {"left_spot": {"x": Center_shoulder_extreme_left[0], "y": Center_shoulder_extreme_left[1]},
                            "right_spot": {"x": Center_shoulder_extreme_right[0], "y": Center_shoulder_extreme_right[1]}}
    # print("왼손>>>", Left_Hand_point_result)
    # print("오른손>>>", Right_Hand_point_result)
    left_hand_dict = {"point": Left_Hand_point_result}
    right_hand_dict = {"point": Right_Hand_point_result}
    # print("왼손>>>", left_hand_dict)
    # print("오른손>>>", right_hand_dict)

    res = ImQzAnalysis(file_key=fileKey, user_key=userKey, qz_group=qzGroup, qz_num=qzNum, group_code=groupCode,
                       face_check=Face_analy_result,
                       sound_check=sound_confirm, emotion_surprise=round(Emotion_surprise_mean, 5),
                       emotion_fear=round(Emotion_fear_mean, 5), emotion_aversion=round(Emotion_aversion_mean, 5),
                       emotion_happy=round(Emotion_happy_mean, 5), emotion_sadness=round(Emotion_sadness_mean, 5),
                       emotion_angry=round(Emotion_angry_mean, 5), emotion_neutral=round(Emotion_neutral_mean, 5),
                       gaze=json.dumps(gaze_dict), face_angle=round(Roll_mean_value, 5),
                       shoulder_angle=round(Shoulder_slope_mean_value, 5), left_shoulder=json.dumps(left_shoulder_dict),
                       left_shoulder_move_count=left_shoulder_vertically_move_count,
                       right_shoulder=json.dumps(right_shoulder_dict), right_shoulder_move_count=right_shoulder_vertically_move_count,
                       center_shoulder=json.dumps(center_shoulder_dict),
                       center_shoulder_left_move_count=center_shoulder_horizontally_left_move_count,
                       center_shoulder_right_move_count=center_shoulder_horizontally_right_move_count,
                       left_hand=json.dumps(left_hand_dict), left_hand_time=Left_Hand_time,
                       left_hand_move_count=Left_hand_count,
                       right_hand=json.dumps(right_hand_dict), right_hand_time=Right_Hand_time,
                       right_hand_move_count=Right_hand_count,
                       gaze_x_score=scoring.GAZE_X_scoring(Gaze_std_value[0]),
                       gaze_y_score=scoring.GAZE_Y_scoring(Gaze_std_value[1]),
                       shoulder_vertical_score=scoring.SHOULDER_VERTICAL_scoring(shouleder_vertically_max_length_value),
                       shoulder_horizon_score=scoring.SHOULDER_HORIZON_scoring(shoulder_horizontally_max_length_value),
                       face_angle_score=scoring.FACE_ANGLE_scoring(Roll_mean_value),
                       gesture_score=scoring.SHOULDER_ANGLE_scoring(Shoulder_slope_mean_value),
                       watchfullness=watchfullness, document_sentiment_score=documentSentimentScore,
                       document_sentiment_magnitude=documentSentimentMagnitude, voice_db=voiceDb,
                       voice_db_score=voiceDbScore,
                       voice_tone=voiceTone, voice_tone_score=voiceToneScore, voice_speed=voiceSpeed,
                       voice_speed_score=voiceSpeedScore, stt=stt, qz_tts=qzTts, watchfullness_type=watchfullnessType)
    # , job_noun=json.dumps(job_noun, ensure_ascii=False))

    res.save()

    return 0