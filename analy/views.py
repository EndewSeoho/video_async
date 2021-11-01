from asgiref.sync import sync_to_async, async_to_sync
from rest_framework.response import Response
from .models import ImQzFile, ImQzAnalysis
import json
from .tasks import video
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.http import JsonResponse
import time
import asyncio
from multiprocessing import Pool, freeze_support

# @api_view(['GET'])
# def helloAPI(request):
#     return Response('hello')

@api_view(['POST'])
@permission_classes([AllowAny])
def post(request):
    request1 = json.dumps(request.data)
    insert_data = json.loads(request1)
    userKey = insert_data.get("userKey")
    qzGroup = insert_data.get("qzGroup")
    groupCode = insert_data.get("groupCode")
    qzNum = insert_data.get("qzNum")
    fileKey = insert_data.get("fileKey")
    fileUrl = insert_data.get("fileUrl")
    zqCode = insert_data.get("zqCode")
    watchfullnessType = insert_data.get("watchfullnessType")
    # zqCode = str(zqCode)
    stt = insert_data.get("stt")
    qzTts = insert_data.get("qzTts")
    documentSentimentScore = insert_data.get("documentSentimentScore")
    documentSentimentMagnitude = insert_data.get("documentSentimentMagnitude")
    voiceDb = insert_data.get("voiceDb")
    voiceDbScore = insert_data.get("voiceDbScore")
    voiceTone = insert_data.get("voiceTone")
    voiceToneScore = insert_data.get("voiceToneScore")
    voiceSpeed = insert_data.get("voiceSpeed")
    voiceSpeedScore = insert_data.get("voiceSpeedScore")

    try:
        video(userKey, qzGroup, groupCode, qzNum, fileKey, fileUrl, zqCode, stt, qzTts, documentSentimentScore, documentSentimentMagnitude, voiceDb, voiceDbScore, voiceTone, voiceToneScore, voiceSpeed, voiceSpeedScore, watchfullnessType)


        db_update = ImQzFile.objects.filter(file_key=fileKey).first()
        # print(db_update)
        db_update.qz_type = 'Y'
        db_update.save()
    #
    except Exception as e:
        db_update = ImQzFile.objects.filter(file_key=fileKey).first()
        db_update.qz_type = 'Y'
        db_update.save()
        point_dict = {"point": []}
        vertical_spot_dict = {"low_spot": {"x": 0, "y": 0},
                          "high_spot": {"x": 0, "y": 0}}
        horizon_spot_dict = {"left_spot": {"x": 0, "y": 0},
                            "right_spot": {"x": 0, "y": 0}}
        res = ImQzAnalysis(file_key=fileKey, user_key=userKey, qz_group=qzGroup, qz_num=qzNum, group_code=groupCode,
                           face_check=1,
                           sound_check=1, emotion_surprise=0,
                           emotion_fear=0, emotion_aversion=0,
                           emotion_happy=0, emotion_sadness=0,
                           emotion_angry=0, emotion_neutral=0,
                           gaze=json.dumps(point_dict), face_angle=0,
                           shoulder_angle=0,
                           left_shoulder=json.dumps(vertical_spot_dict),
                           left_shoulder_move_count=0,
                           right_shoulder=json.dumps(vertical_spot_dict),
                           right_shoulder_move_count=0,
                           center_shoulder=json.dumps(horizon_spot_dict),
                           center_shoulder_left_move_count=0,
                           center_shoulder_right_move_count=0,
                           left_hand=json.dumps(point_dict), left_hand_time=0,
                           left_hand_move_count=0,
                           right_hand=json.dumps(point_dict), right_hand_time=0,
                           right_hand_move_count=0,
                           gaze_x_score=0,
                           gaze_y_score=0,
                           shoulder_vertical_score=0,
                           shoulder_horizon_score=0,
                           face_angle_score=0,
                           gesture_score=0,
                           watchfullness=0, document_sentiment_score=0,
                           document_sentiment_magnitude=0, voice_db=0,
                           voice_db_score=0,
                           voice_tone=0, voice_tone_score=0, voice_speed=0,
                           voice_speed_score=0, stt=stt, qz_tts=qzTts,
                           watchfullness_type=watchfullnessType)
        res.save()
        response_dict = {"msessage": "Fail", "status": "200"}

        return JsonResponse(response_dict)
        # return Response(response_dict)



    response_dict = {"msessage": "OK", "status": "200"}


    return JsonResponse(response_dict)
    # return Response(response_dict)

# def handler500(request):
#     print("request", request)
#     request1 = json.dumps(request.data)
#     insert_data = json.loads(request1)
#     fileKey = insert_data.get("fileKey")
#     context = "asdAdsasd"
#     response = render(request, context=context)
#     response.status_code = 2021
#     db_update = ImQzFile.objects.get(file_key=fileKey)
#     db_update.qz_type = 'F'
#     db_update.save()
#     return response
