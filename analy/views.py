from rest_framework.response import Response
from .models import ImQzFile
import json
from .tasks import video
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.http import JsonResponse

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

        db_update = ImQzFile.objects.get(file_key=fileKey)
        db_update.qz_type = 'Y'
        db_update.save()
    #
    except Exception as e:
        db_update = ImQzFile.objects.get(file_key=fileKey)
        db_update.qz_type = 'F'
        db_update.save()
        response_dict = {"msessage": "Fail", "status": "200"}

        return JsonResponse(response_dict)

    # if result == 0:
    #     db_update = ImQzFile.objects.get(file_key=fileKey)
    #     db_update.qz_type = 'Y'
    #     db_update.save()
    # else :
    #     db_update = ImQzFile.objects.get(file_key=fileKey)
    #     db_update.qz_type = 'F'
    #     db_update.save()
    # print(result)
    response_dict = {"msessage": "OK", "status": "200"}


    return JsonResponse(response_dict)

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