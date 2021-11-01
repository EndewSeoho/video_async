from django.db import models
from django.utils import timezone


class ImQzAnalysis(models.Model):
    anlys_key = models.AutoField(db_column='ANLYS_KEY', primary_key=True)  # Field name made lowercase.
    file_key = models.IntegerField(db_column='FILE_KEY', blank=True, null=True)  # Field name made lowercase.
    user_key = models.IntegerField(db_column='USER_KEY', blank=True, null=True)  # Field name made lowercase.
    qz_group = models.IntegerField(db_column='QZ_GROUP', blank=True, null=True)  # Field name made lowercase.
    qz_num = models.IntegerField(db_column='QZ_NUM', blank=True, null=True)  # Field name made lowercase.
    group_code = models.IntegerField(db_column='GROUP_CODE', blank=True, null=True)  # Field name made lowercase.
    face_check = models.CharField(db_column='FACE_CHECK', max_length=45, blank=True, null=True)  # Field name made lowercase.
    sound_check = models.CharField(db_column='SOUND_CHECK', max_length=45, db_collation='latin1_swedish_ci', blank=True, null=True)  # Field name made lowercase.
    emotion_surprise = models.FloatField(db_column='EMOTION_SURPRISE', blank=True, null=True)  # Field name made lowercase.
    emotion_fear = models.FloatField(db_column='EMOTION_FEAR', blank=True, null=True)  # Field name made lowercase.
    emotion_aversion = models.FloatField(db_column='EMOTION_AVERSION', blank=True, null=True)  # Field name made lowercase.
    emotion_happy = models.FloatField(db_column='EMOTION_HAPPY', blank=True, null=True)  # Field name made lowercase.
    emotion_sadness = models.FloatField(db_column='EMOTION_SADNESS', blank=True, null=True)  # Field name made lowercase.
    emotion_angry = models.FloatField(db_column='EMOTION_ANGRY', blank=True, null=True)  # Field name made lowercase.
    emotion_neutral = models.FloatField(db_column='EMOTION_NEUTRAL', blank=True, null=True)  # Field name made lowercase.
    gaze = models.CharField(db_column='GAZE', max_length=5000, db_collation='latin1_swedish_ci', blank=True, null=True)  # Field name made lowercase.
    face_angle = models.FloatField(db_column='FACE_ANGLE', blank=True, null=True)  # Field name made lowercase.
    shoulder_angle = models.FloatField(db_column='SHOULDER_ANGLE', blank=True, null=True)  # Field name made lowercase.
    left_shoulder = models.CharField(db_column='LEFT_SHOULDER', max_length=5000, db_collation='latin1_swedish_ci', blank=True, null=True)  # Field name made lowercase.
    left_shoulder_move_count = models.IntegerField(db_column='LEFT_SHOULDER_MOVE_COUNT', blank=True, null=True)  # Field name made lowercase.
    right_shoulder = models.CharField(db_column='RIGHT_SHOULDER', max_length=5000, db_collation='latin1_swedish_ci', blank=True, null=True)  # Field name made lowercase.
    right_shoulder_move_count = models.IntegerField(db_column='RIGHT_SHOULDER_MOVE_COUNT', blank=True, null=True)  # Field name made lowercase.
    center_shoulder = models.CharField(db_column='CENTER_SHOULDER', max_length=5000, db_collation='latin1_swedish_ci', blank=True, null=True)  # Field name made lowercase.
    center_shoulder_left_move_count = models.IntegerField(db_column='CENTER_SHOULDER_LEFT_MOVE_COUNT', blank=True, null=True)  # Field name made lowercase.
    center_shoulder_right_move_count = models.IntegerField(db_column='CENTER_SHOULDER_RIGHT_MOVE_COUNT', blank=True, null=True)  # Field name made lowercase.
    left_hand = models.CharField(db_column='LEFT_HAND', max_length=500, db_collation='latin1_swedish_ci', blank=True, null=True)  # Field name made lowercase.
    left_hand_time = models.FloatField(db_column='LEFT_HAND_TIME', blank=True, null=True)  # Field name made lowercase.
    left_hand_move_count = models.IntegerField(db_column='LEFT_HAND_MOVE_COUNT', blank=True, null=True)  # Field name made lowercase.
    right_hand = models.CharField(db_column='RIGHT_HAND', max_length=5000, db_collation='latin1_swedish_ci', blank=True, null=True)  # Field name made lowercase.
    right_hand_time = models.FloatField(db_column='RIGHT_HAND_TIME', blank=True, null=True)  # Field name made lowercase.
    right_hand_move_count = models.IntegerField(db_column='RIGHT_HAND_MOVE_COUNT', blank=True, null=True)  # Field name made lowercase.
    gaze_x_score = models.IntegerField(db_column='GAZE_X_SCORE', blank=True, null=True)  # Field name made lowercase.
    gaze_y_score = models.IntegerField(db_column='GAZE_Y_SCORE', blank=True, null=True)  # Field name made lowercase.
    shoulder_vertical_score = models.IntegerField(db_column='SHOULDER_VERTICAL_SCORE', blank=True, null=True)  # Field name made lowercase.
    shoulder_horizon_score = models.IntegerField(db_column='SHOULDER_HORIZON_SCORE', blank=True, null=True)  # Field name made lowercase.
    face_angle_score = models.IntegerField(db_column='FACE_ANGLE_SCORE', blank=True, null=True)  # Field name made lowercase.
    gesture_score = models.IntegerField(db_column='GESTURE_SCORE', blank=True, null=True)  # Field name made lowercase.
    voice_db = models.IntegerField(db_column='VOICE_DB', blank=True, null=True)  # Field name made lowercase.
    voice_db_score = models.IntegerField(db_column='VOICE_DB_SCORE', blank=True, null=True)  # Field name made lowercase.
    voice_tone = models.IntegerField(db_column='VOICE_TONE', blank=True, null=True)  # Field name made lowercase.
    voice_tone_score = models.IntegerField(db_column='VOICE_TONE_SCORE', blank=True, null=True)  # Field name made lowercase.
    voice_speed = models.IntegerField(db_column='VOICE_SPEED', blank=True, null=True)  # Field name made lowercase.
    voice_speed_score = models.IntegerField(db_column='VOICE_SPEED_SCORE', blank=True, null=True)  # Field name made lowercase.
    document_sentiment_score = models.FloatField(db_column='DOCUMENT_SENTIMENT_SCORE', blank=True, null=True)  # Field name made lowercase.
    document_sentiment_magnitude = models.FloatField(db_column='DOCUMENT_SENTIMENT_MAGNITUDE', blank=True, null=True)  # Field name made lowercase.
    stt = models.CharField(db_column='STT', max_length=2500, blank=True, null=True)  # Field name made lowercase.
    qz_tts = models.CharField(db_column='QZ_TTS', max_length=1000, blank=True, null=True)  # Field name made lowercase.
    watchfullness = models.IntegerField(db_column='WATCHFULLNESS', blank=True, null=True)  # Field name made lowercase.
    watchfullness_type = models.IntegerField(db_column='WATCHFULLNESS_TYPE')
    regdate = models.DateTimeField(db_column='REGDATE', blank=True, null=True, auto_now_add=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'IM_QZ_ANALYSIS'


class ImQzAnalysisJobword(models.Model):
    aj_key = models.AutoField(db_column='AJ_KEY', primary_key=True)  # Field name made lowercase.
    file_key = models.IntegerField(db_column='FILE_KEY', blank=True, null=True)  # Field name made lowercase.
    qz_group = models.IntegerField(db_column='QZ_GROUP', blank=True, null=True)  # Field name made lowercase.
    word = models.CharField(db_column='WORD', max_length=1000, blank=True, null=True)  # Field name made lowercase.
    count = models.IntegerField(db_column='COUNT', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'IM_QZ_ANALYSIS_JOBWORD'

# class ImQzAnalysisFile(models.Model):
#     idx = models.AutoField(db_column='IDX', primary_key=True)  # Field name made lowercase.
#     file_key = models.IntegerField(db_column='FILE_KEY', blank=True, null=True)  # Field name made lowercase.
#     qz_num = models.IntegerField(db_column='QZ_NUM', blank=True, null=True)  # Field name made lowercase.
#     qz_group = models.IntegerField(db_column='QZ_GROUP', blank=True, null=True)  # Field name made lowercase.
#     qz_type = models.CharField(db_column='QZ_TYPE', max_length=45, blank=True, null=True)  # Field name made lowercase.
#
#     class Meta:
#         managed = False
#         db_table = 'IM_QZ_ANALYSIS_FILE'

# class ImQzFile(models.Model):
#     file_key = models.AutoField(db_column='FILE_KEY', primary_key=True)  # Field name made lowercase.
#     qz_group = models.IntegerField(db_column='QZ_GROUP', blank=True, null=True)  # Field name made lowercase.
#     qz_platform = models.IntegerField(db_column='QZ_PLATFORM', blank=True, null=True)  # Field name made lowercase.
#     user_key = models.IntegerField(db_column='USER_KEY', blank=True, null=True)  # Field name made lowercase.
#     qz_num = models.IntegerField(db_column='QZ_NUM', blank=True, null=True)  # Field name made lowercase.
#     qz_type = models.CharField(db_column='QZ_TYPE', max_length=45, blank=True, null=True)  # Field name made lowercase.
#     group_code = models.IntegerField(db_column='GROUP_CODE', blank=True, null=True)  # Field name made lowercase.
#     rtsp_url = models.CharField(db_column='RTSP_URL', max_length=500, blank=True, null=True)  # Field name made lowercase.
#     file_url = models.CharField(db_column='FILE_URL', max_length=500, blank=True, null=True)  # Field name made lowercase.
#     json_express = models.CharField(db_column='JSON_EXPRESS', max_length=500, blank=True, null=True)  # Field name made lowercase.
#     json_action = models.CharField(db_column='JSON_ACTION', max_length=500, blank=True, null=True)  # Field name made lowercase.
#     json_tone = models.CharField(db_column='JSON_TONE', max_length=500, blank=True, null=True)  # Field name made lowercase.
#     json_db = models.CharField(db_column='JSON_DB', max_length=500, blank=True, null=True)  # Field name made lowercase.
#     json_speed = models.CharField(db_column='JSON_SPEED', max_length=500, blank=True, null=True)  # Field name made lowercase.
#     thum_url = models.CharField(db_column='THUM_URL', max_length=500, blank=True, null=True)  # Field name made lowercase.
#     regdate = models.CharField(db_column='REGDATE', max_length=45, blank=True, null=True)  # Field name made lowercase.
#     # regdate = models.CharField(db_column='REG+DATE', max_length=45, blank=True, null=True)  # Field name made lowercase.
#     qz_org_tts = models.CharField(db_column='QZ_ORG_TTS', max_length=1000, blank=True, null=True)  # Field name made lowercase.
#
#     class Meta:
#         managed = False
#         db_table = 'IM_QZ_FILE'

class ImQzFile(models.Model):
    file_key = models.AutoField(db_column='FILE_KEY', primary_key=True)  # Field name made lowercase.
    qz_group = models.IntegerField(db_column='QZ_GROUP', blank=True, null=True)  # Field name made lowercase.
    qz_platform = models.IntegerField(db_column='QZ_PLATFORM', blank=True, null=True)  # Field name made lowercase.
    user_key = models.IntegerField(db_column='USER_KEY', blank=True, null=True)  # Field name made lowercase.
    qz_num = models.IntegerField(db_column='QZ_NUM', blank=True, null=True)  # Field name made lowercase.
    qz_type = models.CharField(db_column='QZ_TYPE', max_length=45, blank=True, null=True)  # Field name made lowercase.
    group_code = models.IntegerField(db_column='GROUP_CODE', blank=True, null=True)  # Field name made lowercase.
    rtsp_url = models.CharField(db_column='RTSP_URL', max_length=500, blank=True, null=True)  # Field name made lowercase.
    file_url = models.CharField(db_column='FILE_URL', max_length=500, blank=True, null=True)  # Field name made lowercase.
    json_express = models.CharField(db_column='JSON_EXPRESS', max_length=500, blank=True, null=True)  # Field name made lowercase.
    json_action = models.CharField(db_column='JSON_ACTION', max_length=500, blank=True, null=True)  # Field name made lowercase.
    json_tone = models.CharField(db_column='JSON_TONE', max_length=500, blank=True, null=True)  # Field name made lowercase.
    json_db = models.CharField(db_column='JSON_DB', max_length=500, blank=True, null=True)  # Field name made lowercase.
    json_speed = models.CharField(db_column='JSON_SPEED', max_length=500, blank=True, null=True)  # Field name made lowercase.
    thum_url = models.CharField(db_column='THUM_URL', max_length=500, blank=True, null=True)  # Field name made lowercase.
    regdate = models.CharField(db_column='REGDATE', max_length=45, blank=True, null=True)  # Field name made lowercase.
    qz_org_tts = models.CharField(db_column='QZ_ORG_TTS', max_length=1000, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'IM_QZ_FILE'
