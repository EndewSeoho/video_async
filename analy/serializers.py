from rest_framework import serializers
from .models import ImQzAnalysis

class AnalySerializer(serializers.ModelSerializer):
    class Meta:
        model = ImQzAnalysis
        fields = '__all__'