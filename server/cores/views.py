from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from django.http import JsonResponse
from rest_framework import status

from text_summarization.basic import summarize as basic_summarize

class TextSummarizer(APIView):
    def post(self, request):
        try:
            data = request.data
            text = data["text"]

            return JsonResponse({
                "summary": basic_summarize(
                    text, 
                    data.get('keep', 0.3)
                )
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)