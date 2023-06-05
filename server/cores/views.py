from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from django.http import JsonResponse
from rest_framework import status
from enum import Enum

from text_summarization.summarizer import use_word_frequency as basic_summarize

class TextSummarizer(APIView):
    class Summarizer(str, Enum):
        BASIC = "basic"
        BERT = "bert"
        T5 = "t5"
        GPT = "gpt"
        
    def get(self, request):
        return JsonResponse({
            "message": "Text Summarization API",
            "notes": [
                "This endpoint is used to summarize the given document or text.",
                "There are three summarization methods available:",
                "    1. Basic Summarization. Using frequency table and statistical method",
                "    2. BERT Summarization base",
                "    3. T5 Summarization base",
                "    4. GPT Summarization base, GPT 2 is used for now",
                "To use the service, send a POST request to this endpoint with the following data:",
                "    1. text: the text to be summarized",
                "    2. keep: the percentage of the text to be kept (default: 0.3)",
                "    3. summarizer (basic | bert | t5 | gpt 2): the summarization method to be used (default: t5)",
            ]
        }, indent = 4, status=status.HTTP_200_OK)

    def post(self, request):
        try:
            data = request.data
            text, method, keep = data.get("text", None), TextSummarizer.Summarizer(data.get("method", "t5")), data.get('keep', 0.3)
            
            if type(keep) != float:
                try: keep = float(keep)
                except: keep = 0.3

            if not text:
                raise "Error while processing the requests, text should be provided. If something is going wrong, remind us via [link](https://google.com)."
            
            return JsonResponse({
                "summary": basic_summarize(
                    text, 
                    keep
                )
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return JsonResponse({"summary": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)