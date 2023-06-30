import requests
import json

# Create your views here.
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse
from rest_framework import status
from enum import Enum
import traceback

from text_summarization.summarizer import use_word_frequency as basic_summarize
from text_summarization.summarizer import use_t5

class TextSummarizer(APIView):
    class Summarizer(str, Enum):
        BASIC = "basic"
        T5 = "t5"
        
    def get(self, request):
        return JsonResponse({
            "message": "Text Summarization API",
            "notes":
                "This endpoint is used to summarize the given document or text.\n"
                "There are three summarization methods available:\n"
                "    1. Basic Summarization. Using frequency table and statistical method\n"
                "    3. T5 Summarization base\n"
                "To use the service, send a POST request to this endpoint with the following data:\n"
                "    1. text: the text to be summarized\n"
                "    2. keep: the percentage of the text to be kept (default: 0.3)\n"
                "    3. summarizer (basic | t5): the summarization method to be used (default: t5)\n"
        }, status=status.HTTP_200_OK)

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
                "summary": use_t5(
                    text, 
                    keep
                )
            }, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({"summary": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
class Upload(APIView):
    """
    A view that can accept POST requests with JSON content.
    """
    parser_classes = (MultiPartParser,)

    def post(self, request, format=None):
        try:
            obj_file = request.data['filename']
            payload = {
                'isOverlayRequired': False,
                'apikey': 'K85079096388957',
            }
            r = requests.post(
                    'https://api.ocr.space/parse/image',
                    files={'filename': obj_file},
                    data=payload,
                )
            decode = r.content.decode()
            res_data = json.loads(decode)
            parsed_text = res_data['ParsedResults'][0]['ParsedText']
            return JsonResponse({'ocr': parsed_text}, status=status.HTTP_200_OK)
        except Exception as e:
            return JsonResponse({'ocr': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)