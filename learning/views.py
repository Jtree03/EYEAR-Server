from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import UploadFileForm
from django.core.files.storage import FileSystemStorage
from eyear_server import settings
from . import sound_analysis
import os
import json


@csrf_exempt
def get_data(request):
    print(request)
    fs = FileSystemStorage()
    if os.path.isfile(os.path.join(settings.MEDIA_ROOT, 'file.wav')):
        os.remove(os.path.join(settings.MEDIA_ROOT, 'file.wav'))
    fs.save('file.wav', request.FILES['file'])
    answer = sound_analysis.main()
    return HttpResponse(json.dumps(str(answer)))

