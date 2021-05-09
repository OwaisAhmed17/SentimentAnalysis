from django.shortcuts import render
import requests
import sys
from subprocess import run,PIPE
from . import tweet

def button(request):

    return render(request,'home.html')

def output(request):
    data=requests.get("https://www.google.com/")
    print(data.text)
    data=data.text
    return render(request,'home.html',{'data':data})
def external(request):
    inp= request.POST.get('param')
    chart = tweet.sentiment(inp,100)
    return render(request,'home.html',{'chart':chart})