from django.shortcuts import render
from matplotlib.pyplot import bar
import requests
import sys
from subprocess import run,PIPE

from textblob.blob import Word
from . import tweet
from . import textanalysis

def button(request):

    return render(request,'home.html')

def output(request):
    data=requests.get("https://www.google.com/")
    print(data.text)
    data=data.text
    return render(request,'home.html',{'data':data})
def external(request):
    inp = request.POST.get('param')
    num = request.POST.get('num')
    num = int(num)
    out = tweet.sentiment(inp,num)
    chart = out[0]
    text = out[1]
    nums = out[2]
    word = out[3]
    outstr = textanalysis.predict_personality(text)[0]
    pre = textanalysis.predict_personality(text)[1]
    pos = nums[0]
    neg = nums[1]
    avg_sub = nums[2]
    avg_pol = nums[3]
    not_exist = None
    if pos == -1:
        pos = None
        neg = None
        avg_sub = None
        avg_pol =None
        not_exist = True

    i0000=None
    i0001=None
    i0010=None
    i0011=None
    i0100=None
    i0101=None
    i0110=None
    i0111=None
    i1000=None
    i1001=None
    i1010=None
    i1011=None
    i1100=None
    i1101=None
    i1110=None
    i1111=None

    if pre==[0,0,0,0]:
        i0000 = True
        
    elif pre==[0,0,0,1]:
        i0001 = True
        
    elif pre==[0,0,1,0]:
        i0010 = True
        
    elif pre ==[0,0,1,1]:
        i0011 = True
        
    elif pre==[0,1,0,0]:
        i0100 = True
        
    elif pre==[0,1,0,1]:
        i0101 = True
        
    elif pre==[0,1,1,0]:
        i0110 = True
        
    elif pre==[0,1,1,1]:
        i0111 = True
        
    elif pre == [1,0,0,0]:
        i1000 = True
        
    elif pre == [1,0,0,1]:
        i1001 = True
        
    elif pre == [1,0,1,0]:
        i1010 = True
        
    elif pre == [1,0,1,1]:
        i1011 = True
        
    elif pre == [1,1,0,0]:
        i1100 = True
        
    elif pre == [1,1,0,1]:
        i1101 = True
        
    elif pre == [1,1,1,0]:
        i1110 = True
        
    elif pre == [1,1,1,1]:
        i1111 = True
        


    
    
    return render(request,'home.html',{'exist':not_exist,'chart':chart,'pos':pos,'neg':neg,'avg_sub':avg_sub,'avg_pol':avg_pol,'word':word,'pers':outstr['pers'],'cog':outstr['cog'],'dom':outstr['dom'],'aux':outstr['aux'],'ter':outstr['ter'],'inf':outstr['inf'],'car':outstr['car'],'int':outstr['int'],'fri':outstr['fri'],'par':outstr['par'],'rel':outstr['rel'],'i0000':i0000,'i0001':i0001,'i0010':i0010,'i0011':i0011,'i0100':i0100,'i0101':i0101,'i0110':i0110,'i0111':i0111,'i1000':i1000,'i1001':i1001,'i1010':i1010,'i1011':i1011,'i1100':i1100,'i1101':i1101,'i1110':i1110,'i1111':i1111})