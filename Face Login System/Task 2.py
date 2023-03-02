import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import tkinter.ttk as ttk
import tkinter.font as font
import numpy as np
from PIL import Image, ImageTk
import csv
import json
import pandas as pd

window=tk.Tk()
window.title("Login/Logout Website")
window.geometry('1366x768')

def loading_data():
    file=open('data.txt','r',encoding='utf-8')
    data=json.load(file)
    file.close()
    return data

def saving_data(data):
    file=open('data.txt','w',encoding='utf-8')
    json.dump(data,file,ensure_ascii=False)
    file.close()

data=loading_data()
print(data)

login_label= tk.Label(window, text="Login Here" ,bg="light blue"  ,fg="black"  ,width=20 ,height=1,font=('times', 30, 'bold')) 
login_label.place(x=40, y=150)

reg_label= tk.Label(window, text="Register Here" ,bg="light blue"  ,fg="black"  ,width=20 ,height=1,font=('times', 30, 'bold')) 
reg_label.place(x=700, y=150)

msg_label=tk.Label(window,text="Notification: ",bg='light blue' ,fg='black',width=10,height=1,font=('times',15,'bold'))
msg_label.place(x=350,y=500)
message = tk.Label(window,text="",bg="light blue" ,fg="black", width=30, height=1, font=('times',15,'bold'))
message.place(x=500,y=500)

lbl = tk.Label(window, text="Enter ID :",width=10,height=1  ,fg="black"  ,bg="gold" ,font=('times', 15, ' bold ') ) 
lbl.place(x=40, y=210)

txt = tk.Entry(window,width=25  ,bg="gold" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=250, y=210)

lbl2= tk.Label(window, text="Enter Password :",width=15  ,height=1  ,fg="black"  ,bg="gold" ,font=('times', 15, ' bold ') ) 
lbl2.place(x=40, y=250)

txt2 = tk.Entry(window,width=25  ,bg="gold" ,show='*',fg="black",font=('times', 15, ' bold '))
txt2.place(x=250, y=250)

lbl3 = tk.Label(window, text="Enter ID :",width=10  ,height=1  ,fg="black"  ,bg="gold" ,font=('times', 15, ' bold ') ) 
lbl3.place(x=700, y=213)

txt3 = tk.Entry(window,width=25  ,bg="gold" ,fg="black",font=('times', 15, ' bold '))
txt3.place(x=940, y=213)

lbl4= tk.Label(window, text="Enter Password",width=15  ,height=1  ,fg="black"  ,bg="gold" ,font=('times', 15, ' bold ') ) 
lbl4.place(x=700, y=250)

txt4 = tk.Entry(window,width=25  ,bg="gold" ,show='*',fg="black",font=('times', 15, ' bold '))
txt4.place(x=940, y=250)

def login_clear():
    txt.delete(0,'end')
    txt2.delete(0,'end')
def reg_clear():
    txt3.delete(0,'end')
    txt4.delete(0,'end')

def TrackImages(UserId):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df=pd.read_csv("Details\Details.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX          
    run_count=0;run=True
    while run:
        
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print(Id, conf)
            if(conf < 50):
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                if (str(Id)==UserId):
                    message.configure(text="Face Recognsed Successfully")
                    run=False
            else:
                Id='Unknown'                
                tt=str(Id)            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        run_count += 1    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q') or run_count==150):
            message.configure(text="Unable to Recognise Face")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    


def login_submit():
    a=txt.get()
    b=txt2.get()
    if (a in data):
        if(data[a] == b):
            TrackImages(a)
        else:
            message.configure(text="Id and Password does not Match")
    else:
        message.configure(text="Entered Id does not Exists")

    login_clear()

def TakeImages():        
    Id=(txt3.get())
    name=(txt4.get())
    ret=0
    if (Id not in data):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                sampleNum=sampleNum+1
                cv2.imwrite("TrainingImage\ "+name +'.'+Id+'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('frame',img) 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>100:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('Details\Details.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
        ret=1
    else:
        res = "User name Already Exists...Try another one!!!"
        message.configure(text= res)
    return ret

def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res="Registration Successful"
    message.configure(text= res)
    return True
    

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids


def reg_submit():
    Userid=txt3.get()
    if Userid.isdigit():
        if TakeImages()==1:
            if TrainImages():
                data[txt3.get()] = txt4.get()
                saving_data(data)
            else:
                pass
    
    else:
        message.configure(text="User Id Should contain number only!!!")
    reg_clear()
    print(data)

submit = tk.Button(window, text="Submit",fg="black",command= login_submit, bg="grey"  ,width=25  ,height=1 ,activebackground = "Red" ,font=('times', 10, ' bold '))
submit.place(x=40, y=300)

clearButton = tk.Button(window, text="Clear",fg="black", command=login_clear,bg="grey"  ,width=25  ,height=1, activebackground = "Red" ,font=('times', 10, ' bold '))
clearButton.place(x=300, y=300)

submit2 = tk.Button(window, text="Submit",fg="black", command=reg_submit, bg="grey"  ,width=25  ,height=1 ,activebackground = "Red" ,font=('times', 10, ' bold '))
submit2.place(x=700, y=300)

clearButton2 = tk.Button(window, text="Clear",command=reg_clear, fg="black"  ,bg="grey"  ,width=25  ,height=1, activebackground = "Red" ,font=('times', 10, ' bold '))
clearButton2.place(x=940, y=300)

quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="black"  ,bg="red"  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1000, y=550)
window.mainloop()
