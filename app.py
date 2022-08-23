import tkinter as tk
from PIL import Image,ImageTk
import cv2,os
import csv
import numpy as np
import pandas as pd
import time
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder
import pickle

window=tk.Tk()
window.title('Fare recognition')
window.geometry("1400x900")
load=Image.open('images/7.jpg')
render= ImageTk.PhotoImage(load)
img=tk.Label(window, image = render)
img.place(x=0, y=0)

x_cord = 75;
y_cord = 20;
checker=0;

message = tk.Label(window, text="FACIAL RECOGNITION USING ML AND OPENCV BASED ON PYTHON" ,bg="black"  ,fg="white"  ,width=63  ,height=1,font=('Times New Roman', 28, 'bold underline'))
message.place(x=0, y=95)
lbl = tk.Label(window, text="Enter Your Name",width=15  ,fg="white",bg="#c95b06"  ,font=('Times New Roman', 20, ' bold ') )
lbl.place(x=360-x_cord, y=250-y_cord)
txt = tk.Entry(window,width=33,bg="#d6ad65", fg="black",font=('Times New Roman', 15, ' bold '))
txt.place(x=320-x_cord, y=300-y_cord)
lbl3 = tk.Label(window, text="NOTIFICATION",width=15  ,fg="white",bg="#c95b06" ,font=('Times New Roman', 20, ' bold '))
lbl3.place(x=815-x_cord, y=250-y_cord)
message = tk.Label(window, text=""  ,bg="#d6ad65", fg="black"  ,width=30  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold '))
message.place(x=755-x_cord, y=300-y_cord)
lbl4 = tk.Label(window, text="STEP 1", width=12, fg="white",bg="#c95b06",  font=('Times New Roman', 20, ' bold '))
lbl4.place(x=280 - x_cord, y=420 - y_cord)

lbl5 = tk.Label(window, text="STEP 2", width=12, fg="white",bg="#c95b06",font=('Times New Roman', 20, ' bold '))
lbl5.place(x=660 - x_cord, y=420 - y_cord)

lbl6 = tk.Label(window, text="STEP 3", width=12, fg="white",bg="#c95b06",font=('Times New Roman', 20, ' bold '))
lbl6.place(x=1060 - x_cord, y=420 - y_cord)


def clear1():
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)


def TakeImages():
    Id = (txt.get())
    # name = (txt2.get())
    if not Id:
        res = "Please enter Name"
        message.configure(text=res)
        MsgBox = tk.messagebox.askquestion("Warning", "Please enter your name properly , press yes if you understood",
                                           icon='warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo('Your need', 'Please go through the readme file properly')

    elif (Id.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = r"haarcascade\haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage/ " + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                # display the frame

            else:
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 200:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for First Name : " + Id
        row = [Id]
        with open(r'person_details\person_details.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
        tk.messagebox.showinfo('Completed', 'Captured images successfully!!')
    else:
        if (Id.isalpha()):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
def TrainImages():
    le = LabelEncoder()
    faces, Id = getImagesAndLabels("TrainingImage")
    Id=le.fit_transform(Id)
    output = open('label_encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(Id))
    recognizer.save(r"model\Trainner.yml")
    res = "Image Trained"
    clear1();
    # clear2();
    message.configure(text=res)
    tk.messagebox.showinfo('Completed', 'Your model has been trained successfully!!')
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = str(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read(r"model\Trainner.yml")
    harcascadePath = r"haarcascade\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df = pd.read_csv(r"person_details\person_details.csv")
    cam = cv2.VideoCapture(0)
    print(df)

    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['First Name', 'Last Name']
    pkl_file = open('label_encoder.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            # print(conf)
            if (conf < 50):
                tt = le.inverse_transform([Id])
                # aa = df.loc[df['Id'] == Id]['Name'].values
                # tt = str(Id) + "-" + aa
                print(tt)
                tt = tt[0]
            else:
                Id = 'Unknown'
                tt = str(Id)
                # print(tt)
            if (conf > 75):
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite(r"ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()
    # res = attendance
    # message.configure(text=res)
    res = "face recognized"
    message.configure(text=res)
    tk.messagebox.showinfo('Completed', 'Congratulations ! Your face successfully detected!!')


def quit_window():
    MsgBox = tk.messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application',
                                       icon='warning')
    if MsgBox == 'yes':
        tk.messagebox.showinfo("Greetings", "Thank You very much for using our software. Have a nice day ahead!!")
        window.destroy()


takeImg = tk.Button(window, text="IMAGE CAPTURE BUTTON", command=TakeImages, fg="white", bg="#666617", width=25, height=2,
                    activebackground="pink", font=('Times New Roman', 15, ' bold '))
takeImg.place(x=215 - x_cord, y=475 - y_cord)
trainImg = tk.Button(window, text="MODEL TRAINING BUTTON", command=TrainImages, fg="white",bg="#666617", width=25,
                     height=2, activebackground="pink", font=('Times New Roman', 15, ' bold '))
trainImg.place(x=615 - x_cord, y=475 - y_cord)
trackImg = tk.Button(window, text="RECOGNIZE FACE", command=TrackImages, fg="white", bg="#666617", width=25,
                     height=2, activebackground="pink", font=('Times New Roman', 15, ' bold '))
trackImg.place(x=1000 - x_cord, y=475 - y_cord)
quitWindow = tk.Button(window, text="QUIT", command=quit_window, fg="white", bg="#3d2e0d", width=10, height=2,
                       activebackground="pink", font=('Times New Roman', 15, ' bold '))
quitWindow.place(x=650, y=640 - y_cord)

window.mainloop()