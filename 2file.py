import pyttsx3
from math import *
from imutils.video import VideoStream
import imutils
import time
import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import os
import shutil
import threading
from time import strftime
import tkinter.font as tkFont
import pickle
from embedding_extract import embedding
from train_model import create_model
from tkinter import ttk
import numpy as np


class RecognizePanel:
    def __init__(self, frame2, frame_c):
        self.helv18 = tkFont.Font(family="Helvetica", size=12, weight="bold")
        self.helv16 = tkFont.Font(family="Helvetica", size=10, weight="bold")
        self.frame2 = frame2
        self.frame_c = frame_c

        self.time_label = Label(self.frame2, font=self.helv16)
        self.date_label = Label(self.frame2, font=self.helv16)
        self.time_label.grid(row=0, column=0, sticky="W", columnspan=2, padx=5, pady=5)
        self.date_label.grid(row=1, column=0, sticky="W", padx=5, pady=5)

        self.button_before = Button(self.frame2, text="BEFORE", command=self.change_to_frame1)
        self.button_before.grid(row=2, column=1, sticky="E", padx=10)

        self.my_time()
        self.frame2.pack(fill=BOTH, expand=True)

    def my_time(self):
        time_string_time = strftime('%I:%M:%S %p')  # time format
        time_string_date = strftime("%x %A")
        self.time_label.config(text=time_string_time)
        self.date_label.config(text=time_string_date)
        self.time_label.after(1000, self.my_time)

    def change_to_frame1(self):
        bv1.set(1)
        self.frame_c.pack(fill=BOTH, expand=True)
        self.frame2.pack_forget()


class GetDataPanel:
    def __init__(self, frame1, frame_c):
        self.my_list = os.listdir('./facialdataset')
        self.frame1 = frame1
        self.frame_c = frame_c
        # list boxe thu muc trong "facialdataset"

        self.frame1.columnconfigure(0, weight=4)
        self.frame1.columnconfigure(1, weight=1)
        self.frame1.columnconfigure(2, weight=1)

        # nhap ten thu muc, tao thu muc moi trong folder  "facialdataset"
        self.newFolder = StringVar()
        self.path_entry = Entry(self.frame1, textvariable=self.newFolder).grid(column=0, row=0, pady=10, padx=10,
                                                                               stick="new")

        self.listboxframe = Frame(self.frame1)
        self.listboxframe.grid(column=0, row=1, stick="new", padx=10)

        self.list_items = StringVar(value=self.my_list)
        self.listbox = Listbox(self.listboxframe, relief=RIDGE, activestyle='underline', listvariable=self.list_items,
                               selectmode=SINGLE)
        self.listbox.pack(side=LEFT, fill=BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.chooseItem)

        self.yScroll = Scrollbar(self.listboxframe, orient=VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=self.yScroll.set)
        self.yScroll.pack(side=RIGHT, fill=Y)

        # button de tao folder, new label
        self.add_btn = Button(self.frame1, text="Add", command=self.click_add)
        self.add_btn.grid(column=1, row=0, padx=5, pady=10, stick="new")

        # button delete
        self.delete_btn = Button(self.frame1, text="Delete", command=self.click_delete)
        self.delete_btn.grid(row=0, column=2, stick="ew", padx=5, pady=10)

        # button de chup, nhan giu de lay du lieu
        self.hold_btn = Button(self.frame1, text="Hold")
        self.hold_btn.grid(column=1, row=1, padx=5, stick="new")
        self.hold_btn.bind('<ButtonPress-1>', self.start_click)
        self.hold_btn.bind('<ButtonRelease-1>', self.start_threading)

        self.update_btn = Button(self.frame1, text="Update", command=self.update_click)
        self.update_btn.grid(column=2, row=1, padx=5, stick="new")

        self.my_string_var = StringVar()
        # label choosed item
        self.item_label = Label(self.frame1, textvariable=self.my_string_var).grid(column=0, row=2, padx=10, pady=5,
                                                                                   stick="w")
        self.button_back = Button(self.frame1, text="BACK", command=self.change_to_frame2).grid(column=1, row=3,
                                                                                                padx=10, pady=5,
                                                                                                stick="EW")
        self.button_back = Button(self.frame1, text="TRAIN", command=self.training_thread).grid(column=2, row=3,
                                                                                                padx=10, pady=5,
                                                                                                stick="EW")
        self.progress_trainning = ttk.Progressbar(self.frame1, orient='horizontal', mode='indeterminate')

    def training_thread(self):
        thread_train = threading.Thread(target=self.trainning)
        thread_train.start()

    def trainning(self):
        self.progress_trainning.grid(column=0, row=3, stick="NSEW", padx=5, pady=5)
        self.progress_trainning.start()
        starttime = time.perf_counter()
        embedding()
        create_model()
        self.my_string_var.set("finished in {0:.2f} seconds".format(time.perf_counter() - starttime))
        self.progress_trainning.stop()
        self.progress_trainning.grid_forget()
        recognizer = pickle.loads(open('./output/best_model.pickle', 'rb').read())

    def change_to_frame2(self):
        bv1.set(0)
        self.frame_c.pack(fill='both', expand=1)
        self.frame1.pack_forget()

    def chooseItem(self, event):
        # get selected list box color item
        folder = self.listbox.get(ANCHOR)
        # print(dataset_path.substitute({'x':folder}))
        dataset_path = os.path.join("./facialdataset\\", folder)
        self.my_string_var.set(folder)

    def click_add(self, ):
        if not self.newFolder.get():
            self.my_string_var.set("S.O.S ! nhập tên folder")
        else:
            path = os.path.join("./facialdataset\\", self.newFolder.get())
            if not os.path.exists(path):
                self.listbox.insert(END, self.newFolder.get())
                os.mkdir(path)
                # listbox.activate(END)
            else:
                self.my_string_var.set("folder exists")

    def click_delete(self):
        folder = self.listbox.get(ANCHOR)
        if folder:
            path = os.path.join("./facialdataset\\", folder)
            if not os.path.exists(path):
                self.my_string_var.set("folder does not exist")
            else:
                try:
                    shutil.rmtree(path)
                    self.listbox.delete(ANCHOR)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
                self.my_string_var.set("Detete is successful")

    def start_click(self, event):
        self.my_string_var.set("Starting....")

    def click_hold(self):
        initial_count = 0
        folder = self.listbox.get(ANCHOR)
        path = r"./facialdataset/{}/".format(folder)
        if os.path.exists(path):
            for s in os.listdir(path):
                if os.path.isfile(os.path.join(path, s)):
                    initial_count += 1
            for i in range(initial_count, initial_count + 5):
                p = os.path.join(path, "{}_{}.jpg".format(folder, i))
                print(p)
                cv2.imwrite(p, orig)
                time.sleep(0.8)
            self.my_string_var.set("DONE....")
        else:
            self.listbox.delete(ANCHOR)
            self.my_string_var.set("Path is not existing")

    def start_threading(self, event):
        thread2 = threading.Thread(target=self.click_hold)
        thread2.start()

    def update_click(self):
        self.my_list = os.listdir('./facialdataset')
        self.list_items.set(value=self.my_list)


def videoLoop():

    # net = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)

    try:
        # keep looping over frames until we are instructed to stop
        # sự kiện ngưng việc hiển thị frame từ camera
        while not stopEvent.is_set():
            global orig
            frame = vs.read()
            orig = frame.copy()
            # frame = imutils.resize(frame, width=500,high=400)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img2 = cv2.flip(cv2image, 1)
            frame = imutils.resize(img2, width=int(window_width / 1.7), height=int(
                window_height / 1.7))  # width=int(window_width / 1.7), height=int(window_width / 1.7)
            origctv = cv2image.copy()
            h, w = frame.shape[:2]

            radius_center = 100
            thickness = 2
            center_frame_x = w / 2
            center_frame_y = h / 2

            # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            # cv2.circle(frame, (int(center_frame_x), int(center_frame_y)),radius_center , (226, 0,0), thickness)
            cv2.circle(frame, (int(center_frame_x), int(center_frame_y)), radius_center, (113, 190, 35), 8)
            # (1 , 1, 200 ,7)
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.8:
                    continue

                # Determine the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # box = detections[0, 0, i, 3:7] * np.array([140, 160, 140, 160])
                (startX, startY, endX, endY) = box.astype("int")

                dx = max(abs(center_frame_x - endX), abs(center_frame_x - startX))
                dy = max(abs(center_frame_y - endY), abs(center_frame_y - startY))
                length = int(sqrt(pow(dx, 2) + pow(dy, 2)))

                if length < radius_center + 2.5:
                    cv2.circle(frame, (int(center_frame_x), int(center_frame_y)), radius_center, (255, 0, 128), 8)
                    if bv1.get():
                        text = "{:.2f}%".format(confidence * 100)
                    else:
                        face = frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]
                        if fW < 20 or fH < 20:
                            continue
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True,
                                                         crop=False)
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()  # (1,128)
                        preds = recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = le.classes_[j]
                        # text = "{}: {:.2f}%".format(name, proba * 100)
                        text = "{}".format(name)

                        if proba * 100 > 90:
                            popup(origctv, name,pop)

            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)
            panel.configure(image=image)
            panel.image = image
    except:
        print("[INFO] caught a RuntimeError")


def popup(myface, name, pop):
    def click_yes():
        pass
    def click_no():
        pass

    list = os.listdir('./facialdataset')
    list2 = ["Lâm Phúc Hải", "Nguyễn Thế Phát", "Quang Huy"]
    dict_from_list = dict(zip(list, list2))

    global me
    myface = imutils.resize(myface, width=200, height=200)
    image = Image.fromarray(myface)
    me = ImageTk.PhotoImage(image)

    if pop is None or Toplevel.winfo_ismapped(pop) == False:
        pass
    else:
        pop.destroy()

    pop = Toplevel(root)
    pop.resizable(0, 0)
    pop.grab_set()
    pop.geometry("400x200")  # width x height
    pop.columnconfigure(0, weight=1)
    pop.columnconfigure(1, weight=5)
    pop.columnconfigure(2, weight=5)
    pop.rowconfigure(0, weight=1)
    pop.rowconfigure(1, weight=1)
    pop.rowconfigure(2, weight=1)
    pop.rowconfigure(3, weight=1)

    image_label = Label(pop, image=me)
    name_label = Label(pop, text=name, font=("helvetica", 16))
    yes_btn = Button(pop, text="YES", font=("helvetica", 12), command=click_yes)
    no_btn = Button(pop, text="NO", font=("helvetica", 12), command=click_no)

    image_label.grid(row=0, column=0, sticky="EWNS", rowspan=4)
    name_label.grid(row=0, column=1, columnspan=2)
    yes_btn.grid(row=3, column=2, sticky="EW", padx=5)
    no_btn.grid(row=3, column=1, sticky="EW", padx=5)

    engine.say("Chào {}".format(dict_from_list[name]))
    engine.runAndWait()

    # pop.after(2000, pop.destroy)

caffeModel = "./res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "./deploy.prototxt.txt"
embedder = cv2.dnn.readNetFromTorch("./openface_nn4.small2.v1.t7")
net = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)
engine = pyttsx3.init()
voices = engine.getProperty("voices")
rate = engine.getProperty("rate")
engine.setProperty("voice", voices[1].id)
engine.setProperty("rate", 90)

# load the actual face recognition model along with the label encoder
global recognizer
global pop
pop = None
recognizer = pickle.loads(open('./output/best_model.pickle', 'rb').read())
le = pickle.loads(open("./output/le.pickle", "rb").read())  # args["le"]
root = tk.Tk()
root.title("Collecting data")
bv1 = tk.BooleanVar(root)  # declare Boolean Variable
bv1.set(0)

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

window_width = int(screen_width / 1.9)
window_height = int(screen_height / 2.5)

leftTopX = int((screen_width / 2) - (window_width / 2))
leftTopY = int((screen_height / 2) - (window_height / 2))

root.geometry(f"{window_width}x{window_height}+{leftTopX}+{leftTopY}")
root.resizable(0, 0)

vs = VideoStream(src=0).start()
time.sleep(2.0)

frame1 = Frame(root)
greet = Frame(root)  # frame1
order = Frame(root)  # frame2

panel = Label(frame1)
panel.pack(fill=BOTH)
# start a thread that constantly pools the video sensor for
# the most recently read frame
cond = threading.Condition()
stopEvent = threading.Event()
thread = threading.Thread(target=videoLoop)
thread.start()
frame1.pack(side=LEFT)

GetDataPanel(greet, order)
RecognizePanel(order, greet)

root.mainloop()

# y = startY - 10 if startY - 10 > 10 else startY + 10
# cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
