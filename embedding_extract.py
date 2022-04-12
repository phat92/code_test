from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os


def embedding():
    modelPath = r"./res10_300x300_ssd_iter_140000.caffemodel"
    protoPath = r"./deploy.prototxt.txt"

    # nhan dien mat nguoi
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")

    # Torch-based and is responsible for extracting facial embeddings via deep learning feature extraction
    # trich xuat embedding cua 1 face

    embedder = cv2.dnn.readNetFromTorch(r"./openface_nn4.small2.v1.t7")

    # load dataset de facial detector va trich xuat dac tinh qua embedding

    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images("./facialdataset/"))
    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownEmbeddings = []
    knownNames = []
    # initialize the total number of faces processed
    total = 0

    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        # name = imagePath.split(os.path.sep)[-2]
        name = os.path.split(imagePath)[0]
        name = os.path.split(name)[1]
        print(name)
        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                          swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # quet mang faces duoc detection
        # neu phat hien guong mat thì crop mat
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > 0.9:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]  # width height cua faces
                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue
                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                # trich xuat dac trung 128-d feature vector
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("./output/embedding.pickle", "wb")  # args["embeddings"]
    f.write(pickle.dumps(data))
    f.close()


"""
FILE DUNG DE TRICH XUAT 
Compute 128-d face embeddings to quantify a face
NEXT STEP : Train a Support Vector Machine (SVM) on top of the embeddings
"""
# embedding()
