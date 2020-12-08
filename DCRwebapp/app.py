import base64
import numpy as np
import io
import cv2
from PIL import Image,ImageOps
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.image import img_to_array
from flask import Flask,jsonify, request,render_template

app  = Flask(__name__)


def get_model():
    global model
    model=load_model('./dcr.hdf5')
    global classes,devanagri_classes
    classes = ['ka','kha', 'ga', 'gha', 'kna', 'cha','chha','ja','jha','yna','taamatar','thaa',
 'daa','dhaa','adna','tabala','tha','da','dha','na','pa','pha','ba','bha','ma','yaw','ra',
 'la','waw','motosaw','petchiryakha','patalosaw','ha','chhya','tra','gya','0','1','2',
 '3','4','5','6','7','8','9']
    devanagri_classes=['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड',
 'ढ','ण','त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श','ष',
 'स','ह','क्ष','त्र','ज्ञ','०','१','२','३','४','५','६','७','८','९']
    print(" * Model loaded successfully ")

def preprocess_image(image,target_size=(32,32)):
    image=ImageOps.grayscale(image)
    image=image.resize(target_size)
    img=np.array(image)
    img=img.reshape(1,32,32,1)
    img=img/255.0
    img=1-img

    return img

def borders(here_img, thresh, bthresh=0.092):
    shape = here_img.shape
    check= int(bthresh*shape[0])
    image = here_img[:]
    top, bottom = 0, shape[0] - 1
    bg = np.repeat(thresh, shape[1])
    count = 0
    for row in range(1, shape[0]):
        if  (np.equal(bg, image[row]).any()) == True:
            count += 1
        else:
            count = 0
        if count >= check:
            top = row - check
            break
    bg = np.repeat(thresh, shape[1])
    count = 0
    rows = np.arange(1, shape[0])
    for row in rows[::-1]:
        if  (np.equal(bg, image[row]).any()) == True:
            count += 1
        else:
            count = 0
        if count >= check:
            bottom = row + count
            break
    d1 = (top - 2) >= 0
    d2 = (bottom + 2) < shape[0]
    d = d1 and d2
    if(d):
        b = 2
    else:
        b = 0

    return (top, bottom, b)


def preprocess(bgr_img):#gray image
    blur = cv2.GaussianBlur(bgr_img,(5,5),0)
    ret,th_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    rows, cols = th_img.shape
    bg_test = np.array([th_img[i][i] for i in range(5)])
    if bg_test.all() == 0:
        text_color = 255
    else:
        text_color = 0

    tb = borders(th_img, text_color)
    lr = borders(th_img.T, text_color)
    dummy = int(np.average((tb[2], lr[2]))) + 2
    template = th_img[tb[0]+dummy:tb[1]-dummy, lr[0]+dummy:lr[1]-dummy]
    return (template, tb, lr)

def segmentation(bordered, thresh=255, min_seg=10, scheck=0.25):
    try:
        shape = bordered.shape
        check = int(scheck * shape[0])
        image = bordered[:]
        image = image[check:].T
        shape = image.shape
        bg = np.repeat(255 - thresh, shape[1])
        bg_keys = []
        for row in range(1, shape[0]):
            if  (np.equal(bg, image[row]).all()):
                bg_keys.append(row)

        lenkeys = len(bg_keys)-1
        new_keys = [bg_keys[1], bg_keys[-1]]
        for i in range(1, lenkeys):
            if (bg_keys[i+1] - bg_keys[i]) > check:
                new_keys.append(bg_keys[i])
        new_keys = sorted(new_keys)
        segmented_templates = []
        first = 0
        bounding_boxes = []
        for key in new_keys[1:]:
            segment = bordered.T[first:key]
            if segment.shape[0]>=min_seg and segment.shape[1]>=min_seg:
                segmented_templates.append(segment.T)
                bounding_boxes.append((first, key))
            first = key
        last_segment = bordered.T[new_keys[-1]:]
        if last_segment.shape[0]>=min_seg and last_segment.shape[1]>=min_seg:
            segmented_templates.append(last_segment.T)
            bounding_boxes.append((new_keys[-1], new_keys[-1]+last_segment.shape[0]))


        return(segmented_templates, bounding_boxes)
    except:
        return [bordered, (0, bordered.shape[1])]

def classifier(img):
    x = np.asarray(img, dtype = np.float32).reshape(1, 32, 32, 1) / 255.0
    output = model.predict(x)
    output = output.reshape(46)
    predicted = np.argmax(output)
    devanagari_label = devanagri_classes[predicted]
    success = output[predicted] * 100
    return devanagari_label, success

def generate_sentence(segments):
    pred_lbl = ""
    acc = []
    for segment in segments:
        segment = cv2.resize(segment, (32, 32))
        segment = cv2.GaussianBlur(segment, (3, 3), 0)
        segment = cv2.erode(segment, (3, 3), 1)
        lbl, a = classifier(segment)
        pred_lbl+=lbl
        acc.append(a)
    return pred_lbl, np.array(acc).mean()

print(" * Loading Keras model.....")
get_model()


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST','GET'])
def predict():
    if request.method=='POST':
        message=request.get_json(force=True)
        encoded=message['image']
        decoded=base64.b64decode(encoded)
        nparr=np.fromstring(decoded,np.uint8)
        img=cv2.imdecode(nparr,cv2.IMREAD_GRAYSCALE)
        #image=Image.open(io.BytesIO(decoded))
        #processed_image=preprocess_image(image)
        #prediction=model.predict(processed_image)
        #answer=np.argmax(prediction)
        prepimg, tb, lr = preprocess(img)
        segments=segmentation(prepimg)
        ans,accuracy=generate_sentence(segments[0])
        response = {
            'predict' : {
                'answer' : ans,
                'probability' :str(accuracy),
            }
        }
        print(ans)
        return jsonify(response)
    return jsonify(response={'answer':'Image Upload'})

if __name__=="__main__":
    app.run()
