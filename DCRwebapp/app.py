import base64
import numpy as np
import io
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
    global classes,devanagri_class
    classes = ['ka','kha', 'ga', 'gha', 'kna', 'cha','chha','ja','jha','yna','taamatar','thaa',
 'daa','dhaa','adna','tabala','tha','da','dha','na','pa','pha','ba','bha','ma','yaw','ra',
 'la','waw','motosaw','petchiryakha','patalosaw','ha','chhya','tra','gya','0','1','2',
 '3','4','5','6','7','8','9']
    devanagri_class=[' क',' ख',' ग',' घ',' ङ',' च',' छ',' ज',' झ',' ञ',' ट',' ठ',' ड',
 ' ढ',' ण',' त',' थ',' द',' ध',' न',' प',' फ',' ब',' भ',' म',' य',' र',' ल',' व',' श',' ष',
 ' स',' ह',' क्ष',' त्र',' ज्ञ',' ०',' १',' २',' ३',' ४',' ५',' ६',' ७',' ८',' ९']
    print(" * Model loaded successfully ")

def preprocess_image(image,target_size=(32,32)):
    image=ImageOps.grayscale(image)
    image=image.resize(target_size)
    img=np.array(image)
    img=img.reshape(1,32,32,1)
    img=img/255.0
    img=1-img

    return img

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
        image=Image.open(io.BytesIO(decoded))
        processed_image=preprocess_image(image)

        prediction=model.predict(processed_image)
        answer=np.argmax(prediction)
        response = {
            'predict' : {
                'answer' : classes[answer],
                'probability' :str( prediction[0][answer]*100),
                'devanagri' : devanagri_class[answer] 
            }
        }
        print(prediction[0][answer])
        return jsonify(response)
    return jsonify(response={'answer':'Image Upload'})

if __name__=="__main__":
    app.run()
