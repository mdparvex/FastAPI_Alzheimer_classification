from PIL import Image
from tensorflow.keras.preprocessing.image import load_img , img_to_array
import io

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']


def model_predict(file , model):
    contents = file.file.read()
    img = load_img(io.BytesIO(contents), target_size=(128, 128))
    img = img_to_array(img)
    img = img.reshape(1 , 128 ,128 ,3)

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(4):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:4]
    
    prob_result = []
    class_result = []
    for i in range(4):
        prob_result.append(round(float(prob[i] * 100), 2)) 
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result