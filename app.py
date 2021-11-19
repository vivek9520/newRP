import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import pickle
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
from datetime import datetime
import requests



from scipy.spatial import distance

from sklearn.decomposition import PCA


import tensorflow as tf
print(tf.__version__)

# model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)



# model = pickle.load(open("RPmodel.pkl","rb"))
images_pickle = pickle.load(open("images.pkl","rb"))
features_pickle = pickle.load(open("feature.pkl","rb"))
pca_feature_pickle = pickle.load(open("pca_features.pkl","rb"))




def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
# feat_extractor.summary()




def search():
    new_image, x = load_image("x.jpeg")
    features = np.array(features_pickle)
    pca = PCA(n_components=300)
    pca.fit(features)
# pca_features = pca.transform(features)
#   # project it into pca space

    new_features = feat_extractor.predict(x)
    new_pca_features = pca.transform(new_features)[0]

#   # calculate its distance to all the other images pca feature vectors
    distances = [ distance.cosine(new_pca_features, feat) for feat in pca_feature_pickle ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:5]  # grab first 5
# results_image = get_concatenated_images(idx_closest, 200)
    imageName=[]
    for i in idx_closest:
        imageName.append(os.path.basename(images_pickle[i]))
  
 
    print(imageName)
    return (imageName)
# print(idx_closest)


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './static/uploaded/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return "welcome"



@app.route("/upload", methods=["POST"])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save("x.jpeg")

    x = search()
    result=[]
    for i in x:
        m =i.split()[0]
        result.append(m)
        print(m)
    result = list(dict.fromkeys(result))

    res = requests.get("http://localhost:2000/api/v1/shops/"+result[0] )
    
    # for k in result:
    #     res = requests.get("http://localhost:4000/api/v1/shops/"+k, verify=False)
    #     print(res.json())


    return jsonify(res.json())

if __name__ == "__main__":
    app.run(debug=True)