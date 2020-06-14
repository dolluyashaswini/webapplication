from flask import *
from PIL import Image
from torchvision.utils import save_image
import mlengine
from mlengine import load_model,load_transforms,predict
app=Flask(__name__)
model=load_model()
preprocess=load_transforms()
@app.route('/')
def home():
	return render_template('index.html')
@app.route('/upload',methods=['GET','POST']) 
def upload():
	if request.method=='POST':
		img_file=request.files['image']
		img=Image.open(img_file)
		output=predict(model,preprocess,img)
		save_image(output,"./static/output.png")
		return render_template('result.html')
	else:
		return render_template('index.html')
if __name__=='__main__':
    app.run(debug=True)