from flask import Flask, render_template, request
import os
from predict_emotion import predict_emotion  # your existing function

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'audiofile' in request.files:
            file = request.files['audiofile']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = predict_emotion(filepath)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)



