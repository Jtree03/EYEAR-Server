import json

from flask import Flask, request
from werkzeug.utils import secure_filename

import sound_analysis

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'This is EYEAR API Server.'


@app.route('/learning/data', methods=['POST'])
def analyze_sound():
    file = request.files['file']
    file.save(secure_filename(file.filename))
    answer = sound_analysis.analyze(file.filename)

    return json.dumps(str(answer))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000')
