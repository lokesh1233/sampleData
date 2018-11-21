from flask import Flask
from flask_cors import CORS
from Router import chatEntity


app = Flask(__name__)
CORS(app)

chatEntity(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=False, threaded=True)
