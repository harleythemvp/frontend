from flask import Flask, request
from flask_cors import CORS # type: ignore

from machine_learning import grammar_check

app = Flask(__name__)
CORS(app)

@app.route('/get')
def members():
    return({'returned_value':'Message from backend'})

@app.route('/receive_data', methods=['POST'])
def receive_data():
    data_json = request.json
    stringData = data_json.get('input_data')
    corrected_sentence = grammar_check(stringData)
    print('===========================')
    print("Received data:", stringData)
    print('===========================')
    return ({'returned_data':corrected_sentence})

if __name__  == "__main__":
    app.run(host='192.168.43.84', port=3000,debug=True)
    # http://192.168.43.84:3000/get Jeffrey Dahmmer
    # http://172.30.3.74:3000/get
    # http://192.168.182.84:3000/get
    