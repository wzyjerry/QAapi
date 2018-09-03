import json
import time
from service.ner import *
from service.classify import *
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'QA-API'


tag = {
    0: '搜学者',
    1: '搜文章',
    2: '搜会议'
}


@app.route('/query/<text>')
def query(text):
    st = time.time()
    ret = ner(text)
    ret['intent': tag[classify(text)]]
    print('Parse time:', ed - st)
    return json.dumps(ret, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9507, debug=True)
