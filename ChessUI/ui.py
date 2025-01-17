from flask import Flask

htmlContent = open('index.html', 'r').read()
vueContent = open('vue.global.prod.min.js', "r").read()

app = Flask(__name__)

@app.route('/')
def hello_world():
    return htmlContent

@app.route('/<path:path>')
def static_file(path):
    print(path)
    return open('./' + path, 'rb').read()

if __name__ == '__main__':
    app.run(debug=True)
