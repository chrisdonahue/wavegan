from flask import Flask, request, send_from_directory

app = Flask(__name__, static_url_path='')

@app.route('/js/<path:path>')
def send_js(path):
  return send_from_directory('js', path)

@app.route('/img/<path:path>')
def send_img(path):
  return send_from_directory('img', path)

@app.route('/css/<path:path>')
def send_css(path):
  return send_from_directory('css', path)

@app.route('/ckpts/<path:path>')
def send_ckpts(path):
  return send_from_directory('ckpts', path)

@app.route('/')
def root():
  return send_from_directory('', 'index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006)
