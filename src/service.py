from flask import Flask, request, render_template, send_file
from flask_uploads import UploadSet, IMAGES, configure_uploads
from werkzeug.utils import secure_filename
from shutil import copyfile
import requests
import json
import uuid
import time
import threading

app = Flask(__name__)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/uploads'
app.config['UPLOADED_PHOTOS_ALLOW'] = IMAGES
photos = UploadSet('photos', IMAGES)

configure_uploads(app, photos)

NAMESPACE_SERVICE = uuid.uuid3(uuid.NAMESPACE_DNS, "service")

missions = {}

def start(filename, id, timestamp):
    res = requests.post("http://127.0.0.1:2023/segmentation", data={ "origin": filename })
    d = res.json()
    if d["status"] != 1200:
        print("Error when start a segmentation mission: %s" % d["message"])
        return
    mission_id = d["id"]
    while True:
        time.sleep(5)
        res = requests.post("http://127.0.0.1:2023/query", data={ "id": mission_id })
        d = res.json()
        if d["status"] != 1200:
            print("Error when query a segmentation mission: %s" % d["message"])
            return
        if d["finished"]:
            break
    
    input_filename = "static/segment/classes/%s_no" % mission_id
    mask_filename = "static/segment/classes/%s_mask" % mission_id
    res = requests.post("http://127.0.0.1:2022/anonymize", data={ "origin": filename, "input": input_filename, "mask": mask_filename })
    d = res.json()
    mission_id = d["id"]
    while True:
        time.sleep(5)
        res = requests.post("http://127.0.0.1:2022/query", data={ "id": mission_id })
        d = res.json()
        if d["status"] != 1200:
            print("Error when query a segmentation mission: %s" % d["message"])
            return
        if d["finished"]:
            break
    
    copyfile("static/inpaint/%s.png" % mission_id, "static/results/%s.png" % id)
    missions[id] = timestamp


@app.route("/", methods=["GET"])
def ui():
    return render_template("main-ui.html")

@app.route("/main", methods=["POST"])
def main():
    f = request.files["file"]
    filename = secure_filename(f.filename)
    photos.save(f, name=filename)

    filename = "static/uploads/" + filename
    res = { "status": 1200, "src": filename, "id": str(uuid.uuid3(NAMESPACE_SERVICE, filename)), "time": int(time.time()) }

    t = threading.Thread(target = start, args = (filename, res["id"], res["time"]))
    t.setDaemon(True)
    t.start()
    
    return json.dumps(res)

@app.route("/query", methods=["POST"])
def query():
    id = request.form["id"]
    if id in missions:
        if time.time() - missions[id] > 86400:
            return json.dumps({ "status": 1402, "message": "Not a valid time." })
        return json.dumps({ "status": 1200, "finished": True })
    return json.dumps({ "status": 1200, "finished": False })

if __name__ == "__main__":
    app.run(port=1145, host="127.0.0.1", debug=True, use_reloader=False)