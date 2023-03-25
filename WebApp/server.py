from flask import Flask, render_template, request
from blur_img import censor_img, censor_video2
from numpy import fromstring, uint8
import base64
import os

home_screen = "This censor app was built using a retrained Faster RCNN from Pytorch, and it targets human faces and glasses to censor. Colored images work best, and videos take a long time to process."
no_input = "No file was uploaded."
wrong_format = "The *Censor Cam* does not support the file format that was uploaded. Please try a .jpg, .jpeg, .png, .mp4, .webm, or .ogg file."
error = "An error occurred while processing your file. Please try again with another one."

app = Flask(__name__, static_folder='static')


@app.route("/")
def index():
    return render_template("index.html", text=home_screen)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", text=no_input)

    else:
        if file.filename.split(".")[-1] in ("jpeg", "png", "jpg"):

            file = fromstring(file.read(), uint8)
            processed = censor_img(file)
            if type(processed) == str:
                return render_template("index.html", text=error)
            else:
                img_str = base64.b64encode(processed).decode('utf-8')
                return render_template("index.html", file="data:image/jpeg;base64,"+img_str)

        elif file.filename.split(".")[-1] in ("mp4", "webm", "ogg"):
            upload_folder = "./static"
            if not os.path.exists('./static'):
                os.makedirs('./static')
            file.save(os.path.join(upload_folder, file.filename))

            censor_video2(file.filename, os.path.join(upload_folder, file.filename))

            return render_template("index.html", vdo=f"../static/{file.filename.split('.')[0]}_processed.mp4")

        else:
            return render_template("index.html", file=wrong_format)


if __name__ == "__main__":
    app.run(debug=True)