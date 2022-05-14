import io
from base64 import b64encode
from flask import Flask, redirect, render_template, request
from flask_ngrok import run_with_ngrok

from inference import inference_image

app = Flask(__name__)
run_with_ngrok(app)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files.get("file")
        if not file:
            return
        img_bytes = file.read()
        processed_img = inference_image(img_bytes)

        file_object = io.BytesIO()
        processed_img.save(file_object, "PNG")
        base64img = "data:image/png;base64," + b64encode(file_object.getvalue()).decode(
            "ascii"
        )

        return render_template("result.html", result_img=base64img)

    return render_template("index.html")


if __name__ == "__main__":
    app.run()
