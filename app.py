from beam import App, Runtime, Image, Volume
from inferences.swinsr import swinsr

# Beam Volume to store cached models
CACHE_PATH = "./cached_models"

app = App(
    name="inference-quickstart",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        gpu="T4",
        image=Image(
            python_version="python3.9",
            python_packages="requirements.txt"
        ),
    ),
    # Storage Volume for model weights
    volumes=[Volume(name="cached_models", path=CACHE_PATH)],
)


# This function runs once when the container boots
def load_models():
    load, _ = swinsr()
    return load(CACHE_PATH)


# Rest API initialized with loader
@app.rest_api(loader=load_models)
def predict(**inputs):
    print("inputs", inputs)
    _, infer = swinsr()

    image_base64 = inputs["image"]
    model, processor = inputs["context"]
    processed_image_base64 = infer(image_base64, model, processor)
    print(processed_image_base64)

    return { "image": processed_image_base64 }
