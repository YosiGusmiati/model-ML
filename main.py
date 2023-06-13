import os
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import uvicorn


app = FastAPI()

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# This endpoint is for image prediction


@app.post("/predict_image")
async def predict_image(file: UploadFile):
    try:
        # Checking if it's an image
        if file.content_type not in ["image/jpeg", "image/png"]:
            return {"error": "File is not an image"}

        # Preprocess the image
        img = Image.open(file.file)
        img = img.resize((224, 224))
        img = img.convert("RGB")
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output = interpreter.get_tensor(output_details[0]['index'])

        # Define the class labels
        labels = {
            0: 'freshapples',
            1: 'freshbanana',
            2: 'freshoranges',
            3: 'rottenapples',
            4: 'rottenbanana',
            5: 'rottenoranges'
        }

        # Get the predicted class index
        class_index = np.argmax(output[0])
        class_label = labels[class_index]

        result = "This is " + class_label

        return {"result": result}

    except Exception as e:
        return {"error": str(e)}

# Starting the server
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
