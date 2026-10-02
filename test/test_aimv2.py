from transformers import AutoModel

print("Loading model...")

model = AutoModel.from_pretrained(
    "apple/aimv2-large-patch14-224",
    revision="ac764a25c832c7dc5e11871daa588e98e3cdbfb7",
    trust_remote_code=True,
)

print("Loaded successfully!")

print(type(model))
print(model.config)
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained(
    "apple/aimv2-large-patch14-224",
    revision="ac764a25c832c7dc5e11871daa588e98e3cdbfb7",
)

model = AutoModel.from_pretrained(
    "apple/aimv2-large-patch14-224",
    revision="ac764a25c832c7dc5e11871daa588e98e3cdbfb7",
    trust_remote_code=True,
)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)

print(type(outputs))
print(outputs.keys())
print(outputs.last_hidden_state.shape)
print(model.config.hidden_size)
print(model.config.embed_dim)