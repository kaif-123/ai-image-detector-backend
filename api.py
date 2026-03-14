from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
from torchvision import transforms
from file import SimpleCNN
import torch.nn.functional as F

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")

model = SimpleCNN()
model.load_state_dict(torch.load("ai_vs_real_detector.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probabilities = F.softmax(output, dim=1)

        confidence = round(probabilities.max().item() * 100)

        predicted = probabilities.argmax(dim=1)

    if predicted.item() == 0:
        label = "Real Image"
    else:
        label = "AI Generated"

    return {
        "prediction": label,
        "confidence": confidence
    }