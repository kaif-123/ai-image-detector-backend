import torch
from PIL import Image
from torchvision import transforms
from file import SimpleCNN   # tumhara model class

device = torch.device("cpu")

model = SimpleCNN()
model.load_state_dict(torch.load("ai_vs_real_detector.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image = Image.open("Figure2.png").convert("RGB")

img = transform(image)
img = img.unsqueeze(0)

with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output,1)

if predicted.item()==0:
    print("Real Image")
else:
    print("AI Generated Image")