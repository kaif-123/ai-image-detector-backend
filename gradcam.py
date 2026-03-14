import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from file import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("ai_vs_real_detector.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image = Image.open("Figure2.png").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Hook to capture gradients
gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# hook last conv layer
model.conv2.register_forward_hook(forward_hook)
model.conv2.register_backward_hook(backward_hook)

output = model(input_tensor)
pred_class = output.argmax()

model.zero_grad()
output[0, pred_class].backward()

grad = gradients[0].detach().numpy()[0]
act = activations[0].detach().numpy()[0]

weights = np.mean(grad, axis=(1,2))
cam = np.zeros(act.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * act[i]

cam = np.maximum(cam, 0)
cam = cam / cam.max()
cam = cv2.resize(cam, (224,224))

img = cv2.imread("Figure2.png")
img = cv2.resize(img,(224,224))

heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
result = heatmap * 0.4 + img

plt.imshow(cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()