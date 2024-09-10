from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

app = FastAPI()

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)

model = MNISTModel()
model.load_state_dict(torch.load('mnist.pth', map_location=torch.device('cpu')), strict=False)

model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(file: UploadFile) -> torch.Tensor:
    image = Image.open(BytesIO(file.file.read()))  
    image = transform(image).unsqueeze(0)  
    return image

@app.post("/detect_digit/")
async def detect_digit(file: UploadFile = File(...)):
    image = preprocess_image(file)

    with torch.no_grad():
        output = model(image)
        predicted_digit = output.argmax(1).item() 

    return {"detected_digit": predicted_digit}

