import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# ---- Define CNN Model ----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self._to_linear = None
        self.convs(torch.randn(1, 3, 224, 224))
        self.fc1 = nn.Linear(self._to_linear, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self._to_linear is None:
            self._to_linear = x.view(-1).shape[0]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---- Load Model ----
model_path = "model.pth"
net = Net()
net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ---- Image Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ---- Prediction Function ----
def predict_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = net(image)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
    return {
        "prediction": classes[predicted.item()],
        "confidence": round(probs[0][predicted.item()].item(), 4),
        "is_tumor": classes[predicted.item()] != "notumor"
    }

# ---- FastAPI App ----
app = FastAPI(
    title="Brain Tumor Prediction API",
    description="Upload an MRI image to predict brain tumor type.",
    version="1.0"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = predict_image(file.file)
    return result

# ---- Run Uvicorn (for dev, use CMD in Docker for prod) ----
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)