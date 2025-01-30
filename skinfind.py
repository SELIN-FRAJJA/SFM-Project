import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define the label mapping (should match training label mapping)
class_label_mapping = {
    0: 'Actinic keratosis',
    1: 'Atopic Dermatitis',
    2: 'Benign keratosis',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevus',
    5: 'Melanoma',
    6: 'Squamous cell carcinoma',
    7: 'Vascular lesion'
}

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.vgg16(pretrained=True)

# Adjust the classifier to match training setup
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, len(class_label_mapping))  # Number of classes
)

# Load trained weights
model.load_state_dict(torch.load('morning.pth', map_location=device))
model = model.to(device)
model.eval()

# Define the same preprocessing steps used for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict the class of an unseen image (received as PIL Image)
def predict_skin_disease(image, model, transform, class_label_mapping):
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image to the appropriate device
    image = image.to(device)

    # Get the model's prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    # Get the class name from the label mapping
    class_name = class_label_mapping[predicted_class.item()]
    return class_name
