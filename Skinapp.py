from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import skinfind

app = Flask(__name__)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier[0].in_features, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(4096, len(skinfind.class_label_mapping))  # Number of classes
)
model.load_state_dict(torch.load('morning.pth', map_location=device))
model = model.to(device)
model.eval()

# Define the same preprocessing steps used for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('skinindex.html')

@app.route('/get_user_data', methods=['GET'])
def get_user_data():
    try:
        with open('user_data.txt', 'r') as file:
            user_data = file.read().strip()
        return user_data  # Sending plain text response
    except FileNotFoundError:
        return "Age and Sex: Data not available"
    
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    
    # Open and preprocess the image
    image = Image.open(file.stream).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        result = skinfind.class_label_mapping[predicted.item()]

    # Collect answers from the form
    answers = {}
    for i in range(1, 11):
        answer = request.form.get(f'q{i}')
        if answer == 'Others':
            others_answer = request.form.get(f'others_answer_{i}')
            answers[f'Question {i}'] = f"Others - {others_answer}"
        else:
            answers[f'Question {i}'] = answer

    # Prepare the response
    response = {
        'result': result,
        'symptoms': answers
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5001, debug=True)