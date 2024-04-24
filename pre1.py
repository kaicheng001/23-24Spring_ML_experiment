# predict.py
import torch
from torchvision import transforms
from PIL import Image

def load_model(model_path):
    """
    Load the PyTorch model from the specified path.
    """
    model = torch.load(model_path)
    #model.eval()  # Set the model to evaluation mode
    return model

def predict(model, image):
    """
    Make a prediction for the given image using the provided model.
    """
    # Assuming image is a PIL image, we must transform it first.
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the PIL Image to a tensor.
        transforms.Normalize((0.5,), (0.5,))  # Normalize the image
    ])
    
    # Apply the transformations and add batch dimension
    image = transform(image).unsqueeze(0)
    
    # No need to track gradients for prediction
    with torch.no_grad():
        output = model(image)
    
    # Get the predicted class with the highest score
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()  # Return the index as a Python int

# main.py (or any other script where you want to make a prediction)



# Load the model
model = load_model('fashion_mnist.pt')

# Load an image (ensure the image is in the correct format for the model)
image = Image.open('image.png')

# Make a prediction
predicted_class = predict(model, image)
print('Predicted class:', predicted_class)
