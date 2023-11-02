import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import albumentations as A
import numpy as np

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=256, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)




    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block

    def forward(self, X):

        contracting_11_out = self.contracting_11(X)

        contracting_12_out = self.contracting_12(contracting_11_out)

        contracting_21_out = self.contracting_21(contracting_12_out)

        contracting_22_out = self.contracting_22(contracting_21_out)

        contracting_31_out = self.contracting_31(contracting_22_out)

        contracting_32_out = self.contracting_32(contracting_31_out)

        middle_out = self.middle(contracting_32_out)

        expansive_11_out = self.expansive_11(middle_out)

        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_31_out), dim=1))

        expansive_21_out = self.expansive_21(expansive_12_out)

        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_21_out), dim=1))

        expansive_31_out = self.expansive_31(expansive_22_out)

        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_11_out), dim=1))

        output_out = self.output(expansive_32_out)
        return output_out
# Load the trained model
model = Net(12)
model.load_state_dict(torch.load('SegNet-Aug.zip', map_location='cpu'))

# Define the predict function
def predict(image):
    # Preprocess the image
    image = preprocess(image)
    # Make a prediction
    with torch.no_grad():
        output = model(image)
    # Postprocess the output
    output = postprocess(output)
    return output

# Define the preprocess function
def preprocess(image):
    trans_obj = A.Compose([A.Resize(128, 128),
                                   A.Normalize([0.,0.,0.], [1.,1.,1.])])
    
    image = np.array(Image.open(image))

    transformed = trans_obj(image = image)

    return torch.from_numpy(transformed['image']).unsqueeze(0).permute(0, 3, 1, 2).float()



# Define the postprocess function
def postprocess(output):
    # Remove the batch dimension
    output = output.squeeze(0)
    # Convert the output to a numpy array
    output = output.cpu().numpy()
    # Convert the output to a segmentation mask
    output = np.argmax(output, axis=0)
    # Convert the segmentation mask to an RGB image
    output = colormap[output]
    return output

# Define the colormap
colormap = np.array([[ 42.31849768, 138.68802268, 107.34976084],
       [127.04043101,  64.08636608, 127.54335584],
       [ 67.73835184,  70.81071668,  69.68390725],
       [ 77.50990865,  18.95552339,  73.79703489],
       [133.96612288,   3.85362098,   4.37284647],
       [217.97385828,  43.76697629, 229.43543734],
       [164.94309451, 125.22840326,  81.50394562],
       [157.10134385, 155.26893603, 193.22678809],
       [ 66.53429189,  32.62107138, 188.45387454],
       [157.58165138, 243.49941618, 159.97381151],
       [  6.98127537,   5.22420501,   6.82420501],
       [ 48.88183862, 203.80514614, 203.66699975]
          ])

# Define the main function
def main():
    # Set the title
    st.title('CityScape Image Segmentation')
    # Upload the image
    image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if image is not None:
        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Make a prediction
        output = predict(image)
        # Display the output
        output=output/255.0
        st.image(output, caption='Segmentation Mask', use_column_width=True)

# Run the main function
if __name__ == '__main__':
    main()