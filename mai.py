import cv2
import numpy as np
import insightface
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Initialize FaceAnalysis and the face swapper model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

def img_FaceSwap(source_img_path, reference_img_path):
    # Load source and reference images
    source_img = cv2.imread(source_img_path)
    reference_img = cv2.imread(reference_img_path)

    # Detect faces in the source image
    source_faces = app.get(source_img)
    if len(source_faces) == 0:
        raise ValueError("No face detected in the source image.")
    source_face = source_faces[0]

    # Detect faces in the reference image
    reference_faces = app.get(reference_img)
    if len(reference_faces) == 0:
        raise ValueError("No face detected in the reference image.")
    
    # Create a copy of the reference image for swapping
    swapped_img = reference_img.copy()

    # Perform face swapping
    for ref_face in reference_faces:
        swapped_img = swapper.get(swapped_img, ref_face, source_face, paste_back=True)

    return swapped_img

# Example usage
source_image_path = './resource/source/gagan_snow.jpeg'  # Replace with your source image path
reference_image_path = './resource/reference/vmen19.jpg'  # Replace with your reference image path

# Perform face swap
result_image = img_FaceSwap(source_image_path, reference_image_path)

# Display the result
plt.imshow(result_image[:, :, ::-1])  # Convert BGR to RGB for displaying
plt.axis('off')  # Hide axes
plt.show()  # Show the final image