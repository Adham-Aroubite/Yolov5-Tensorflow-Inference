import tensorflow as tf
import numpy as np
import cv2


# Define variables
image_path = ""  # Replace with your image path
model_path = ""  # Replace with your model path
img_size = 640
iou_threshold = 0.3
score_threshold = 0.4
class_names = ["Signature"]  # Placeholder: replace with actual class names if available

def load_and_preprocess_image(image_path, img_size=640):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    original_image = image.copy()
    h, w, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    print(f"Original image size: {h}x{w}, Resized image size: {img_size}x{img_size}")
    return original_image, image, (h, w)

def non_max_suppression(boxes, scores, iou_threshold, score_threshold=0.6):
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size=1000, iou_threshold=iou_threshold, score_threshold=score_threshold)
    return indices

def draw_boxes(image, boxes, scores, classes, class_names):
    for box, score, cls in zip(boxes, scores, classes):
        y_min, x_min, y_max, x_max = box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
        label = f"{class_names[int(cls)]}: {score:.2f}"
        cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

def predict(image_path, model_path, img_size=640, iou_threshold=0.3, score_threshold=0.4, class_names=None):
    original_image, image, (orig_h, orig_w) = load_and_preprocess_image(image_path, img_size)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    print(f"Loading model from: {model_path}")
    model = tf.saved_model.load(model_path)
    print("Model loaded successfully.")

    predictions = model(image)[0].numpy()  # Access the first (and likely only) element in the output list and convert to numpy

    # Verify the complete prediction shape
    print(f"Prediction shape: {predictions.shape}")

    # Print first 5 elements of each prediction tensor
    print(f"Predictions (first 5 elements):")
    print(f"\tboxes:\n\t\t{predictions[0, :5, :4]}")
    print(f"\tscores:\n\t\t{predictions[0, :5, 4]}")
    print(f"\tclasses:\n\t\t{predictions[0, :5, 5]}")

    # Adjust output interpretation
    boxes = predictions[0, :, :4]
    scores = predictions[0, :, 4]
    classes = predictions[0, :, 5]

    # Convert from [x_center, y_center, width, height] to [xmin, ymin, xmax, ymax]
    print("Converting bounding box coordinates to [xmin, ymin, xmax, ymax] format...")
    x_center = boxes[:, 0] * img_size
    y_center = boxes[:, 1] * img_size
    width = boxes[:, 2] * img_size
    height = boxes[:, 3] * img_size

    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2

    # Convert back to original image dimensions
    xmin = xmin * (orig_w / img_size)
    ymin = ymin * (orig_h / img_size)
    xmax = xmax * (orig_w / img_size)
    ymax = ymax * (orig_h / img_size)

    boxes = np.stack([ymin, xmin, ymax, xmax], axis=1)

    print(f"Converted boxes (first 5 elements):\n{boxes[:5]}")

    # Apply non-max suppression
    print("Applying non-max suppression...")

    indices = non_max_suppression(boxes, scores, iou_threshold, score_threshold)
    boxes = boxes[indices]
    scores = scores[indices]
    classes = classes[indices]

    if len(boxes) > 0:
        # Get the coordinates of the first (and likely only) bounding box
        ymin, xmin, ymax, xmax = boxes[0]
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)

        # Crop the image using the bounding box coordinates
        cropped_image = original_image[ymin:ymax, xmin:xmax]

        return "yes", cropped_image
    else:
        return "no", None

# Call the predict function with defined variables
detection_result, cropped_image = predict(image_path, model_path, img_size, iou_threshold, score_threshold, class_names)

print("Detection Result:", detection_result)

if detection_result == "yes":
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
