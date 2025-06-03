
import numpy as np
import yaml
import cv2
from skimage import morphology

def get_corners(cx, cy, win_w, win_h, image_w, image_h):
    # Calculate the search window coordinates, ensuring they are within image bounds
    x1 = max(0, int(cx - win_w / 2))
    y1 = max(0, int(cy - win_h / 2))
    x2 = min(image_w, x1 + win_w)
    y2 = min(image_h, y1 + win_h)
    return x1, x2, y1, y2

def to_surface_coordinates(u, v, H):
    """
    Converts pixel coordinates (u, v) to surface coordinates using a homography matrix H.
    
    - Ensures u and v are NumPy arrays.
    - If u or v is a single float or int, converts them into a NumPy array.
    - If the result is a single point, returns x and y as scalars.
    """
    # Ensure u and v are NumPy arrays
    u = np.array([u]) if isinstance(u, (int, float)) else np.asarray(u)
    v = np.array([v]) if isinstance(v, (int, float)) else np.asarray(v)

    # Create homogeneous coordinates
    pixels = np.array([u, v, np.ones_like(u)])  # Shape (3, N)

    # Apply homography transformation
    points = H @ pixels  # Matrix multiplication (3,3) @ (3,N) -> (3,N)

    # Normalize x, y by w
    x, y = points[:2] / points[-1]  # Normalize x and y by the last row (homogeneous coordinate)

    # If x and y are single-element arrays, unpack them to return scalars
    if len(x) == 1 and len(y) == 1:
        return x[0], y[0]

    return x, y


def read_transform_config(filepath):
    """
    Reads calibration data from a YAML file and returns individual variables, including
    values from a nested dictionary.

    Args:
    filepath (str): Path to the YAML file containing calibration data.

    Returns:
    tuple: Returns individual variables extracted from the YAML file.
    """
    # Load the YAML data from the specified file
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)

    # Use eval to convert homography string to list of lists
    homography_str = data.get('homography', None)
    H = np.array(eval(homography_str))
    return H
    

def draw_box(image, im_canny, corners, color=(0, 255, 0), thickness = 2):

    x1, x2, y1, y2 = corners

    # Draw the box on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)

    # Extract the patch from the grayscale image
    patch = cv2.cvtColor(im_canny, cv2.COLOR_GRAY2RGB)[y1:y2, x1:x2, :]

    # Replace the corresponding region in the RGB image with the patch
    image[y1:y2, x1:x2, :] = patch

    return image

def parse_predictions(predictions, id2label: dict):
    """
    Process model predictions and generate a mask image for specified classes.

    Parameters:
    - predictions (list): Model output containing masks, class_ids, and scores.
    - id2label (dict): Mapping from class ID to label name (e.g., {0: 'car', 2: 'sign'}).

    Returns:
    - bool: True if any matching detections were found, False otherwise.
    - numpy.ndarray or None: Output mask with class regions, or None if no match.
    - dict: Mapping from class ID to {'label': str, 'score': float} for detected classes.
    """

    # Initialize score dictionary with 0.0 confidence per class ID
    scores = {id_: {'label': lbl, 'score': 0.0} for id_, lbl in id2label.items()}

    if len(predictions) == 0:
        return False, None, scores

    # We only process one image at a time (batch size = 1)
    p = predictions[0]

    bboxs = p.boxes
    ids = bboxs.cls.cpu().numpy()        # Class IDs e.g., center(0), stop(1)
    confidences = bboxs.conf.cpu().numpy() # Confidence scores (not used here)

    masks = p.masks

    # Only keep detections for class IDs we care about
    class_ids = list(id2label.keys())
    cls_mask = np.isin(ids, class_ids)

    # If none of the detections match the desired class_ids, exit early
    if not cls_mask.any():
        return False, None, scores

    shape = masks.orig_shape
    (height, width) = shape

    # Each detected object has its own mask
    data = masks.data.cpu().numpy()  # Shape: (N, W, H) — N = number of masks

    # Keep only the masks and IDs that match our class of interest
    ids = ids[cls_mask]
    data = data[cls_mask]
    conf = confidences[cls_mask]

    # Update score dictionary with max score for each class ID
    for id_ in np.unique(ids):
        max_score = float(conf[ids == id_].max())
        scores[id_]['score'] = max_score

    # Create an empty output image to store our final mask
    output = np.zeros(shape=shape, dtype=np.uint8)

    for i, mask in enumerate(data):
        # Resize the mask to match the original image size
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # We expect only one left and one right lane line
        # If there are multiple detections, we combine them into one mask
        # (Another option would be to keep only the detection with the highest confidence)
        output[mask == 1] = ids[i] + 1  # We add +1 because background is 0

    return True, output, scores


def get_base(mask, N = 100):
    y, x = np.nonzero(mask)
    xs = x[np.argsort(y, )][-N:]
    ys = y[np.argsort(y)][-N:]

    cx, cy = np.mean([xs, ys], axis = 1)

    return cx, cy

def draw_circle(image, x, y, radius=5, color=(0, 255, 0), thickness=-1):
    center = (int(x), int(y))  # Create center tuple from x and y
    cv2.circle(image, center, radius, color, thickness)
    return image

def detect_bbox_center(predictions, target_id):

    # Check if there are any predictions
    if len(predictions) == 0:
        return False, None, None

    p = predictions[0]  # Get the first prediction

    all_boxes = np.array(p.boxes)  # Access the bounding boxes
    ids = np.array(p.class_ids) # Class IDs for each detected object
    confidences = np.array(p.scores)  # Confidence scores for each detection

    # Check if the target class ID is present in the predictions
    if target_id not in ids:
        return False, None, None

    # Filter the boxes with the target ID
    boxes = all_boxes[ids == target_id]

    # Extract the center and size (xyxy) of the first box
    x1, y1, x2, y2 = boxes[0]

    # Calculate the center X-coordinate
    center_x = (x1 + x2) /2

    return True, float(center_x), float(y2)  # Return the bottom center coordinates

def get_bounding_boxes(predictions, objects_3d):

    # Check if there are any predictions
    if len(predictions) == 0:
        return False, None

    p = predictions[0].cpu()  # Get the first prediction (move to CPU)
    all_boxes = p.boxes  # Access the bounding boxes
    ids = all_boxes.cls.numpy()  # Class IDs for each detected object
    confidences = all_boxes.conf.numpy()  # Confidence scores for each detection

    detections = {}
    for id, lbl in objects_3d.items():
      # Check if the class ID is present in the predictions
      if id in ids:

        # Filter the boxes with the target ID
        boxes = all_boxes[ids == id]
        conf = confidences[ids == id]

        # Take the bbox with the highest confidence
        corners = boxes.xyxy[np.argmax(conf)].numpy()

        # Extract the center and size (xyxy) of the box with the highest confidence
        detections[id] = {'label': lbl, 'score': np.max(conf).item(), 'corners': corners}

    if len(detections) == 0:
      return False, None
    else:
      return True, detections

def get_onnx_boxes(predictions, objects_3d):

    # Check if there are any predictions
    if len(predictions) == 0:
        return False, None

    p = predictions[0]  # Get the first prediction

    all_boxes = np.array(p.boxes)  # Access the bounding boxes
    ids = np.array(p.class_ids) # Class IDs for each detected object
    confidences = np.array(p.scores)  # Confidence scores for each detection
    
    detections = {}
    for id, lbl in objects_3d.items():
      # Check if the class ID is present in the predictions
      if id in ids:

        # Filter the boxes with the target ID
        boxes = all_boxes[ids == id]
        conf = confidences[ids == id]

        # Extract the center and size (xyxy) of the first box
        corners = boxes[np.argmax(conf)]

        # Extract the center and size (xyxy) of the box with the highest confidence
        detections[id] = {'label': lbl, 'score': np.max(conf), 'corners': corners}

    if len(detections) == 0:
      return False, None
    else:
      return True, detections

def pixels_in_box(pixels, corners):
    """
    Returns a boolean mask for which pixels fall inside a given 2D bounding box.

    Args:
        pixels (np.ndarray): Nx2 array of [u, v] image coordinates.
        corners (list): [x1, y1, x2, y2] bounding box corners.

    Returns:
        np.ndarray: Boolean mask of shape (N,) with True for pixels inside the box.
    """
    u, v = pixels.T
    x1, y1, x2, y2 = corners

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    return (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)


def display_distances(image, distance_dict):
    """
    Draws all label: distance entries in the top-left corner of the image
    with a single white semi-transparent background box.

    Args:
        image (np.ndarray): The input BGR image.
        distance_dict (dict): Dictionary with {label: distance} entries.

    Returns:
        np.ndarray: Annotated image.
    """
    overlay = image.copy()
    output = image.copy()
    
    x, y = 10, 20
    dy = 20  # Line spacing
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Smaller font
    text_color = (0, 0, 0)  # Black text
    box_color = (255, 255, 255)  # White background
    thickness = 1
    alpha = 0.6  # Transparency

    # Prepare all lines and calculate max text width
    lines = [f"{label}: {dist:.2f} m" for label, dist in distance_dict.items()]
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max(w for w, h in text_sizes)
    total_height = dy * len(lines)

    # Draw one background box
    top_left = (x - 5, y - 15)
    bottom_right = (x + max_width + 5, y - 15 + total_height)
    cv2.rectangle(overlay, top_left, bottom_right, box_color, -1)

    # Draw each line of text
    for i, line in enumerate(lines):
        line_y = y + i * dy
        cv2.putText(overlay, line, (x, line_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Blend overlay and original image
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output

def generate_errors(predictions, H, window_size=1.5,half_lane_width=.700):
    name2id = {'left_lane':1, 'right_lane':2}


    p = predictions[0].cpu()  # Get the first prediction (move to CPU)
    all_boxes = p.boxes  # Access the bounding boxes
    ids = all_boxes.cls.numpy()  # Class IDs for each detected object
    confidences = all_boxes.conf.numpy()  # Confidence scores for each detection

    masks = p.masks

    if len(ids) == 0: return False, None, None

    shape = masks.orig_shape
    (height, width) = shape

    # Each detected object has its own mask
    data = masks.data.cpu().numpy()  # Shape: (N, W, H) — N = number of masks

    
    lane_classes = (name2id["left_lane"], name2id["right_lane"])   # {1,2}

    # indices of all lane detection
    lane_idxs   = np.where(np.isin(ids, lane_classes))[0]

    # take highest confidence lane
    best_idx    = lane_idxs[np.argmax(confidences[lane_idxs])]

    # keep only the most confident masks
    ids   = ids[[best_idx]]
    data  = data[[best_idx]]

    output = np.zeros(shape=(height,width), dtype=np.uint8)
    for i, mask in enumerate(data):
        # Resize the mask to match the original image size
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        skeleton = morphology.skeletonize(mask)
        output[skeleton == 1] = ids[i]

    # transform mask points into surface coordinates
    v, u = np.where(output != 0)
    x, y = to_surface_coordinates(u, v, H)

    # define a window of interest in the surface coordinates
    win_low_bound = x.min()
    win_upp_bound = win_low_bound + window_size

    win_mask = (x > win_low_bound) & (x < win_upp_bound)
    x_filt = x[win_mask]
    y_filt = y[win_mask]

    if x_filt.size < 2 or y_filt.size < 2: # filtering may result in no useable points
        return False, None, None

    # fit a line through points
    a, b = np.polyfit(y_filt, x_filt, 1)     # least-squares fit

    # calculate normal to the fitted line
    normal_start_x = 0.5 * (win_low_bound + win_upp_bound)
    normal_start_y = (normal_start_x - b) / a
    normal_start = np.array([normal_start_x,normal_start_y])

    n = np.array([-1.0, a])
    n /= np.linalg.norm(n)                      # unit length

    # decide which way the normal line should go
    waypoint_try1 = normal_start + half_lane_width * n
    waypoint_try2 = normal_start + half_lane_width * (n * -1)
    if abs(waypoint_try1[1]) < abs(waypoint_try2[1]):
        waypoint = waypoint_try1
    else:
        waypoint = waypoint_try2

    heading_err = -np.arctan(1/a)
    waypoint_err = -waypoint[1]

    return True, heading_err, waypoint_err
