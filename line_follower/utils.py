
import numpy as np
import yaml
import cv2

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


def draw_circle(image, x, y, radius=5, color=(0, 255, 0), thickness=-1):
    center = (int(x), int(y))  # Create center tuple from x and y
    cv2.circle(image, center, radius, color, thickness)
    return image

# Function to determine if dashline is present.  Takes in the predictions and
# a class id if necessary (set by default) and returns a flag to indicate if
# we were successful in finding a dashed line and the centroid of the dashed
# line.
def forkline_visible(predictions, cls_id = 0):

  # Step 1: Check if there are any predictions
  if len(predictions) == 0:
      return False, None, None
  _, dash_masks, dash_conf = get_masks_of_id(predictions, cls_id)
  if dash_masks is None:
    return False, None, None
  # If we have more than one detected dashed line take the highest confidence
  mask = get_highest_confidence(dash_masks, dash_conf)
  if mask is None:
    return False, None, None
  # Resize the mask to match the original image size
  c = get_top(mask, N=5)
  cx = c[0]
  cy = c[1]
  # Step 3: Return True (forkline visible) and the center x position
  return True, cx, cy


# Function to count the amount of people.  Takes in predictions and a class id
# to count (set by default), and returns an np array of centroids of the people
# in the image.  Returns an empty np array if no people are detected
def count_people(predictions, cls_id=2):
  # Returns the center x of all detected persons
  # Step 1: Check if there are any predictions
  if len(predictions) == 0:
      return np.array([])
  centers = []
  # Get the masks of the people
  _, ppl_masks, _ = get_masks_of_id(predictions, cls_id)

  # Make sure there are actually people in the scene
  if ppl_masks is None:
    return np.array([])

  # Get the centroid of each person (in the x direction)
  for mask in ppl_masks:
    c = get_centroid(mask)
    centers.append(c[0])

  return np.array(centers)

# Helper function that gets the centroid of a mask.  Takes in a mask and returns
# a list of the coordinates of the mask in [x, y]
def get_centroid(mask):
  # find the locations of the 1's in the mask image
  locs = np.argwhere(mask==1)
  # return the average position of the 1s in the image (flipping to return x,y)
  return np.flip(locs.mean(axis=0))

# Helper function, originally created by Professor Engel, that takes in a mask
# and returns the base of the mask
def get_base(mask, N = 100):
    y, x = np.nonzero(mask)
    xs = x[np.argsort(y, )][-N:]
    ys = y[np.argsort(y)][-N:]

    cx, cy = np.mean([xs, ys], axis = 1)

    return [cx, cy]

# Get top, exactly the same as get base but gets the top of the mask instead
def get_top(mask, N = 100):
    y, x = np.nonzero(mask)
    xs = x[np.argsort(y, )][:N]
    ys = y[np.argsort(y)][:N]

    cx, cy = np.mean([xs, ys], axis = 1)

    return [cx, cy]

# Combines the centroid and base getters to return the x value of the centroid
# of the mask but the y value of the base of the mask.  Doing this made the
# homographic transformation the most stable in testing for the path following.
# Getting just the centroid lead to wild swings in the steering angle, and just
# getting the base didnt steer the car enough (and provided wrong results)
def get_centroid_at_base(mask):
  # Get the y value of the base of the mask
  _, y = get_base(mask)
  # get the x value of the centroid of the mask
  c = get_centroid(mask)
  return [c[0], y]

# Function to find the highest confidence mask in a list of masks. Takes in
# lists of masks and matching confidences and returns the highest confidence
# mask.
def get_highest_confidence(masks, conf):
  # perform a check to make sure we have more than one mask
  if len(masks) > 1:
    # if we do, get the highest confidence mask
    i = np.argmax(conf)
    mask = masks[i]
  else:
    # if not just return the mask
    mask = masks[0]
  return mask


# Helper function (abstracting out this code from each of the driving functions)
# that returns a lists of ids, masks, and confidences of a particular mask.
# Takes in a prediction to parse through and the id of the masks you want to
# return.  Returns the ids, masks, and confidences associated with that id.
def get_masks_of_id(predictions, id):
  # Get the first prediciton if we have multiple
  p = predictions[0]

  bboxs = p.boxes
  ids = bboxs.cls.cpu().numpy()        # Class IDs e.g., center(0), stop(1)
  conf = bboxs.conf.cpu().numpy() # Confidence scores (not used here)
  
  masks = p.masks

  # If we have no masks, (i.e. nothing was predicted), return None
  if p.masks is None:
    return None, None, None

  # Get the ids, confidences, and masks from the prediction

  shape = masks.orig_shape
  (height, width) = shape
  
  # Each detected object has its own mask
  masks = masks.data.cpu().numpy()  # Shape: (N, W, H) — N = number of masks
  
  # Create a mask for detections that match our target class IDs
  cls_mask = np.isin(ids, id)

  # If we have none of the desired class, return None
  if len(cls_mask) == 0:
    return None, None, None

  # Use logical indexing to get the desired ids masks and confidences
  parsed_ids = ids[cls_mask]
  parsed_masks = masks[cls_mask]
  parsed_conf = conf[cls_mask]

  # Final check to make sure the parsed lists are not empty
  if len(parsed_ids) == 0:
    parsed_ids = None
  if len(parsed_masks) == 0:
    parsed_masks = None
  if len(parsed_conf) == 0:
    parsed_conf = None

  # return the parsed lists
  return parsed_ids, parsed_masks, parsed_conf


# Function to return either the left most left line or right most right line.
# Takes in a list of masks and a string that is either 'left' or 'right' to
# determine if we are deciding the left most line or right most line (set by
# default to left).  Returns the either left or right most mask and its centroid
# at its base (x is the centroid, y is the base).
def get_outside_most(masks, lr='left'):
  # Perform a check to make sure lr is either 'left' or 'right'
  if lr.lower() != 'left' and lr.lower() != 'right':
    raise ValueError("lr must be either 'left' or 'right'")

  #  Allocate empty list of x and y values
  mask_x_values = []
  mask_y_values = []

  # Loop through the masks and store their x and y coordinates in the above lists
  for mask in masks:
    c = get_centroid_at_base(mask)
    mask_x_values.append(c[0])
    mask_y_values.append(c[1])

  # Convert the lists to np arrays
  mask_x_values = np.array(mask_x_values)
  mask_y_values = np.array(mask_y_values)

  # Check if we are looking for leftmost or rightmost
  # if its left most, get the index of the minimum x value and store those x, y,
  # and masks in respective variables.
  if lr.lower() == 'left':
    leftmost_in = np.argmin(mask_x_values)
    cx = mask_x_values[leftmost_in]
    cy = mask_y_values[leftmost_in]
    mask = masks[leftmost_in]

  # If we want the right most, same process but get the maximum x value
  elif lr.lower() == 'right':
    rightmost_in = np.argmax(mask_x_values)
    cx = mask_x_values[rightmost_in]
    cy = mask_y_values[rightmost_in]
    mask = masks[rightmost_in]

  # Final check to make sure we have 'left' or 'right' as our lr variable
  else:
    raise ValueError("lr must be either 'left' or 'right'")

  # Return the mask and the x and y coordinates
  return mask, cx, cy


def calculate_drive_waypoint(predictions, left_id = 1, right_id = 3):
  # check to make sure we have predictions
  if len(predictions) == 0:
      return False, None, None

  # get the size of the screen in the predictions
  """DOES NOT WORK FOR ONNX RUNTIME I THINK"""
  #screen_size = predictions[0].orig_shape
  screen_size = [360, 640]
  # use that to get the center of the screen
  center_screen_x = screen_size[1]/2
  center_screen_y = screen_size[0]/2


  # Assume a fixed offset for one line driving.  Basically, the goal is to shift
  # the line we are following to the center of the screen to treat it like a
  # line follower. This fixed offset is assumed to be in the center of the
  # screen.  The right offset is negative since the offset is added
  left_offset = center_screen_x
  right_offset = -center_screen_x


  # get the list of right line and left line masks
  left_ids, left_masks, left_conf = get_masks_of_id(predictions, left_id)
  right_ids, right_masks, right_conf = get_masks_of_id(predictions, right_id)

  # Confirm we have left and right lines in the image
  if left_masks is None and right_masks is None:
    return False, None, None

  # predefine these values
  left_cx = 0.0
  right_cx = 0.0
  left_cy = 0.0
  right_cy = 0.0

  # clip offset is for shrinking the clip window so the waypoint can't go past it
  clip_offset = 50

  # For the left and right masks, we need to decide which line (if there are
  # multiple) we should follow.  For other instances, it might make sense to get
  # the highest confidence line.  We chose to instead get the left most line for
  # the left lines and rightmost line for the right lines.
  if left_masks is not None:
    left_mask, left_cx, left_cy = get_outside_most(left_masks)
  else:
    left_mask = None

  if right_masks is not None:
    right_mask, right_cx, right_cy = get_outside_most(right_masks, lr='right')
  else:
    right_mask = None

  # Sometimes, left side is like way too far left (same for right some times)
  # so if a left or right side is further than 3/4 of the way to the other side
  # of the screen, disregard the line
  if left_mask is not None and left_cx > screen_size[1]*0.75:
      left_mask = None
  if right_mask is not None and right_cx < screen_size[1]*0.25:
      right_mask = None

  # Get Center of the screen for the following
  # First deal with the condition there is both a left and a right line
  # If both are present, average their centroids and return
  if left_mask is not None and right_mask is not None:
    return True, (left_cx+right_cx)/2, (left_cy+right_cy)/2

  # Next deal with the left line
  if right_mask is None and left_mask is not None:
    # The waypoint we should follow should be offset by the constant offset
    # I am also clipping it so it stays on the screen.  This helps prevents
    # major spikes.
    left_cx = max(min(left_cx + left_offset, screen_size[1])-clip_offset,clip_offset)
    return True, left_cx, left_cy

  # Next deal with the right line
  # Now do the same for the right side
  if right_mask is not None and left_mask is None:
    # The waypoint we should follow should be offset by the constant offset
    # I am also clipping it so it stays on the screen.  This helps prevents
    # major spikes.
    right_cx = max(min(right_cx + right_offset, screen_size[1]-clip_offset),clip_offset)
    return True, right_cx, right_cy

  # if all else fails, return error state
  return False, None, None
