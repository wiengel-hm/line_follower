import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit
import numpy as np
import time
import os
from functools import wraps
from glob import glob
import cv2

# trtexec --onnx=model.onnx --saveEngine=model.trt


def time_tracker(func):
    """
    A decorator that tracks the time taken to execute the decorated function.

    Args:
        func (callable): The function to be timed.

    Returns:
        callable: The wrapped function with time tracking.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result

    return wrapper

class TensorRTModel:
    def __init__(self, engine_path):
        """
        Initialize the TensorRT model.

        Args:
            engine_path (str): Path to the TensorRT engine file.
            input_shape (tuple): Shape of the input tensor (e.g., (1, 35, 160, 1)).
        """

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

        self.engine_path = engine_path

        # Load the TensorRT engine
        # The engine represents the optimized neural network model in TensorRT.
        # It contains all the layers, weights, and configurations required for inference.
        self.engine = self.load_engine()

        # Create an execution context
        # The execution context is a stateful object that manages the actual execution of the engine.
        # It handles memory allocations, manages input and output bindings, and performs the inference.
        # Each context corresponds to a single instance of inference execution.
        self.context = self.engine.create_execution_context()

        # No need to set the binding shape since the input shape is static.
        # Bindings correspond to the input or output tensors of the engine. Each binding has a specific index.
        # For this model, the input shape is fixed (static) and does not require adjustment before inference.
        # The expected input shape is (1, 35, 160, 1) as defined in the engine.
        # 
        # If the input shape were dynamic (e.g., [-1, 35, 160, 1]), we would need to set the shape explicitly:
        # self.context.set_binding_shape(0, input_shape)

        # Input
        input_name = self.engine.get_tensor_name(0)
        self.input_shape = self.engine.get_tensor_shape(input_name)
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(input_name))

        # Output
        output_name = self.engine.get_tensor_name(1)
        self.output_shape = self.engine.get_tensor_shape(output_name)
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(output_name))

        print(f"Expected input  shape: {self.input_shape}, dtype: {self.input_dtype}")
        print(f"Expected output shape: {self.output_shape}, dtype: {self.output_dtype}")


        # Allocate host and device memory for inputs and outputs
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

        self.warmup_model()

    @property
    def sample_input(self):
        # Get input shape (excluding batch dimension)
        _, h, w, c = self.input_shape
        return np.random.rand(h, w, c).astype(self.input_dtype)
    
    @time_tracker
    def warmup_model(self, warmup_rounds: int = 100):
        """Warm up the model with random data"""
        for _ in range(warmup_rounds):
            _ = self.predict(self.sample_input)


    @time_tracker
    def load_engine(self):
        """
        Load a TensorRT engine from a file.

        Returns:
            engine: The deserialized TensorRT engine.
        """
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def allocate_buffers(self):
        """
        Allocate host and device memory for the inputs and outputs of the TensorRT engine.

        Host memory: CPU memory that is page-locked for faster transfers to the GPU.
        Device memory: GPU memory used for inference.

        Returns:
            inputs, outputs, bindings, stream: Memory buffers and CUDA stream.
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            # Each binding corresponds to an input or output of the model. Bindings are defined during the engine creation
            # and represent the data connections to the engine. In TensorRT, bindings are indexed and need to be configured
            # correctly to match the model's inputs and outputs.

            # Calculate the size and data type for the binding
            # The size of the binding is determined by the volume (total number of elements) of its shape.
            # The trt.volume function computes this, and self.engine.get_tensor_shape(binding) retrieves the shape.
            # Example: For an input image with shape [1, 35, 160] (batch size = 1, height = 35, width = 160),
            # the size would be 35 * 160 = 5600 elements.
            # The self.engine.get_tensor_dtype(binding) provides the data type of the binding, such as float32.

            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            size = trt.volume(shape)

            # In the PilotNet model used here:
            # - The input is an image tensor with shape [1, 35, 160]. Its size is 35 x 160 = 5600 elements, 
            #   each of type float32, making it an array of 5600 float32 values.
            # - The output represents the predicted steering angle, which is a single scalar value.
            #   Its size is 1 element of type float32.

            # Bindings must be configured to correctly match the inputs and outputs of the model. For example:
            # - Input bindings are populated with input data (e.g., the preprocessed image).
            # - Output bindings hold the results of inference (e.g., the predicted steering angle).
            # The input/output role is determined using self.engine.get_tensor_mode(binding)

            # Allocate host (CPU) memory and device (GPU) memory
            host_mem = cuda.pagelocked_empty(size, dtype)  # Host memory
            device_mem = cuda.mem_alloc(host_mem.nbytes)   # Device memory

            # Categorize the binding as input or output
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))

            # Append device memory to the bindings list
            # The bindings list stores CUDA memory addresses for input/output tensors,
            # which TensorRT uses during inference to access input data and store results.
            bindings.append(int(device_mem))

        return inputs, outputs, bindings, stream


    def predict(self, image):
        """
        Perform inference on a raw input image.

        This method handles preprocessing, TensorRT inference,
        and postprocessing (parsing) of the model output.

        Args:
            image (numpy.ndarray): Raw input image (BGR).

        Returns:
            numpy.ndarray: Parsed model prediction (e.g., binary mask).
        """

        # Preprocess image (e.g., resize, normalize, convert to RGB)
        image = self.preprocess_image(image)

        # Ensure the input is of the correct shape and type
        assert image.shape == self.input_shape[1:], "Input shape mismatch"
        assert image.dtype == np.float32, "Input type must be np.float32"

        # Copy input data to host memory
        np.copyto(self.inputs[0][0], image.ravel())

        # Transfer input data from host to device memory
        cuda.memcpy_htod_async(self.inputs[0][1], self.inputs[0][0], self.stream)

        # Run inference asynchronously
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Transfer output data from device to host memory
        cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)

        # Synchronize the stream to ensure all GPU operations are complete
        self.stream.synchronize()

        # Get raw output from host memory
        prediction = self.outputs[0][0]

        # Postprocess raw model output (e.g., reshape, threshold, resize)
        result = self.parse_prediction(prediction)

        return result

    def preprocess_image(self, image):
        return image

    def parse_prediction(self, prediction):
        return prediction
    

class UNet(TensorRTModel):
    def __init__(self, engine_path, im_w = 640, im_h = 360):
        super().__init__(engine_path)

        in_h, in_w, _ = self.input_shape[1:]
        out_h, out_w, _ = self.output_shape[1:]

        self.preprocess_image = lambda im: self.preprocess_image_(im, in_w, in_h)
        self.parse_prediction = lambda y: self.parse_prediction_(y, out_w, out_h, im_w, im_h)

    @staticmethod
    def preprocess_image_(img_bgr, target_width, target_height):
        # Convert image from BGR (OpenCV default) to RGB (model input format)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to target dimensions using bilinear interpolation (similar to PIL default)
        img_rgb_resized = cv2.resize(img_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Normalize pixel values to [0, 1] as float32
        return img_rgb_resized.astype(np.float32) / 255.0

    @staticmethod
    def parse_prediction_(prediction, out_w, out_h, im_w, im_h):
        # Step 1: Reshape output
        mask = prediction.reshape(out_h, out_w)

        # Step 2: Binarize the mask
        mask_bin = (mask > 0.5).astype("uint8")

        # Step 3: Resize to original image size
        mask_resized = cv2.resize(mask_bin, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        return mask_resized


# Example usage
if __name__ == "__main__":
    # Path to the TensorRT engine file
    engine_path = "/mxck2_ws/src/line_follower/models/unet.trt"

    # Initialize the TensorRT model
    model = UNet(engine_path)

    # # Generate a batch of random images for inference
    num_images = 5
    

    start_time = time.time()
    for _ in range(num_images):
        results = model.predict(model.sample_input)
    end_time = time.time()

    # Calculate and print FPS
    total_time = end_time - start_time
    fps = num_images / total_time
    print(f"Processed {num_images} images in {total_time:.2f} seconds. FPS: {fps:.2f}")


    im_dir = '/mxck2_ws/src/line_follower/data'

    # Load all images matching the pattern 'sample*.png'
    images = [cv2.imread(path) for path in glob(im_dir + '/sample*.png')]


    # Run inference on the first image
    mask = model.predict(images[1])

    # Save the predicted mask as a color image
    rgb_mask = cv2.cvtColor((mask * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    cv2.imwrite(im_dir + '/result01.png', rgb_mask)
