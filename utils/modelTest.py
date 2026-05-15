# testopenvinoDetector2.py
import openvino as ov
import cv2
import numpy as np
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenVINO with optimizations
core = ov.Core()
model = core.read_model("../models/20251008_233645/exported_model.xml")

# CPU optimization settings
openvino_config = {"PERFORMANCE_HINT": "LATENCY"}  # Renamed to avoid conflict
compiled_model = core.compile_model(model, "CPU", openvino_config)

# Get input and output info
input_layer = compiled_model.input(0)
print("COMPILED MODEL INPUT:", input_layer.get_shape())
output_layer = compiled_model.output(0)
print("COMPILED MODEL OUTPUT:", output_layer.get_partial_shape())

def preprocess(image, input_size=(416, 416)):
    """Optimized preprocessing"""
    h, w = image.shape[:2]
    scale = min(input_size[0]/h, input_size[1]/w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    logging.info(f"Input image shape: {image.shape}, Scale: {scale:.4f}, Resized shape: ({new_h}, {new_w})")
    
    # Use INTER_LINEAR for faster resizing
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pre-allocate padded array
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    logging.info(f"Padded shape before transpose: {padded.shape}")
    
    # Convert to RGB (model likely expects RGB, not BGR)
    padded = padded[:, :, ::-1]  # Reverse channels
    
    # Vectorized operations
    input_tensor = padded.transpose(2, 0, 1).astype(np.float32)
    input_tensor *= (1.0/255.0)  # Faster than division
    input_tensor = np.expand_dims(input_tensor, 0)
    
    logging.info(f"Input tensor shape after processing: {input_tensor.shape}")
    
    return input_tensor, scale, (new_h, new_w)

def refine_bbox_to_edges(image, bbox):
    """Refine bbox to tightly fit plate edges using edge detection"""
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    
    # Convert to gray and apply Canny edges
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours and select the largest rectangular one (plate)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Filter for rectangular shapes (approx 4 sides)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:  # Top 5 largest
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:  # Quadrilateral (plate)
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                # Adjust back to full image coords
                x1, y1, x2, y2 = x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch
                return [x1, y1, x2, y2]
    
    # Fallback to original if no good contour
    return bbox

def postprocess(output, original_shape, scale, resized_shape, conf_threshold=0.75, iou_threshold=0.4, roi_offset=(0, 0)):
    tensor = output[0]
    logging.info(f"Model output shape: {tensor.shape}")
    logging.info(f"Raw tensor content: {tensor}")  # Inspect values
    
    # Assume [1,N,C] -> (N,C); adjust if different
    predictions = tensor.reshape(-1, tensor.shape[-1])
    
    max_conf = np.max(predictions[:, -1])  # Assume last col is conf
    logging.info(f"Max confidence: {max_conf:.4f}")
    
    # Filter on conf (last col) and ignore padding (conf > 0)
    conf_mask = (predictions[:, -1] > conf_threshold) & (predictions[:, -1] > 0)
    if not np.any(conf_mask):
        return []
    
    filtered_preds = predictions[conf_mask]
    
    boxes = []
    confidences = []
    
    for detection in filtered_preds:
        confidence = detection[-1]  # Last is conf (no obj*cls if integrated)
        x_center, y_center, width, height = detection[:4]
        
        # Convert to corner coords
        x1 = int((x_center - width / 2))
        y1 = int((y_center - height / 2))
        x2 = int((x_center + width / 2))
        y2 = int((y_center + height / 2))
        
        logging.info(f"Raw box (padded space): [{x1}, {y1}, {x2}, {y2}], Confidence: {confidence}")
        
        boxes.append([x1, y1, x2, y2])
        confidences.append(float(confidence))
    
    if not boxes:
        return []
    
    # NMS with width/height
    nms_boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, conf_threshold, iou_threshold)
    
    if len(indices) == 0 and len(boxes) > 0:  # Force include if NMS empty (e.g., single box)
        indices = np.arange(len(boxes))
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            
            # Scale back to original image size and adjust for ROI offset
            x1 = max(0, min(int(box[0] / scale) + roi_offset[0], original_shape[1]))
            y1 = max(0, min(int(box[1] / scale) + roi_offset[1], original_shape[0]))
            x2 = max(0, min(int(box[2] / scale) + roi_offset[0], original_shape[1]))
            y2 = max(0, min(int(box[3] / scale) + roi_offset[1], original_shape[0]))
            
            # Apply shrink after scaling
            shrink_factor = 0.1
            w = x2 - x1
            h = y2 - y1
            x1 = int(x1 + w * shrink_factor / 2)
            y1 = int(y1 + h * shrink_factor / 2)
            x2 = int(x2 - w * shrink_factor / 2)
            y2 = int(y2 - h * shrink_factor / 2)
            
            logging.info(f"Scaled/Adjusted box (original space): [{x1}, {y1}, {x2}, {y2}]")
            
            final_boxes.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidences[i]
            })
    
    return final_boxes

def detect_plates(image, roi_coords=None):
    """Main detection function"""
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        # Add small margin (e.g., 10% of ROI size) to catch edges
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.shape[1], x2 + margin_x)
        y2 = min(image.shape[0], y2 + margin_y)
        cropped = image[y1:y2, x1:x2]
        roi_offset = (x1, y1)  # For postprocess adjustment
    else:
        cropped = image
        roi_offset = (0, 0)
    
    input_tensor, scale, resized_shape = preprocess(cropped)
    result = compiled_model([input_tensor])
    plates = postprocess(result, image.shape, scale, resized_shape, roi_offset=roi_offset)
    return plates

def draw_detections(image, detections):
    """Simplified drawing for images"""
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        bbox = refine_bbox_to_edges(image, bbox)  # Refine to edges
        
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)  # Green for plates
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw confidence
        label = f"Plate: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    parser = argparse.ArgumentParser(description="Detect license plates in an image")
    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument("--output_image", type=str, default="output.jpg", help="Path to save the output image (default: output.jpg)")
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.input_image)
    if image is None:
        logging.error(f"Failed to load image: {args.input_image}")
        return
    
    # Optional ROI - can be configured if needed, here using full image
    roi_coords = None  # Or set [x1, y1, x2, y2] manually
    
    # Detect plates
    detections = detect_plates(image, roi_coords)
    
    # Draw detections
    draw_detections(image, detections)
    
    # Save or show output
    cv2.imwrite(args.output_image, image)
    logging.info(f"Output saved to {args.output_image}")
    # cv2.imshow("Detected Plates", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()