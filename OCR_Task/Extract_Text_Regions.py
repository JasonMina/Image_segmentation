import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
from sklearn.cluster import DBSCAN
from skimage.measure import shannon_entropy

# --- YOUR ORIGINAL TextRegionExtractor CLASS AND HELPER FUNCTIONS (UNCHANGED) ---
def choose_gamma_by_std(image: np.ndarray, verbose: bool = False) -> float:
    """
    Using linear interpolation for gamma choice based on std
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Interpolated gamma value
    """
    std_intensity = np.std(image.astype(np.float64))
    
    if verbose:
        print(f"Image std: {std_intensity:.2f}")
    
    # Define control points (std, gamma) for interpolation
    # Matches the threshold logic: std >= 32.5 gets gamma = 1.33
    control_points = np.array([
        [0, 0.5],       # Very low std -> low gamma (brighten)
        [20, 0.5],
        [25, 0.8],
        [30, 1],
        [31, 1],       
        [31.5, 1.05],
        [32, 1.11],
        [32.5, 1.22],
        [33, 1.33],   # Jump to 1.33 at std >= 32.5
        [100, 1.33]    # Keep 1.33 for all higher std values
    ])
    
    # Interpolate gamma value
    gamma = np.interp(std_intensity, control_points[:, 0], control_points[:, 1])
    
    if verbose:
        print(f"Interpolated gamma: {gamma:.3f}")
    
    return gamma


class TextRegionExtractor:
    def __init__(self):
        self.debug = False
    
    def extract_text_regions(self, image_path: str, output_dir: str = "extracted_regions") -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Extract text regions from an image and save them as separate images.
        
        Args:
            image_path (str): Path to the input image
            output_dir (str): Directory to save extracted regions
            
        Returns:
            List[np.ndarray]: List of extracted text region images
            List[Tuple[int, int, int, int]]: List of bounding boxes (x, y, w, h) for filtered regions
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply different preprocessing techniques to enhance text visibility
        processed_images = self._preprocess_image(gray)
        
        all_regions = []
        all_bounding_boxes = []
        
        # Try different preprocessing approaches
        for i, processed in enumerate(processed_images):
            regions, bboxes = self._find_text_regions(processed, image)
            
            # Extend lists with regions and their bounding boxes
            all_regions.extend(regions)
            all_bounding_boxes.extend(bboxes)
        
      
        
        # Save filtered regions
        for j, region in enumerate(filtered_regions):
            filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_region{j}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, region)
            print(f"Saved region: {filepath}")
        
        return filtered_regions, filtered_bboxes # Return both regions and bboxes
    
    def _preprocess_image(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Apply various preprocessing techniques to enhance text visibility.
        """
        processed_images = []
        
        # Local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
    
        # Gamma correction to brighten dark areas based on std
        
        gamma = choose_gamma_by_std(gray)
        #print("GAMMA: ", gamma, "INTENA: ", np.mean(gray))
        gamma_corrected = np.power(enhanced / 255.0, gamma)
        gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

        # Binarization of the image

        _, enhanced_thresh = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        processed_images.append(enhanced_thresh)
        
        return processed_images
    
    def _find_text_regions(self, processed: np.ndarray, original: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Find text regions using contour detection and filtering.
        args: 
            processed (np.ndarray): Pre-processed 
            original (np.ndarray): Image Original image
        Returns:
            Tuple containing:
            - List of text region images
            - List of bounding boxes (x, y, w, h)
        """
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #SIMPLE CHAIN ARPROXIMATION
        
        # Filter contours based on area and aspect ratio
        text_regions = []
        bounding_boxes = []
        
        for contour in contours:
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size and aspect ratio
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Threshold to filter detected bounding boxes
            if (area > 500 and area < 10000 and  # Size constraints
                aspect_ratio > 0.4 and aspect_ratio < 10 and  # Aspect ratio constraints
                w > 7 and h > 7):  # Minimum dimensions
                
                # Add padding around the letter region
                padding = 10 
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(original.shape[1], x + w + padding)
                y_end = min(original.shape[0], y + h + padding)
                
                # Extract the region from the original image
                region = original[y_start:y_end, x_start:x_end]
                text_regions.append(region)
                
                # Store bounding box (with padding)
                bounding_boxes.append((x_start, y_start, x_end - x_start, y_end - y_start))
        
        return text_regions, bounding_boxes
    
    
    def visualize_detection(self, image_path: str, show_steps: bool = True):
        """
        args: 
            image_path (str): Image Path

        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        processed_images = self._preprocess_image(gray)
        # Whether to show conversion process or not
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show original
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show preprocessing result
        if processed_images:
            axes[1].imshow(processed_images[0], cmap='gray')
            axes[1].set_title('CLAHE Enhanced')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_bounding_boxes(self, image_path: str):
        """
        args:
            image_path (str): Image Path
        
        Visualize detected bounding boxes on the original image.
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get processed images and find regions
        processed_images = self._preprocess_image(gray)
        
        all_regions = []
        all_bounding_boxes = []
        
        for processed in processed_images:
            regions, bboxes = self._find_text_regions(processed, image)
            all_regions.extend(regions)
            all_bounding_boxes.extend(bboxes)
        
        # Remove overlapping regions
    
        for bbox in all_bounding_boxes:
            x, y, w, h = bbox
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Display the result
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Text Region Detection\nGreen: Detected regions')
        plt.axis('off')
        plt.show()
        
        return all_regions, filtered_bboxes # Return both regions and bboxes


# --- ROUPING ROTATED RECTANGLES for each word segment ---

def group_and_fit_rotated_rects(
    detected_individual_boxes: List[Tuple[int, int, int, int]], # Expects (x, y, w, h) tuples
    dbscan_eps_multiplier: float = 1.8,
    dbscan_min_samples: int = 4,
    expansion_width_px: int = 15,
    expansion_height_px: int = 10
) -> List[Tuple]:
    """
    Groups individual detected 4-sided shapes into text regions using DBSCAN,
    filters regions with less than 'min_samples', and fits expanded rotated rectangles.
    
    Args:
        detected_individual_boxes (List[Tuple[int, int, int, int]]): List of individual bounding boxes (x, y, w, h).
                                                                       These are expected to be the padded boxes from TextRegionExtractor.
        dbscan_eps_multiplier (float): Multiplier for average char size to determine DBSCAN's eps.
        dbscan_min_samples (int): Minimum number of shapes to form a valid text region.
                                   (e.g., 4 for "more than 3 consecutive")
        expansion_width_px (int): Absolute pixel expansion for the width of the final rectangle.
        expansion_height_px (int): Absolute pixel expansion for the height of the final rectangle.
                                         
    Returns:
        List[Tuple]: A list of final rotated rectangle tuples,
                     each in the format ((center_x, center_y), (width, height), angle).
    """
    if not detected_individual_boxes:
        print("No individual bounding boxes provided for grouping.")
        return []

    # Prepare data for DBSCAN (using center points of the *padded* boxes)
    points = np.array([[bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2] for bbox in detected_individual_boxes])

    # Calculate average dimensions for dynamic eps from the *padded* boxes
    average_box_width = np.mean([bbox[2] for bbox in detected_individual_boxes])
    average_box_height = np.mean([bbox[3] for bbox in detected_individual_boxes])
    
    eps_val = max(average_box_width, average_box_height) * dbscan_eps_multiplier
    
    print(f"\n--- Grouping Text Regions ---")
    print(f"DBSCAN parameters: eps={eps_val:.2f}, min_samples={dbscan_min_samples}")

    db = DBSCAN(eps=eps_val, min_samples=dbscan_min_samples).fit(points)
    labels = db.labels_

    grouped_regions = {}
    for i, label in enumerate(labels):
        if label == -1: # Noise (isolated characters or too small clusters)
            continue
        if label not in grouped_regions:
            grouped_regions[label] = []
        grouped_regions[label].append(detected_individual_boxes[i])

    print(f"Initial clusters found: {len(grouped_regions)} (before size filtering)")

    final_expanded_rotated_rectangles = []

    for label, bboxes_in_region in grouped_regions.items():
        if len(bboxes_in_region) < dbscan_min_samples:
            # Filter out clusters that are too small
            continue

        # Collect all 4 corner points for each bounding box in the current region
        # Use the corners of the *padded* bounding boxes for minAreaRect
        all_corners_in_region = []
        for bbox in bboxes_in_region:
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            all_corners_in_region.extend([
                [x1, y1], [x2, y1], [x1, y2], [x2, y2]
            ])
        
        if len(all_corners_in_region) < 3: # minAreaRect needs at least 3 points
            continue
        
        points_np = np.array(all_corners_in_region, dtype=np.float32)

        rect = cv2.minAreaRect(points_np)
        center, (width, height), angle = rect

        # Apply expansion
        new_width = width + 2 * expansion_width_px
        new_height = height + 2 * expansion_height_px

        expanded_rect = (center, (new_width, new_height), angle)
        final_expanded_rotated_rectangles.append(expanded_rect)
        
    print(f"Constructed {len(final_expanded_rotated_rectangles)} final expanded rotated rectangles.")
    return final_expanded_rotated_rectangles

def visualize_grouped_rectangles(
    image_path: str,
    individual_boxes: List[Tuple[int, int, int, int]], # Expects (x, y, w, h) tuples
    grouped_rotated_rects: List[Tuple]
):
    """
    Visualizes individual bounding boxes and the grouped, expanded, rotated rectangles.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image from {image_path}. Creating a blank image for visualization.")
        # Determine canvas size from boxes if no image provided
        max_x = max([b[0] + b[2] for b in individual_boxes]) + 100 if individual_boxes else 1000
        max_y = max([b[1] + b[3] for b in individual_boxes]) + 100 if individual_boxes else 800
        image = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        image.fill(200) # Light gray background
            
    result_image = image.copy()
    
    # Draw the individual filtered bounding boxes (green)
    for bbox in individual_boxes:
        x, y, w, h = bbox
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 1) # Green, thin

    # Draw the final expanded rotated rectangles (red)
    for rect in grouped_rotated_rects:
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points) # Convert to integer coordinates
        cv2.drawContours(result_image, [box_points], 0, (0, 0, 255), 2) # Red, thicker
    
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Text Region Grouping\nGreen: Individual shapes, Red: Grouped & Expanded Rotated Rectangles ({len(grouped_rotated_rects)} regions)')
    plt.axis('off')
    plt.show()

# --- NEW FUNCTION FOR EXTRACTING AND DESKEWING ROTATED RECTANGLES ---

def extract_and_deskew_rects(
    image_path: str,
    rotated_rects: List[Tuple],
    output_dir: str = "detected_regions_words"
) -> List[np.ndarray]:
    """
    Extracts regions defined by rotated rectangles from an image,
    deskews them to be horizontal, and saves them.
    
    Args:
        image_path (str): Path to the original image.
        rotated_rects (List[Tuple]): List of rotated rectangles in OpenCV format
                                     ((center_x, center_y), (width, height), angle).
        output_dir (str): Directory to save the deskewed images.
                                         
    Returns:
        List[np.ndarray]: A list of the extracted and deskewed image regions.
    """
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    extracted_deskewed_regions = []

    for i, rect in enumerate(rotated_rects):
        center, (width, height), angle = rect

        # Adjust angle to be between -45 and 45 degrees
        # OpenCV's minAreaRect returns an angle in [-90, 0)
        # If width < height, it means the rectangle is "taller" and the angle is closer to -90.
        
        if width < height:
            angle = angle + 90
            width, height = height, width
        
        # The angle is now in [-45, 45] (or something close).
        # We want to rotate by -angle to make it horizontal.
        
        # Get rotation matrix for deskewing
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box for the rotated image to avoid cropping
        # First, find the corners of the original image
        img_h, img_w = image.shape[0], image.shape[1]
        corners = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32)
        # Transform image corners
        abs_cos = abs(M[0,0])
        abs_sin = abs(M[0,1])
        bound_w = int(img_h * abs_sin + img_w * abs_cos)
        bound_h = int(img_h * abs_cos + img_w * abs_sin)

        # Update the translation part of the matrix to account for the new center
        M[0, 2] += bound_w / 2 - center[0]
        M[1, 2] += bound_h / 2 - center[1]

        # Perform the rotation on the entire image
        rotated_image = cv2.warpAffine(image, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Now, calculate the new bounding box for the extracted rectangle in the rotated_image
        # The center of the rectangle needs to be transformed by M as well
        new_center = tuple(np.array(M[:, :2] @ np.array(center).reshape(2, 1) + M[:, 2:]).flatten())
        
        # Create a new rotated rectangle for cropping purposes (angle is now 0)
        new_rect = (new_center, (width, height), 0)
        
        # Get the integer coordinates for the upright bounding box
        # This part requires careful handling for non-axis-aligned crop
        # We need the corners of the original rect, transform them, and then get the min/max x/y
        original_rect_pts = cv2.boxPoints(rect)
        transformed_pts = cv2.transform(original_rect_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        
        # Get the bounding box of the transformed points
        x_coords = transformed_pts[:, 0]
        y_coords = transformed_pts[:, 1]
        
        x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
        x_max, y_max = int(np.max(x_coords)), int(np.max(y_coords))

        # Crop the deskewed region
        cropped_region = rotated_image[y_min:y_max, x_min:x_max]

        if cropped_region.shape[0] > 0 and cropped_region.shape[1] > 0:
            extracted_deskewed_regions.append(cropped_region)
            filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_word_region_{i}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, cropped_region)
            print(f"Saved deskewed word region: {filepath}")
        else:
            print(f"Skipping empty or invalid cropped region {i} for {os.path.basename(image_path)}")

    print(f"Extracted and deskewed {len(extracted_deskewed_regions)} word regions.")
    return extracted_deskewed_regions


# --- Main Execution ---
if __name__ == "__main__":
    extractor = TextRegionExtractor()

    # Directory containing your images
    input_image_directory = "OCR"
    
    # Base output directory for all processed images' word regions
    base_output_words_dir = "detected_regions_words"
    os.makedirs(base_output_words_dir, exist_ok=True)

    # Define parameters for grouping and expansion (tune these!)
    GROUPING_EPS_MULTIPLIER = 2.5 # <--- TUNE THIS FOR YOUR IMAGES (e.g., 2.0, 2.5, 3.0, 3.5)
    MIN_SHAPES_IN_CHAIN = 3       # <--- TUNE THIS (e.g., 3 for shorter words/digits)
    EXPANSION_WIDTH_PX = 15       # Pixels to expand horizontally (each side)
    EXPANSION_HEIGHT_PX = 10      # Pixels to expand vertically (each side)

    # List of common image extensions to process
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    for filename in os.listdir(input_image_directory):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(input_image_directory, filename)
            image_name_without_ext = os.path.splitext(filename)[0]
            
            # Create a specific output subdirectory for each image's words
            current_image_output_words_dir = os.path.join(base_output_words_dir, image_name_without_ext)
            os.makedirs(current_image_output_words_dir, exist_ok=True)

            print(f"\n--- Processing image: {filename} ---")
            try:
                # Step 1: Use TextRegionExtractor to get individual character boxes
                # The extract_text_regions method returns regions and bboxes

                
                # Let's create a temporary directory for individual character regions if needed for debugging
                temp_individual_char_regions_dir = os.path.join("temp_individual_char_regions", image_name_without_ext)
                os.makedirs(temp_individual_char_regions_dir, exist_ok=True)

                _, individual_bboxes_from_extractor = extractor.extract_text_regions(image_path, temp_individual_regions_dir)
                print(f"Extracted {len(individual_bboxes_from_extractor)} individual regions using contour method for {filename}")
                
                # Visualize the detection process and bounding boxes (optional, can be commented out for batch processing)
                # These will open plot windows for each image.
                # extractor.visualize_detection(image_path)
                # _, _ = extractor.visualize_bounding_boxes(image_path)

                # Step 3: Call the new standalone function to group and fit rotated rectangles
                final_grouped_rectangles = group_and_fit_rotated_rects(
                    individual_bboxes_from_extractor,
                    dbscan_eps_multiplier=GROUPING_EPS_MULTIPLIER,
                    dbscan_min_samples=MIN_SHAPES_IN_CHAIN,
                    expansion_width_px=EXPANSION_WIDTH_PX,
                    expansion_height_px=EXPANSION_HEIGHT_PX
                )

                # Step 4: Visualize the results with the new grouping (optional for batch)
                visualize_grouped_rectangles(image_path, individual_bboxes_from_extractor, final_grouped_rectangles)

                # Step 5: Extract, Deskew, and Save the Red Rectangles
                deskewed_word_regions = extract_and_deskew_rects(image_path, final_grouped_rectangles, current_image_output_words_dir)
                print(f"Saved {len(deskewed_word_regions)} deskewed word regions for {filename} to '{current_image_output_words_dir}'.")

            except Exception as e:
                print(f"Error processing image {filename}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging

    print("\nBatch processing complete.")
    print(f"All deskewed word regions are saved in subdirectories within '{base_output_words_dir}'.")
    print("If you encountered errors, ensure 'OCR' directory exists and contains valid image files.")
    print("Also, check that scikit-learn and scikit-image are installed (`pip install scikit-learn scikit-image`).")