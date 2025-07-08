import cv2
import easyocr
import os
import matplotlib.pyplot as plt
import textwrap
import numpy as np
import torch # To check for CUDA availability for EasyOCR

# --- Configuration ---
original_images_dir = "OCR" 
processed_words_dir = "detected_regions_words" # Or "final_refined_word_regions"

easyocr_languages = ['en'] 
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

# --- Initialize EasyOCR Reader (do this once) ---
print(f"Initializing EasyOCR reader for languages: {easyocr_languages}...")
reader = easyocr.Reader(easyocr_languages, gpu=torch.cuda.is_available()) 
print(f"EasyOCR reader initialized. Using GPU: {torch.cuda.is_available()}")

print(f"Processing images from: {processed_words_dir}")

# --- Iterate through each subfolder (representing an original image) ---
for original_image_base_name in os.listdir(processed_words_dir):
    current_processed_image_folder = os.path.join(processed_words_dir, original_image_base_name)

    if os.path.isdir(current_processed_image_folder):
        print(f"\n--- Processing folder for: {original_image_base_name} ---")

        original_image_path = None
        for ext in image_extensions:
            potential_path = os.path.join(original_images_dir, original_image_base_name + ext)
            if os.path.exists(potential_path):
                original_image_path = potential_path
                break
        
        if original_image_path is None:
            print(f"  Warning: Original image '{original_image_base_name}' not found in '{original_images_dir}'. Skipping.")
            continue

        original_image = cv2.imread(original_image_path)
        if original_image is None:
            print(f"  Error: Could not load original image '{original_image_path}'. Skipping.")
            continue
        
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        concatenated_easyocr_results = []
        word_image_files = sorted([f for f in os.listdir(current_processed_image_folder) if f.lower().endswith(image_extensions)])
        
        if not word_image_files:
            print(f"  No word images found in {current_processed_image_folder}. Skipping OCR for this folder.")
            concatenated_easyocr_results.append("No processed word regions found.")
        else:
            print(f"  Running OCR on {len(word_image_files)} word regions...")
            for word_img_filename in word_image_files:
                word_img_path = os.path.join(current_processed_image_folder, word_img_filename)
                word_img = cv2.imread(word_img_path)

                if word_img is None:
                    print(f"    Warning: Could not load word image '{word_img_filename}'. Skipping OCR for it.")
                    continue
                
                if word_img.shape[0] == 0 or word_img.shape[1] == 0:
                    print(f"    Warning: Empty image '{word_img_filename}'. Skipping preprocessing and OCR.")
                    continue

                # --- Preprocessing (applied before both OCR attempts) ---
                # 1. Convert to grayscale
                if len(word_img.shape) == 3:
                    preprocessed_img_base = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
                else:
                    preprocessed_img_base = word_img # Already grayscale

                # 2. Apply CLAHE for mild contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4)) 
                preprocessed_img_base = clahe.apply(preprocessed_img_base)
                
                # --- Attempt OCR on original orientation ---
                results_original = reader.readtext(preprocessed_img_base)
                text_original = [text.upper() for (bbox, text, prob) in results_original]
                # Calculate total confidence for this orientation (sum of probabilities)
                conf_original = sum([prob for (bbox, text, prob) in results_original])

                # --- Attempt OCR on 180-degree rotated orientation ---
                rotated_img_180 = cv2.rotate(preprocessed_img_base, cv2.ROTATE_180)
                results_rotated = reader.readtext(rotated_img_180)
                text_rotated = [text.upper() for (bbox, text, prob) in results_rotated]
                # Calculate total confidence for rotated orientation
                conf_rotated = sum([prob for (bbox, text, prob) in results_rotated])

                # --- Compare confidences and choose the best result ---
                if conf_rotated > conf_original:
                    # If rotated image yields higher confidence, use its results
                    extracted_texts = text_rotated
                    # print(f"    '{word_img_filename}' - Chose 180-degree rotation (conf: {conf_rotated:.2f} vs {conf_original:.2f})")
                else:
                    # Otherwise, use the original orientation's results (or if confidences are equal)
                    extracted_texts = text_original
                    # print(f"    '{word_img_filename}' - Chose original orientation (conf: {conf_original:.2f} vs {conf_rotated:.2f})")

                if extracted_texts:
                    concatenated_easyocr_results.extend(extracted_texts)
                # else:
                    # Uncomment for more verbose logging if you want to see which images yield no text
                    # print(f"    No text detected in {word_img_filename} in either orientation.")

        final_text_output = "\n".join(concatenated_easyocr_results)
        if not final_text_output:
            final_text_output = "No text detected in any region for this image."

        # 3. Display the original image and concatenated results in Jupyter
        plt.figure(figsize=(18, 10)) # Adjust figure size as needed

        # Subplot for the original image
        ax1 = plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
        ax1.imshow(original_image_rgb)
        ax1.set_title(f'Original Image: {original_image_base_name}')
        ax1.axis('off')

        # Subplot for the EasyOCR results
        ax2 = plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
        ax2.set_title(f'EasyOCR Results for {original_image_base_name}')
        ax2.axis('off') # Hide axes for text display

        # Wrap text for better readability if it's too long
        wrapped_text = textwrap.fill(final_text_output, width=50) # Adjust width (characters per line) as needed
        
        # Display text. Adjust x, y for position within subplot (0,0 is bottom-left, 1,1 is top-right)
        ax2.text(0.05, 0.95, wrapped_text, 
                 transform=ax2.transAxes, # Position relative to the axes
                 fontsize=12, 
                 verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    else:
        print(f"Skipping non-directory item: {current_processed_image_folder}")

print("\n--- EasyOCR processing and display complete ---")
