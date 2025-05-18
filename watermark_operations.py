import numpy as np

# --- Prediction Error Expansion (PEE) ---
def _pee_embed_core(cover_img: np.ndarray, bits_to_embed: list, threshold_T: int):
    rows, cols = cover_img.shape
    watermarked_img = cover_img.astype(np.int32).copy()
    bits_embedded_count = 0
    payload_len = len(bits_to_embed)

    for r in range(1, rows):
        for c in range(1, cols):
            if bits_embedded_count >= payload_len: break
            
            prediction = int(watermarked_img[r, c - 1])
            error = int(cover_img[r, c]) - prediction
            
            modified_error = error
            if -threshold_T <= error < threshold_T:
                bit_to_embed = bits_to_embed[bits_embedded_count]
                modified_error = 2 * error + bit_to_embed
                bits_embedded_count += 1
            elif error >= threshold_T:
                modified_error = error + threshold_T
            elif error < -threshold_T:
                modified_error = error - threshold_T
            
            watermarked_img[r, c] = prediction + modified_error
        if bits_embedded_count >= payload_len: break
            
    final_wm_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
    op_params = {
        "method_used": "PEE",
        "T_threshold": threshold_T,
        "bits_actually_embedded": bits_embedded_count,
        "requested_payload_length": payload_len
    }
    return final_wm_img, op_params

def _pee_extract_core(watermarked_img: np.ndarray, original_payload_len: int, threshold_T: int):
    rows, cols = watermarked_img.shape
    recovered_img = watermarked_img.astype(np.int32).copy()
    extracted_bits = []
    
    for r in range(1, rows):
        for c in range(1, cols):
            if len(extracted_bits) >= original_payload_len: break
            
            prediction = int(recovered_img[r, c - 1])
            modified_error = int(watermarked_img[r, c]) - prediction
            original_error = modified_error

            if -2 * threshold_T <= modified_error < 2 * threshold_T:
                bit = modified_error % 2
                original_error = (modified_error - bit) // 2
                extracted_bits.append(bit)
            elif modified_error >= 2 * threshold_T:
                original_error = modified_error - threshold_T
            elif modified_error < -2 * threshold_T:
                original_error = modified_error + threshold_T
            
            recovered_img[r, c] = prediction + original_error
        if len(extracted_bits) >= original_payload_len: break
            
    final_rec_img = np.clip(recovered_img, 0, 255).astype(np.uint8)
    return np.array(extracted_bits, dtype=np.uint8), final_rec_img

def embed_pred_error(cover_img: np.ndarray, bits_to_embed: list, T: int = 1):
    return _pee_embed_core(cover_img, bits_to_embed, threshold_T=T)

def extract_pred_error(watermarked_img: np.ndarray, original_payload_len: int, T: int = 1):
    return _pee_extract_core(watermarked_img, original_payload_len, threshold_T=T)

# --- Histogram Shifting (HS) - Simplified for Demo ---
def _find_hs_params(hist):
    peak_val = int(np.argmax(hist))
    zero_val = -1
    sorted_indices = np.argsort(hist)
    for idx in sorted_indices:
        if idx != peak_val:
            zero_val = int(idx)
            break
    if zero_val == -1: zero_val = (peak_val + 1) % 256
    
    p, z = min(peak_val, zero_val), max(peak_val, zero_val)
    if p == z : p = 0; z = 1
    return p, z

def embed_hist_shift(cover_img: np.ndarray, bits_to_embed: list):
    hist, _ = np.histogram(cover_img.ravel(), bins=256, range=(0,255))
    peak_p, zero_z = _find_hs_params(hist)

    watermarked_img = cover_img.astype(np.int32).copy()
    bits_embedded_count = 0
    payload_len = len(bits_to_embed)

    watermarked_img[(cover_img > peak_p) & (cover_img < zero_z)] += 1
    
    original_peak_locations_r, original_peak_locations_c = np.where(cover_img == peak_p)
    
    for i in range(len(original_peak_locations_r)):
        if bits_embedded_count >= payload_len: break
        r, c = original_peak_locations_r[i], original_peak_locations_c[i]
        bit_to_embed = bits_to_embed[bits_embedded_count]
        if bit_to_embed == 1:
            watermarked_img[r,c] = peak_p + 1
        bits_embedded_count += 1
        
    final_wm_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
    op_params = {
        "method_used": "HS",
        "peak_val": peak_p, "zero_val": zero_z,
        "bits_actually_embedded": bits_embedded_count,
        "requested_payload_length": payload_len
    }
    return final_wm_img, op_params

def extract_hist_shift(watermarked_img: np.ndarray, original_payload_len: int, peak_val: int, zero_val: int):
    p, z = min(peak_val, zero_val), max(peak_val, zero_val)
    recovered_img = watermarked_img.astype(np.int32).copy()
    extracted_bits = []
    
    rows, cols = watermarked_img.shape
    for r_idx in range(rows):
        for c_idx in range(cols):
            if len(extracted_bits) >= original_payload_len: break
            current_pixel_val = watermarked_img[r_idx, c_idx]
            if current_pixel_val == p:
                extracted_bits.append(0)
            elif current_pixel_val == p + 1:
                extracted_bits.append(1)
                recovered_img[r_idx, c_idx] = p
        if len(extracted_bits) >= original_payload_len: break
    
    while len(extracted_bits) < original_payload_len:
        extracted_bits.append(0)

    recovered_img[(watermarked_img > p + 1) & (watermarked_img <= z)] -=1
    final_rec_img = np.clip(recovered_img, 0, 255).astype(np.uint8)
    return np.array(extracted_bits[:original_payload_len], dtype=np.uint8), final_rec_img

# --- ML-Assisted (Updated Placeholder: PEE T=1 for demo purposes) ---
def embed_ml_assisted(cover_img: np.ndarray, bits_to_embed: list, **kwargs):
    final_wm_img, op_params = _pee_embed_core(cover_img, bits_to_embed, threshold_T=1) 
    op_params["method_used"] = "ML-Assisted (PEE T=1)" # Specific label for this stub
    return final_wm_img, op_params

def extract_ml_assisted(watermarked_img: np.ndarray, original_payload_len: int, **kwargs):
    return _pee_extract_core(watermarked_img, original_payload_len, threshold_T=1)

# --- Prediction Error Calculation for Capacity/Hybrid Methods ---
def compute_prediction_errors(img, patch_size=3):
    """
    Returns the prediction error image using the left neighbor predictor.
    (Replace prediction logic here if using ML predictor.)
    """
    H, W = img.shape
    pe = np.zeros_like(img, dtype=int)
    for r in range(1, H):
        for c in range(1, W):
            pred = int(img[r, c - 1])
            pe[r, c] = int(img[r, c]) - pred
    return pe
