import numpy as np
# Assuming watermark_operations.py is in the same directory or PYTHONPATH
from watermark_operations import (
    embed_pred_error, extract_pred_error,
    embed_hist_shift, extract_hist_shift,
    embed_ml_assisted, extract_ml_assisted
)

def watermark_pipeline(cover_img_np: np.ndarray,
                       payload_bits: list, # Expects a list of ints (0 or 1)
                       method: str,
                       **kwargs): # For method-specific params like T for PEE
    """
    Orchestrates the embedding and extraction process.

    Returns:
        (watermarked_image_np, recovered_image_np, extracted_bits_np_array, embedding_params_dict)
    """
    original_payload_len = len(payload_bits)
    embedding_params = {} # To store parameters from embedding step

    if method == "prediction_error":
        pee_T_val = kwargs.get('T', 1) # Default T=1 if not provided
        watermarked_image, op_params = embed_pred_error(cover_img_np, payload_bits, T=pee_T_val)
        embedding_params.update(op_params)
        # Extraction needs the original payload length to know how many bits to read
        extracted_bits_list, recovered_image = extract_pred_error(watermarked_image, original_payload_len, T=pee_T_val)

    elif method == "histogram_shift":
        watermarked_image, op_params = embed_hist_shift(cover_img_np, payload_bits)
        embedding_params.update(op_params)
        # HS extraction needs parameters (like peak/zero) determined during embedding
        peak_val = embedding_params.get("peak_val")
        zero_val = embedding_params.get("zero_val")
        if peak_val is None or zero_val is None: # Should be set by embed_hist_shift
            raise ValueError("Histogram shifting parameters (peak/zero) not found after embedding.")
        extracted_bits_list, recovered_image = extract_hist_shift(watermarked_image, original_payload_len, peak_val=peak_val, zero_val=zero_val)

    elif method == "ml_assisted":
        # ML-Assisted placeholder (currently PEE T=0)
        watermarked_image, op_params = embed_ml_assisted(cover_img_np, payload_bits)
        embedding_params.update(op_params) # Will include T=0
        extracted_bits_list, recovered_image = extract_ml_assisted(watermarked_image, original_payload_len)
        embedding_params['info'] = "ML-Assisted currently uses PEE with T=0 as a placeholder."

    else:
        raise ValueError(f"Unsupported watermarking method specified: {method}")

    return watermarked_image, recovered_image, np.array(extracted_bits_list, dtype=np.uint8), embedding_params
