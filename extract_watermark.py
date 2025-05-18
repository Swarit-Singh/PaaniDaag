from watermark_operations import extract_watermark

def extract_watermark_from_image(wm_image, params):
    return extract_watermark(wm_image, params, params.get('method','prediction_error'))
