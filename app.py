import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import io

from watermark_pipeline import watermark_pipeline
from watermark_quality_test import test_single_watermarking_scenario
from visualization_utils import (
    plot_histograms_styled,
    plot_difference_styled,
    plot_comparative_bar
)
from skimage.metrics import (
    peak_signal_noise_ratio as psnr_metric,
    structural_similarity as ssim_metric
)

st.set_page_config(
    page_title="Comprehensive Reversible Watermarking Toolkit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Comprehensive Reversible Watermarking Toolkit")
st.markdown("""
    Embed patient data or custom binary payloads into medical images using various reversible watermarking techniques.
    Analyze performance, visualize effects, and test robustness.
""")
st.markdown("---")

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'payload_str_for_display' not in st.session_state:
    st.session_state.payload_str_for_display = "101101001110001010101011"
if 'payload_bits_list_final' not in st.session_state:
    st.session_state.payload_bits_list_final = []
if 'active_method_params' not in st.session_state:
    st.session_state.active_method_params = {}
if 'pee_T_slider_value' not in st.session_state:
    st.session_state.pee_T_slider_value = 2

def add_visible_text_overlay(img_np, text, font_size, color_hex, opacity_val):
    if not text: return img_np
    pil_img = Image.fromarray(img_np.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        st.warning("Arial font not found. Using default.", icon="⚠️")
    draw = ImageDraw.Draw(overlay)
    bbox = draw.textbbox((0,0), text, font=font, anchor="lt")
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    padding = 10
    pos_x = pil_img.width - text_w - padding if pil_img.width > text_w + 2*padding else padding
    pos_y = pil_img.height - text_h - padding if pil_img.height > text_h + 2*padding else padding
    rgb_color = ImageColor.getrgb(color_hex)
    draw.text((pos_x, pos_y), text, font=font, fill=(*rgb_color, opacity_val))
    final_img = Image.alpha_composite(pil_img, overlay)
    return np.array(final_img.convert("L"))

def parse_binary_string_to_bits(binary_str):
    return [int(b) for b in binary_str.strip() if b in ('0', '1')]

def calculate_pee_capacity(img_np, T_val):
    capacity = 0
    rows, cols = img_np.shape
    for r in range(1, rows):
        for c in range(1, cols):
            prediction = int(img_np[r, c-1])
            error = int(img_np[r, c]) - prediction
            if -T_val <= error < T_val:
                capacity += 1
    return capacity

def calculate_hs_capacity(img_np):
    hist, _ = np.histogram(img_np.ravel(), bins=256, range=(0,255))
    return int(np.max(hist))

def calculate_ml_stub_capacity(img_np):
    # For ML-assisted (HS+PEE+XGBoost), capacity is the peak of the prediction error histogram
    from watermark_operations import compute_prediction_errors
    pe = compute_prediction_errors(img_np)
    pe_flat = pe.flatten()
    min_pe, max_pe = np.min(pe_flat), np.max(pe_flat)
    bins = np.arange(min_pe, max_pe+2)
    hist, bin_edges = np.histogram(pe_flat, bins=bins)
    peak_idx = np.argmax(hist)
    return hist[peak_idx]

with st.sidebar:
    st.header("⚙️ Controls & Patient Data")

    uploaded_file = st.file_uploader("1. Upload Grayscale Image:", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    
    st.markdown("---")
    st.subheader("2. Watermarking Method")
    method_selected = st.selectbox("Select Method:", ("prediction_error", "histogram_shift", "ml_assisted"))

    st.session_state.active_method_params = {} 
    if method_selected == "prediction_error":
        st.session_state.pee_T_slider_value = st.slider(
            "PEE Threshold (T):", 1, 5, 
            st.session_state.pee_T_slider_value, 
            key="pee_T_slider_main_persistent" 
        )
        st.session_state.active_method_params['T'] = st.session_state.pee_T_slider_value

    st.markdown("---")
    st.subheader("3. Payload Configuration")
    st.markdown("**Patient Information (Optional Payload Source)**")
    p_name = st.text_input("Patient Name:", key="p_name")
    p_id = st.text_input("Patient ID:", key="p_id")
    p_age = st.text_input("Age:", key="p_age")
    p_gender = st.selectbox("Gender:", ["Male", "Female", "Other", "Not Specified"], key="p_gender")
    p_diagnosis = st.text_area("Brief Diagnosis:", key="p_diagnosis", height=100)

    if st.button("Generate Binary Payload from Patient Info", key="gen_payload_btn"):
        if not p_name and not p_id:
            st.warning("Please enter at least Patient Name or ID to generate payload.")
        else:
            patient_info_string = f"Name:{p_name};ID:{p_id};Age:{p_age};Gender:{p_gender};Diagnosis:{p_diagnosis}"
            generated_bits_str = ''.join(format(ord(char), '08b') for char in patient_info_string)
            st.session_state.payload_str_for_display = generated_bits_str
            st.success(f"Payload generated ({len(generated_bits_str)} bits). Review/edit below.")

    payload_input_from_user = st.text_area(
        "Binary Payload (Edit or Paste):",
        value=st.session_state.payload_str_for_display,
        height=100,
        key="payload_text_area_main_controlled"
    )
    if payload_input_from_user != st.session_state.payload_str_for_display:
        st.session_state.payload_str_for_display = payload_input_from_user

    st.markdown("---")
    st.subheader("4. Visible Overlay (Optional)")
    apply_visible_wm = st.checkbox("Add Visible Text Overlay", key="apply_visible_checkbox")
    if apply_visible_wm:
        visible_wm_text = st.text_input("Overlay Text:", "CONFIDENTIAL", key="visible_text_input")
        visible_font_s = st.slider("Font Size:", 10, 100, 40, key="visible_font_slider")
        visible_color_hex = st.color_picker("Font Color:", "#FAFAFA", key="visible_color_picker")
        visible_opac = st.slider("Opacity (0-255):", 0, 255, 128, key="visible_opacity_slider")

    st.markdown("---")
    st.header("🚀 Execute")
    process_button = st.button("Embed, Extract & Analyze")
    
    st.markdown("---")
    st.subheader("🧪 Post-Processing Tests")
    show_robustness_tests_cb = st.checkbox("Run Robustness Tests (JPEG, Noise)", value=False)
    show_comparison_plots_cb = st.checkbox("Show Method Comparison Graphs (Actual Data)", value=True)

if uploaded_file is not None:
    original_pil_img = Image.open(uploaded_file).convert("L")
    cover_image_np = np.array(original_pil_img)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Estimated Embedding Capacity:")
    current_payload_to_analyze = parse_binary_string_to_bits(st.session_state.payload_str_for_display)
    estimated_capacity = 0
    
    T_for_pee_capacity = st.session_state.pee_T_slider_value if method_selected == "prediction_error" else 1 

    if method_selected == "prediction_error":
        estimated_capacity = calculate_pee_capacity(cover_image_np, T_for_pee_capacity)
        st.sidebar.info(f"For PEE (T={T_for_pee_capacity}): ~{estimated_capacity} bits")
    elif method_selected == "histogram_shift":
        estimated_capacity = calculate_hs_capacity(cover_image_np)
        st.sidebar.info(f"For HS: ~{estimated_capacity} bits")
    elif method_selected == "ml_assisted":
        estimated_capacity = calculate_ml_stub_capacity(cover_image_np)
        st.sidebar.info(f"For ML-Assisted (PEE+HS+XGBoost): ~{estimated_capacity} bits")
    
    if len(current_payload_to_analyze) > estimated_capacity and estimated_capacity > 0 :
        st.sidebar.warning(f"Current payload ({len(current_payload_to_analyze)} bits) exceeds capacity. It will be truncated to {estimated_capacity} bits if you proceed.")
    elif len(current_payload_to_analyze) > 0 and estimated_capacity == 0:
        st.sidebar.error(f"Estimated capacity for '{method_selected}' is 0. Cannot embed payload of {len(current_payload_to_analyze)} bits.")

    if process_button:
        st.session_state.analysis_done = False
        
        payload_bits_to_embed = parse_binary_string_to_bits(st.session_state.payload_str_for_display)

        if not payload_bits_to_embed:
            st.error("❌ Payload is empty or invalid. Please enter a valid binary string or generate from patient info.")
        elif estimated_capacity == 0 and len(payload_bits_to_embed) > 0:
             st.error(f"❌ Cannot embed. Estimated capacity for '{method_selected}' is 0, but payload has {len(payload_bits_to_embed)} bits.")
        else:
            if len(payload_bits_to_embed) > estimated_capacity:
                payload_bits_to_embed = payload_bits_to_embed[:estimated_capacity]
                st.session_state.payload_str_for_display = "".join(map(str, payload_bits_to_embed))
                st.warning(f"Payload was truncated to {len(payload_bits_to_embed)} bits to fit capacity. The text area has been updated.")

            st.session_state.payload_bits_list_final = payload_bits_to_embed

            image_to_process_np = cover_image_np.copy()
            if apply_visible_wm and 'visible_wm_text' in locals() and visible_wm_text:
                image_to_process_np = add_visible_text_overlay(
                    image_to_process_np, visible_wm_text, visible_font_s,
                    visible_color_hex, visible_opac
                )
                st.info(f"Applied visible overlay: '{visible_wm_text}'. Further metrics are based on this overlaid image.")
            
            st.session_state.processed_input_image_np = image_to_process_np

            with st.spinner(f"Processing with '{method_selected}'... Embedding {len(payload_bits_to_embed)} bits."):
                try:
                    (wm_result_np, rec_result_np,
                     extracted_bits_result, embed_params_result) = watermark_pipeline(
                        image_to_process_np,
                        payload_bits_to_embed,
                        method_selected,
                        **st.session_state.active_method_params 
                    )
                    st.session_state.watermarked_image_np = wm_result_np
                    st.session_state.recovered_image_np = rec_result_np
                    st.session_state.extracted_bits_list = extracted_bits_result
                    st.session_state.embedding_parameters = embed_params_result
                    st.session_state.analysis_done = True
                    st.success(f"✅ Pipeline completed for '{method_selected}'!")

                except Exception as e:
                    st.error(f"Pipeline Error for '{method_selected}': {str(e)}")
                    st.exception(e) 
                    st.session_state.analysis_done = False
else:
    st.info("👋 Please upload a grayscale image to begin.")

if st.session_state.get('analysis_done', False):
    input_img_disp = st.session_state.processed_input_image_np
    watermarked_img_disp = st.session_state.watermarked_image_np
    recovered_img_disp = st.session_state.recovered_image_np
    original_payload_disp = st.session_state.payload_bits_list_final
    extracted_payload_disp = st.session_state.extracted_bits_list
    embed_params_disp = st.session_state.embedding_parameters

    st.markdown("---")
    st.header("🖼️ Image Results")
    img_cols_disp = st.columns(3)
    img_cols_disp[0].image(input_img_disp, caption="Input Image (Processed)", use_container_width=True)
    img_cols_disp[1].image(watermarked_img_disp, caption="Watermarked Image", use_container_width=True)
    img_cols_disp[2].image(recovered_img_disp, caption="Recovered Image", use_container_width=True)

    st.markdown("---")
    st.header("📊 Metrics & Extraction Details")
    
    payload_len_disp = len(original_payload_disp)
    extracted_len_disp = len(extracted_payload_disp)
    bits_actually_embedded_disp = embed_params_disp.get('bits_actually_embedded', 'N/A')

    correctly_extracted_count_disp = 0
    min_comp_len_disp = min(payload_len_disp, extracted_len_disp)
    if min_comp_len_disp > 0 :
        correctly_extracted_count_disp = sum(1 for i in range(min_comp_len_disp) if original_payload_disp[i] == extracted_payload_disp[i])
    
    accuracy_val_disp = (correctly_extracted_count_disp / payload_len_disp * 100) if payload_len_disp > 0 else \
                        (100.0 if extracted_len_disp == 0 and payload_len_disp == 0 else 0.0)

    metric_cols_disp = st.columns(3)
    metric_cols_disp[0].metric("Intended Payload Bits (After Truncation):", payload_len_disp)
    metric_cols_disp[0].metric("Bits Actually Embedded (by method):", bits_actually_embedded_disp)
    metric_cols_disp[0].metric("Extracted Bits:", extracted_len_disp)
    metric_cols_disp[0].metric("Extraction Accuracy:", f"{accuracy_val_disp:.2f}%")

    pixels_changed_val_disp = int(np.count_nonzero(input_img_disp != watermarked_img_disp))
    psnr_val_disp = float('inf') if pixels_changed_val_disp == 0 else \
                    psnr_metric(input_img_disp, watermarked_img_disp, data_range=255)
    metric_cols_disp[1].metric("Pixels Changed (Input vs WM):", pixels_changed_val_disp)
    metric_cols_disp[1].metric("PSNR (Input vs WM):", "∞" if psnr_val_disp == float('inf') else f"{psnr_val_disp:.2f} dB")

    min_img_dim_disp = min(input_img_disp.shape[:2])
    win_size_val_disp = 7 if min_img_dim_disp >= 7 else max(3, (min_img_dim_disp // 2 * 2 + 1))
    ssim_val_disp = ssim_metric(input_img_disp, watermarked_img_disp, data_range=255, channel_axis=None, win_size=win_size_val_disp)
    metric_cols_disp[2].metric("SSIM (Input vs WM):", f"{ssim_val_disp:.4f}")
    
    # perfect_recovery_disp = np.array_equal(input_img_disp, recovered_img_disp)
    # metric_cols_disp[2].metric("Perfect Image Recovery:", "✅ Yes" if perfect_recovery_disp else "❌ No")

    with st.expander("Embedding Parameters & Details", expanded=False):
        st.json(embed_params_disp if embed_params_disp else {"info": "No specific parameters returned."})
        display_n_bits_debug = 32
        st.markdown(f"**Final Original Payload (first {display_n_bits_debug}):** `{''.join(map(str, original_payload_disp[:display_n_bits_debug]))}...`")
        st.markdown(f"**Extracted Payload (first {display_n_bits_debug}):** `{''.join(map(str, extracted_payload_disp[:display_n_bits_debug]))}...`")

    st.markdown("---")
    st.header("📉 Visualizations")
    plot_cols_disp = st.columns(2)
    with plot_cols_disp[0]:
        st.subheader("Histograms")
        try:
            hist_fig_disp = plot_histograms_styled(input_img_disp, watermarked_img_disp)
            st.pyplot(hist_fig_disp, clear_figure=True)
        except Exception as e: st.error(f"Histogram plot failed: {e}")
    with plot_cols_disp[1]:
        st.subheader("Difference Map")
        try:
            diff_fig_disp = plot_difference_styled(input_img_disp, watermarked_img_disp)
            st.pyplot(diff_fig_disp, clear_figure=True)
        except Exception as e: st.error(f"Difference map failed: {e}")

    st.markdown("---")
    st.header("💾 Download Outputs")
    dl_cols_disp = st.columns(2)
    wm_pil_disp = Image.fromarray(watermarked_img_disp.astype(np.uint8))
    buf_wm_disp = io.BytesIO()
    wm_pil_disp.save(buf_wm_disp, format="PNG")
    dl_cols_disp[0].download_button(
        label="Download Watermarked Image", data=buf_wm_disp.getvalue(),
        file_name=f"watermarked_{method_selected}.png", mime="image/png", use_container_width=True
    )
    rec_pil_disp = Image.fromarray(recovered_img_disp.astype(np.uint8))
    buf_rec_disp = io.BytesIO()
    rec_pil_disp.save(buf_rec_disp, format="PNG")
    dl_cols_disp[1].download_button(
        label="Download Recovered Image", data=buf_rec_disp.getvalue(),
        file_name=f"recovered_{method_selected}.png", mime="image/png", use_container_width=True
    )

    if show_robustness_tests_cb:
        st.markdown("---")
        st.header("🛡️ Robustness Test Results")
        with st.spinner("Running robustness tests... This may take a moment."):
            robustness_payload = st.session_state.payload_bits_list_final 
            robustness_results_disp = test_single_watermarking_scenario(
                input_img_disp, 
                robustness_payload,
                method_selected,
                **st.session_state.active_method_params 
            )
        st.metric("JPEG Robustness (Accuracy @75Q):", f"{robustness_results_disp.get('acc_jpeg', 0.0)*100:.2f}%")
        st.metric("Gaussian Noise Robustness (Accuracy @σ=5):", f"{robustness_results_disp.get('acc_noise', 0.0)*100:.2f}%")
        with st.expander("Full Robustness Data Details", expanded=True):
            st.json(robustness_results_disp)

    if show_comparison_plots_cb:
        st.markdown("---")
        st.header("📊 Method Comparison (Actual Uploaded Data)")
        st.info("These graphs use the actual uploaded image. Capacity bars show the *maximum* embeddable bits for each method.", icon="🔬")

        methods = ["prediction_error", "histogram_shift", "ml_assisted"]
        method_labels = ["PEE", "HS", "ML-Assisted"]
        psnr_list, ssim_list, acc_list = [], [], []
        capacity_list = []

    for m in methods:
        # Calculate true capacity for each method (using the full image)
        if m == "prediction_error":
            cap = calculate_pee_capacity(input_img_disp, st.session_state.pee_T_slider_value)
        elif m == "histogram_shift":
            cap = calculate_hs_capacity(input_img_disp)
        elif m == "ml_assisted":
            cap = calculate_ml_stub_capacity(input_img_disp)
        else:
            cap = 0
        capacity_list.append(cap)

        # For PSNR/SSIM/Accuracy, you can still run the pipeline with the truncated payload
        params = {}
        if m == "prediction_error":
            params['T'] = st.session_state.pee_T_slider_value
        try:
            wm_img, rec_img, ext_bits, op_params = watermark_pipeline(
                input_img_disp, original_payload_disp, m, **params
            )
            psnr = psnr_metric(input_img_disp, wm_img, data_range=255)
            ssim = ssim_metric(input_img_disp, wm_img, data_range=255)
            acc = (sum(1 for i in range(min(len(original_payload_disp), len(ext_bits)))
                    if original_payload_disp[i] == ext_bits[i]) / len(original_payload_disp) * 100
                   ) if len(original_payload_disp) > 0 else 0.0
        except Exception as e:
            psnr, ssim, acc = 0, 0, 0
            st.warning(f"{m} failed: {e}")
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        acc_list.append(acc)

    comp_cols_disp = st.columns(2)
    with comp_cols_disp[0]:
        fig_psnr = plot_comparative_bar(
            method_labels, psnr_list, metric_name="PSNR (dB)", title_extra="(Uploaded Image)"
        )
        st.pyplot(fig_psnr, clear_figure=True)
    with comp_cols_disp[1]:
        fig_cap = plot_comparative_bar(
            method_labels, capacity_list, metric_name="Capacity (bits)", title_extra="(Max possible for uploaded image)"
        )
        st.pyplot(fig_cap, clear_figure=True)

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:small; color:#777;">
      <strong>Comprehensive Reversible Watermarking Toolkit</strong> | Enhanced Demo
    </div>
    """, unsafe_allow_html=True
)
