"""
DNA Data Storage Platform
A user-friendly platform for encoding, randomizing, decoding, and comparing DNA-based data storage.
"""

import streamlit as st
import time
import os
from io import BytesIO

# Import custom modules
from dna_codec import (
    encode_to_dna, decode_from_dna, validate_dna_sequence,
    calculate_encoding_density
)
from compression import (
    detect_file_type, compress_data, decompress_data,
    get_compression_stats, TEXT_EXTENSIONS, IMAGE_EXTENSIONS, PDF_EXTENSIONS,
    AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, get_pdf_info, get_audio_info, get_audio_compression_suggestion,
    get_video_info, get_video_compression_suggestion
)
from randomization import (
    calculate_dna_characteristics, randomize_dna, derandomize_dna,
    verify_randomization, calculate_randomization_improvement,
    ChaosSystem, get_chaos_system_info, list_chaos_systems,
    get_chain_description, get_chain_complexity
)
from comparison import (
    compare_data, format_comparison_results, detect_comparison_type
)


# Page configuration
st.set_page_config(
    page_title="DNA Data Storage Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .dna-preview {
        font-family: 'Courier New', monospace;
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 5px;
        word-break: break-all;
        max-height: 300px;
        overflow-y: auto;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #155724;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b8daff;
        padding: 1rem;
        border-radius: 5px;
        color: #004085;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'encoded_dna' not in st.session_state:
        st.session_state.encoded_dna = None
    if 'encoded_metadata' not in st.session_state:
        st.session_state.encoded_metadata = {}
    if 'randomized_dna' not in st.session_state:
        st.session_state.randomized_dna = None
    if 'decoded_data' not in st.session_state:
        st.session_state.decoded_data = None
    if 'decoded_metadata' not in st.session_state:
        st.session_state.decoded_metadata = {}
    if 'original_filename' not in st.session_state:
        st.session_state.original_filename = None
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'example_encoded' not in st.session_state:
        st.session_state.example_encoded = False
    if 'ngs_fragments' not in st.session_state:
        st.session_state.ngs_fragments = None
    if 'ngs_settings' not in st.session_state:
        st.session_state.ngs_settings = {}


def main():
    """Main application."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧬 DNA Data Storage Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Store any data format in DNA sequences with compression, randomization, and NGS preparation</p>', unsafe_allow_html=True)
    
    # Mode selection tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Encode", 
        "🔀 Randomization", 
        "🧫 NGS Prep",
        "📥 Decode", 
        "📊 Comparison",
        "📚 Guide"
    ])
    
    with tab1:
        encode_mode()
    
    with tab2:
        randomization_mode()
    
    with tab3:
        ngs_preparation_mode()
    
    with tab4:
        decode_mode()
    
    with tab5:
        comparison_mode()
    
    with tab6:
        guide_mode()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888;">DNA Data Storage Platform v1.0 | '
        'Encoding density: 2 bits/nucleotide</p>', 
        unsafe_allow_html=True
    )


def encode_mode():
    """Encode mode: Upload file and encode to DNA sequence."""
    st.header("📤 Encode Mode")
    st.markdown("Upload a file and encode it into a DNA sequence.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a file (any format)",
            type=None,
            key="encode_upload",
            help="Supported: Text, Images, and any binary files"
        )
        
        if uploaded_file:
            file_data = uploaded_file.read()
            filename = uploaded_file.name
            file_ext = os.path.splitext(filename)[1].lower()
            file_type = detect_file_type(filename, file_data)
            
            # Store original data for comparison
            st.session_state.original_data = file_data
            st.session_state.original_filename = filename
            
            # File info display
            st.markdown("##### 📋 File Information")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("File Name", filename)
                st.metric("File Size", f"{len(file_data):,} bytes")
            with info_col2:
                st.metric("File Extension", file_ext or "None")
                st.metric("Detected Type", file_type.title())
            
            st.markdown("---")
            
            # Compression options
            st.markdown("##### ⚙️ Encoding Parameters")
            
            compression_option = st.radio(
                "Compression",
                options=["None", "Data-Type-Specific"],
                horizontal=True,
                help="None: Raw encoding | Data-Type-Specific: Brotli for text, WebP for images, Object optimization for PDF"
            )
            
            # Compression quality controls
            brotli_quality = 11  # default
            webp_quality = 85    # default
            pdf_image_dpi = 150  # default
            pdf_image_quality = 85  # default
            pdf_method_code = 'pdf_opt'  # default
            audio_method_code = 'flac'  # default
            audio_bitrate = 192  # default
            video_method_code = 'h264'  # default
            video_crf = 23  # default
            video_preset = 'medium'  # default
            
            if compression_option == "Data-Type-Specific":
                if file_type == 'text' or file_type == 'binary':
                    st.info("📝 **Text/Binary detected** - Will use Brotli compression")
                    brotli_quality = st.slider(
                        "Brotli Quality Level",
                        min_value=0,
                        max_value=11,
                        value=11,
                        help="0 = fastest (less compression) | 11 = best compression (slower). Higher values remove more redundancy."
                    )
                    st.caption(f"Quality {brotli_quality}: {'Maximum compression' if brotli_quality == 11 else 'Faster encoding' if brotli_quality < 5 else 'Balanced'}")
                    
                elif file_type == 'image':
                    st.info("🖼️ **Image detected** - Will use WebP compression")
                    webp_quality = st.slider(
                        "WebP Quality Level",
                        min_value=1,
                        max_value=100,
                        value=85,
                        help="1 = maximum compression (more artifacts) | 100 = lossless quality. Lower values remove more visual information."
                    )
                    st.caption(f"Quality {webp_quality}: {'High compression (lossy)' if webp_quality < 50 else 'Balanced quality' if webp_quality < 90 else 'Near-lossless'}")
                
                elif file_type == 'pdf':
                    st.info("📄 **PDF detected** - Choose compression method")
                    
                    # Show PDF info
                    pdf_info = get_pdf_info(file_data)
                    with st.expander("📋 PDF Details", expanded=False):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Pages", pdf_info['page_count'])
                            st.metric("Has Images", "Yes" if pdf_info['has_images'] else "No")
                        with col_b:
                            st.metric("Has Text", "Yes" if pdf_info['has_text'] else "No")
                            if pdf_info['metadata'].get('title'):
                                st.metric("Title", pdf_info['metadata']['title'][:20])
                    
                    # Method selection
                    st.markdown("**🔧 Compression Method**")
                    pdf_method = st.radio(
                        "Select Method",
                        options=["Object Optimization (pdf_opt)", "Image Downsampling (pdf_gs)"],
                        horizontal=True,
                        index=0,
                        help="pdf_opt: Lossless cleanup | pdf_gs: Lossy image compression"
                    )
                    
                    # Map selection to method code
                    pdf_method_code = 'pdf_opt' if 'pdf_opt' in pdf_method else 'pdf_gs'
                    
                    if pdf_method_code == 'pdf_opt':
                        st.markdown("---")
                        st.markdown("**📝 Object Optimization** (Lossless)")
                        st.caption("✓ Remove unused objects")
                        st.caption("✓ Flatten structure")
                        st.caption("✓ Compress streams")
                        st.success("⚡ **No quality settings** - This method preserves ALL content exactly as-is")
                        st.caption("💡 Best for: Text-heavy PDFs, legal documents, forms")
                        # Set defaults (won't affect output for pdf_opt)
                        pdf_image_dpi = 150
                        pdf_image_quality = 85
                    
                    else:  # pdf_gs
                        st.markdown("---")
                        st.markdown("**🖼️ Image Downsampling** (Lossy)")
                        st.caption("✓ Reduces image resolution")
                        st.caption("✓ Recompresses images")
                        st.caption("✓ Maximum file size reduction")
                        
                        # Single quality slider
                        pdf_quality = st.slider(
                            "Compression Quality",
                            min_value=1,
                            max_value=100,
                            value=50,
                            help="1 = Maximum compression (lowest quality) | 100 = Minimum compression (best quality)"
                        )
                        
                        # Map single slider to DPI and JPEG quality
                        # Quality 1-33: Aggressive (72 DPI, 40-70% JPEG)
                        # Quality 34-66: Balanced (150 DPI, 70-85% JPEG)
                        # Quality 67-100: Quality (300 DPI, 85-95% JPEG)
                        if pdf_quality <= 33:
                            pdf_image_dpi = 72
                            pdf_image_quality = 40 + int((pdf_quality / 33) * 30)  # 40-70
                            quality_label = "⚡ Aggressive"
                            quality_desc = "Maximum compression, noticeable quality loss"
                        elif pdf_quality <= 66:
                            pdf_image_dpi = 150
                            pdf_image_quality = 70 + int(((pdf_quality - 33) / 33) * 15)  # 70-85
                            quality_label = "⚖️ Balanced"
                            quality_desc = "Good compression, suitable for screens"
                        else:
                            pdf_image_dpi = 300
                            pdf_image_quality = 85 + int(((pdf_quality - 66) / 34) * 10)  # 85-95
                            quality_label = "🎨 Quality"
                            quality_desc = "Minimal compression, suitable for printing"
                        
                        st.caption(f"{quality_label}: {quality_desc}")
                        
                        with st.expander("🔍 Technical Details"):
                            st.caption(f"Image Resolution: {pdf_image_dpi} DPI")
                            st.caption(f"JPEG Quality: {pdf_image_quality}%")
                        
                        if not pdf_info['has_images']:
                            st.warning("ℹ️ No images detected - compression benefit may be limited")
                
                elif file_type == 'audio':
                    st.info("🎵 **Audio detected** - Choose compression method")
                    
                    # Show audio info
                    audio_info = get_audio_info(file_data)
                    
                    with st.expander("📋 Audio Details", expanded=False):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if audio_info['duration']:
                                mins = int(audio_info['duration'] // 60)
                                secs = int(audio_info['duration'] % 60)
                                st.metric("Duration", f"{mins}:{secs:02d}")
                            if audio_info['sample_rate']:
                                st.metric("Sample Rate", f"{audio_info['sample_rate']:,} Hz")
                            if audio_info['bitrate']:
                                st.metric("Bitrate", f"{audio_info['bitrate']} kbps")
                        with col_b:
                            if audio_info['channels']:
                                ch_name = "Stereo" if audio_info['channels'] == 2 else "Mono" if audio_info['channels'] == 1 else f"{audio_info['channels']}ch"
                                st.metric("Channels", ch_name)
                            if audio_info['codec']:
                                st.metric("Codec", audio_info['codec'].upper())
                            quality_tier = audio_info.get('quality_tier', 'unknown')
                            tier_emoji = {'lossless': '🎼', 'high': '🎧', 'medium': '🎵', 'low': '📻', 'very_low': '📢'}.get(quality_tier, '❓')
                            st.metric("Quality", f"{tier_emoji} {quality_tier.replace('_', ' ').title()}")
                    
                    # Get compression suggestion
                    suggestion = get_audio_compression_suggestion(audio_info)
                    
                    # Show suggestion
                    st.markdown("---")
                    st.markdown("**💡 Recommended Compression**")
                    
                    method_names = {
                        'flac': 'FLAC (Lossless)',
                        'aac': 'AAC (Lossy - Efficient)',
                        'mp3': 'MP3 (Lossy - Universal)'
                    }
                    
                    rec_method = suggestion['method']
                    rec_bitrate = suggestion.get('bitrate', 192)
                    
                    if suggestion.get('warning'):
                        st.warning(f"⚠️ {suggestion['warning']}")
                    
                    st.success(f"✨ **{method_names.get(rec_method, rec_method)}** - {suggestion['reason']}")
                    
                    # Method selection
                    st.markdown("**🔧 Compression Method**")
                    
                    # Build options list
                    method_options = ["FLAC (Lossless)", "AAC (Lossy)", "MP3 (Lossy)"]
                    default_idx = {'flac': 0, 'aac': 1, 'mp3': 2}.get(rec_method, 1)
                    
                    audio_method = st.radio(
                        "Select Method",
                        options=method_options,
                        horizontal=True,
                        index=default_idx,
                        help="FLAC: Perfect quality | AAC: Efficient at low bitrates | MP3: Universal compatibility"
                    )
                    
                    # Map selection to method code
                    audio_method_code = 'flac' if 'FLAC' in audio_method else ('aac' if 'AAC' in audio_method else 'mp3')
                    
                    if audio_method_code == 'flac':
                        st.markdown("---")
                        st.markdown("**🎼 FLAC** (Free Lossless Audio Codec)")
                        st.caption("✓ Lossless - Perfect reconstruction")
                        st.caption("✓ Typically 50-70% of original WAV size")
                        st.caption("✓ Preserves studio quality for archiving")
                        
                        if audio_info.get('is_lossless') and not audio_info.get('is_compressed'):
                            st.success("⚡ **Ideal choice** - FLAC is perfect for uncompressed audio like WAV/AIFF")
                        elif audio_info.get('is_lossless') and audio_info.get('is_compressed'):
                            st.warning("⚠️ File is already lossless compressed - FLAC may not reduce size further")
                        else:
                            st.warning("⚠️ File is already lossy compressed - FLAC will likely INCREASE file size")
                        
                        audio_bitrate = 0  # Not used for FLAC
                    
                    elif audio_method_code == 'aac':
                        st.markdown("---")
                        st.markdown("**🎧 AAC** (Advanced Audio Coding)")
                        st.caption("✓ More efficient than MP3 at same bitrate")
                        st.caption("✓ Better quality at low bitrates (64-128 kbps)")
                        st.caption("✓ Native support on Apple devices and modern browsers")
                        
                        # Determine default bitrate based on suggestion
                        default_bitrate = rec_bitrate if rec_method == 'aac' else 128
                        
                        # Bitrate selection
                        bitrate_preset = st.select_slider(
                            "Audio Quality",
                            options=["Low (64 kbps)", "Medium (96 kbps)", "Good (128 kbps)", "High (192 kbps)", "Best (256 kbps)"],
                            value=f"Good (128 kbps)" if default_bitrate <= 128 else f"High (192 kbps)",
                            help="Higher bitrate = better quality, larger file"
                        )
                        
                        bitrate_map = {
                            "Low (64 kbps)": 64,
                            "Medium (96 kbps)": 96,
                            "Good (128 kbps)": 128,
                            "High (192 kbps)": 192,
                            "Best (256 kbps)": 256
                        }
                        audio_bitrate = bitrate_map[bitrate_preset]
                        
                        # Quality description
                        quality_tier = audio_info.get('quality_tier', 'unknown')
                        if quality_tier in ('low', 'very_low'):
                            st.info(f"💡 Original is {quality_tier.replace('_', ' ')} quality - AAC is optimal for further compression")
                        elif quality_tier == 'medium':
                            st.info("💡 AAC 128kbps ≈ MP3 160kbps in quality")
                    
                    else:  # mp3
                        st.markdown("---")
                        st.markdown("**🎵 MP3** (MPEG Audio Layer III)")
                        st.caption("✓ Universal compatibility")
                        st.caption("✓ Works on all devices and players")
                        st.caption("✓ Best for maximum compatibility")
                        
                        # Bitrate selection
                        bitrate_preset = st.select_slider(
                            "Audio Quality",
                            options=["Low (96 kbps)", "Medium (128 kbps)", "Good (192 kbps)", "High (256 kbps)", "Best (320 kbps)"],
                            value="Good (192 kbps)",
                            help="Higher bitrate = better quality, larger file"
                        )
                        
                        bitrate_map = {
                            "Low (96 kbps)": 96,
                            "Medium (128 kbps)": 128,
                            "Good (192 kbps)": 192,
                            "High (256 kbps)": 256,
                            "Best (320 kbps)": 320
                        }
                        audio_bitrate = bitrate_map[bitrate_preset]
                        
                        # Quality description
                        quality_tier = audio_info.get('quality_tier', 'unknown')
                        if quality_tier in ('low', 'very_low'):
                            st.warning(f"⚠️ Original is {quality_tier.replace('_', ' ')} quality ({audio_info.get('bitrate', '?')} kbps) - Consider AAC for better efficiency at low bitrates")
                
                elif file_type == 'video':
                    st.info("🎬 **Video detected** - Choose compression method")
                    
                    # Show video info
                    video_info = get_video_info(file_data)
                    
                    with st.expander("📋 Video Details", expanded=False):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if video_info['duration']:
                                mins = int(video_info['duration'] // 60)
                                secs = int(video_info['duration'] % 60)
                                st.metric("Duration", f"{mins}:{secs:02d}")
                            if video_info['width'] and video_info['height']:
                                st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
                            if video_info['fps']:
                                st.metric("Frame Rate", f"{video_info['fps']} fps")
                        with col_b:
                            if video_info['video_codec']:
                                st.metric("Video Codec", video_info['video_codec'].upper())
                            if video_info['bitrate']:
                                st.metric("Bitrate", f"{video_info['bitrate']} kbps")
                            st.metric("Has Audio", "Yes" if video_info['has_audio'] else "No")
                    
                    # Get compression suggestion
                    suggestion = get_video_compression_suggestion(video_info)
                    
                    # Show suggestion
                    st.markdown("---")
                    st.markdown("**💡 Recommended Compression**")
                    
                    method_names = {
                        'h264': 'H.264/AVC (Universal)',
                        'av1': 'AV1 (Best Compression)'
                    }
                    
                    rec_method = suggestion['method']
                    rec_crf = suggestion.get('crf', 23)
                    
                    if suggestion.get('warning'):
                        st.warning(f"⚠️ {suggestion['warning']}")
                    
                    st.success(f"✨ **{method_names.get(rec_method, rec_method)}** - {suggestion['reason']}")
                    
                    # Method selection
                    st.markdown("**🔧 Compression Method**")
                    
                    video_method = st.radio(
                        "Select Method",
                        options=["H.264/AVC (Fast & Compatible)", "AV1 (Better Compression)"],
                        horizontal=True,
                        index=0 if rec_method == 'h264' else 1,
                        help="H.264: Fast encoding, universal playback | AV1: Better compression, slower encoding"
                    )
                    
                    # Map selection to method code
                    video_method_code = 'h264' if 'H.264' in video_method else 'av1'
                    
                    if video_method_code == 'h264':
                        st.markdown("---")
                        st.markdown("**🎥 H.264/AVC** (Advanced Video Coding)")
                        st.caption("✓ Universal compatibility")
                        st.caption("✓ Fast encoding")
                        st.caption("✓ Hardware acceleration support")
                        
                        # Quality selection
                        quality_preset = st.select_slider(
                            "Video Quality",
                            options=["Low (CRF 28)", "Medium (CRF 23)", "High (CRF 18)", "Very High (CRF 14)", "Near Lossless (CRF 10)"],
                            value="Medium (CRF 23)",
                            help="Lower CRF = better quality, larger file"
                        )
                        
                        crf_map = {
                            "Low (CRF 28)": 28,
                            "Medium (CRF 23)": 23,
                            "High (CRF 18)": 18,
                            "Very High (CRF 14)": 14,
                            "Near Lossless (CRF 10)": 10
                        }
                        video_crf = crf_map[quality_preset]
                        
                        # Preset selection
                        speed_preset = st.select_slider(
                            "Encoding Speed",
                            options=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                            value="medium",
                            help="Slower = better compression at same quality"
                        )
                        video_preset = speed_preset
                        
                    else:  # av1
                        st.markdown("---")
                        st.markdown("**🎯 AV1** (AOMedia Video 1)")
                        st.caption("✓ 30-50% better compression than H.264")
                        st.caption("✓ Royalty-free")
                        st.caption("⚠️ Slower encoding")
                        
                        # Quality selection for AV1
                        quality_preset = st.select_slider(
                            "Video Quality",
                            options=["Low (CRF 40)", "Medium (CRF 30)", "High (CRF 22)", "Very High (CRF 15)"],
                            value="Medium (CRF 30)",
                            help="Lower CRF = better quality, larger file"
                        )
                        
                        crf_map = {
                            "Low (CRF 40)": 40,
                            "Medium (CRF 30)": 30,
                            "High (CRF 22)": 22,
                            "Very High (CRF 15)": 15
                        }
                        video_crf = crf_map[quality_preset]
                        
                        # Preset selection for AV1
                        speed_preset = st.select_slider(
                            "Encoding Speed",
                            options=["fast", "medium", "slow"],
                            value="medium",
                            help="Slower = better compression at same quality"
                        )
                        video_preset = speed_preset
                        
                        st.warning("⚠️ AV1 encoding is significantly slower than H.264. For large videos, consider using H.264.")
            
            st.markdown("##### 🧬 DNA Mapping")
            st.code("A=00, T=01, G=10, C=11 (2 bits per nucleotide)", language=None)
            
            # Confirmation and encode button
            st.markdown("---")
            st.markdown("##### ✅ Confirm & Encode")
            
            with st.expander("📋 Parameter Summary", expanded=True):
                st.write(f"**File:** {filename}")
                st.write(f"**Size:** {len(file_data):,} bytes")
                st.write(f"**Type:** {file_type.title()}")
                st.write(f"**Compression:** {compression_option}")
                st.write("**Metadata:** Embedded in DNA sequence")
            
            if st.button("🚀 Start Encoding", type="primary", use_container_width=True):
                with st.spinner("Encoding in progress..."):
                    start_time = time.time()
                    
                    # Apply compression if selected
                    if compression_option == "Data-Type-Specific":
                        compressed_data, comp_type = compress_data(
                            file_data, file_type, 'auto',
                            brotli_quality=brotli_quality,
                            webp_quality=webp_quality,
                            pdf_image_dpi=pdf_image_dpi,
                            pdf_image_quality=pdf_image_quality,
                            pdf_method=pdf_method_code,
                            audio_method=audio_method_code,
                            audio_bitrate=audio_bitrate,
                            video_method=video_method_code,
                            video_crf=video_crf,
                            video_preset=video_preset
                        )
                        compressed = comp_type != 'none'
                    else:
                        compressed_data = file_data
                        comp_type = 'none'
                        compressed = False
                    
                    # Encode to DNA
                    dna_sequence = encode_to_dna(
                        compressed_data, 
                        compressed=compressed,
                        compression_type=comp_type,
                        original_extension=file_ext
                    )
                    
                    encoding_time = time.time() - start_time
                    
                    # Store in session state
                    st.session_state.encoded_dna = dna_sequence
                    st.session_state.encoded_metadata = {
                        'filename': filename,
                        'original_size': len(file_data),
                        'compressed_size': len(compressed_data),
                        'compression_type': comp_type,
                        'dna_length': len(dna_sequence),
                        'encoding_time': encoding_time,
                        'file_type': file_type,
                        'extension': file_ext
                    }
                    
                    st.success("✅ Encoding completed successfully!")
    
    with col2:
        st.subheader("📤 Output")
        
        if st.session_state.encoded_dna:
            dna = st.session_state.encoded_dna
            meta = st.session_state.encoded_metadata
            
            # Result preview
            st.markdown("##### 🔬 Result Preview")
            preview_length = min(500, len(dna))
            preview_text = dna[:preview_length]
            if len(dna) > preview_length:
                preview_text += f"\n... ({len(dna) - preview_length:,} more nucleotides)"
            
            st.text_area(
                "DNA Sequence Preview",
                value=preview_text,
                height=200,
                disabled=True
            )
            
            # Encoding details
            st.markdown("##### 📊 Encoding Details")
            
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.metric("DNA Length", f"{meta['dna_length']:,} nt")
                st.metric("Original Size", f"{meta['original_size']:,} bytes")
                if meta['compression_type'] != 'none':
                    st.metric("Compressed Size", f"{meta['compressed_size']:,} bytes")
            
            with detail_col2:
                density = calculate_encoding_density(meta['original_size'], meta['dna_length'])
                st.metric("Encoding Density", f"{density:.4f} bits/nt")
                st.metric("Encoding Time", f"{meta['encoding_time']:.3f} s")
                st.metric("Compression", meta['compression_type'].upper())
            
            if meta['compression_type'] != 'none':
                ratio = meta['original_size'] / meta['compressed_size']
                st.metric("Compression Ratio", f"{ratio:.2f}x")
            else:
                # Explain why compression is "none"
                st.markdown("---")
                st.markdown("##### ℹ️ Compression Status")
                file_type = meta.get('file_type', 'binary')
                
                # Check if compression was disabled by user choice
                if meta['original_size'] == meta['compressed_size']:
                    explanations = []
                    
                    if file_type in ('text', 'binary'):
                        explanations.append("**Brotli compression** was applied but the compressed data was larger than or equal to the original.")
                        explanations.append("This typically happens with:")
                        explanations.append("- Already compressed files (ZIP, RAR, etc.)")
                        explanations.append("- Small files with little redundancy")
                        explanations.append("- Random or encrypted data")
                    elif file_type == 'image':
                        explanations.append("**WebP compression** was applied but the compressed image was larger than the original.")
                        explanations.append("This can happen with:")
                        explanations.append("- Already highly compressed images (JPEG at low quality)")
                        explanations.append("- Small images with few pixels")
                        explanations.append("- Images with high entropy/noise")
                    elif file_type == 'audio':
                        explanations.append("**Audio compression** was applied but did not reduce file size.")
                        explanations.append("This typically happens with:")
                        explanations.append("- Already compressed audio (MP3, AAC)")
                        explanations.append("- Very short audio clips")
                    elif file_type == 'video':
                        explanations.append("**Video compression** was applied but did not reduce file size.")
                        explanations.append("This typically happens with:")
                        explanations.append("- Already highly compressed video")
                        explanations.append("- Very short video clips")
                    elif file_type == 'pdf':
                        explanations.append("**PDF optimization** was applied but did not reduce file size.")
                        explanations.append("This typically happens with:")
                        explanations.append("- Already optimized PDFs")
                        explanations.append("- PDFs without embedded images")
                    
                    for exp in explanations:
                        st.caption(exp)
                    
                    st.info("💡 **Result:** Data stored uncompressed at the baseline 2.0 bits/nucleotide density.")
            
            # Download button
            st.markdown("##### 💾 Download")
            st.download_button(
                label="📥 Download DNA Sequence (.txt)",
                data=dna,
                file_name=f"{os.path.splitext(meta['filename'])[0]}_dna.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("👈 Upload a file and click 'Start Encoding' to see results here.")


def randomization_mode():
    """Randomization mode: Apply chaos map to DNA sequence."""
    st.header("🔀 Randomization Mode")
    st.markdown("Apply chaos map to randomize and secure DNA sequences.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input")
        
        # Data source selection
        input_source = st.radio(
            "DNA Sequence Source",
            options=["From Encode Mode", "Upload DNA File"],
            horizontal=True
        )
        
        dna_sequence = None
        
        if input_source == "From Encode Mode":
            if st.session_state.encoded_dna:
                dna_sequence = st.session_state.encoded_dna
                st.success(f"✅ Loaded DNA sequence ({len(dna_sequence):,} nt) from Encode mode")
            else:
                st.warning("⚠️ No DNA sequence available from Encode mode. Please encode a file first.")
        else:
            uploaded_dna = st.file_uploader(
                "Upload DNA sequence file (.txt)",
                type=['txt'],
                key="random_upload"
            )
            if uploaded_dna:
                dna_sequence = uploaded_dna.read().decode('utf-8').strip()
                dna_sequence = ''.join(c for c in dna_sequence.upper() if c in 'ATGC')
                if validate_dna_sequence(dna_sequence):
                    st.success(f"✅ Loaded valid DNA sequence ({len(dna_sequence):,} nt)")
                else:
                    st.error("❌ Invalid DNA sequence. Only A, T, G, C characters allowed.")
                    dna_sequence = None
        
        if dna_sequence:
            # Show original characteristics
            st.markdown("##### 📊 Original DNA Characteristics")
            orig_chars = calculate_dna_characteristics(dna_sequence)
            
            char_col1, char_col2 = st.columns(2)
            with char_col1:
                st.metric("Length", f"{orig_chars['length']:,} nt")
                st.metric("Max Homopolymer", f"{orig_chars['max_homopolymer']} nt")
            with char_col2:
                st.metric("GC Ratio", f"{orig_chars['gc_ratio']:.4f}")
                st.metric("Shannon Entropy", f"{orig_chars['shannon_entropy']:.4f}")
            
            st.markdown("---")
            
            # Randomization options
            st.markdown("##### ⚙️ Randomization Options")
            
            randomization_option = st.radio(
                "Randomization Method",
                options=["None", "Chaos Map"],
                horizontal=True
            )
            
            forward_primer = ""
            reverse_primer = ""
            selected_systems = []
            
            if randomization_option == "Chaos Map":
                st.info("🔐 **Chaos Map Encryption** uses primer sequences as keys. Chain multiple systems for enhanced security!")
                
                # Mode selection: Single or Multi
                chain_mode = st.radio(
                    "Mode",
                    options=["Single System", "Multi-System Chain"],
                    horizontal=True,
                    help="Chain multiple chaos systems for stronger randomization"
                )
                
                if chain_mode == "Single System":
                    # Single system selection
                    chaos_options = {
                        "Logistic Map (1D - Simple)": ChaosSystem.LOGISTIC,
                        "Hénon Map (2D - Medium)": ChaosSystem.HENON,
                        "Lorenz System (3D - Complex)": ChaosSystem.LORENZ
                    }
                    
                    selected_system_name = st.selectbox(
                        "Chaos System",
                        options=list(chaos_options.keys()),
                        index=1,  # Default to Hénon
                        help="Choose chaotic system complexity: Simple → Medium → Complex"
                    )
                    selected_systems = [chaos_options[selected_system_name]]
                    
                    # Show system info
                    system_info = get_chaos_system_info(selected_systems[0])
                    with st.expander("ℹ️ System Details", expanded=False):
                        st.markdown(f"**{system_info['name']}** ({system_info['dimensions']}D)")
                        st.markdown(f"*Complexity:* {system_info['complexity']}")
                        st.markdown(f"*Equation:* `{system_info['equation']}`")
                        st.markdown(f"*Description:* {system_info['description']}")
                
                else:  # Multi-System Chain
                    st.markdown("**Select systems to chain** (order: top to bottom)")
                    
                    # Checkboxes for each system
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        use_logistic = st.checkbox(
                            "1️⃣ Logistic Map",
                            value=True,
                            help="1D - Simplest system"
                        )
                    
                    with col_b:
                        use_henon = st.checkbox(
                            "2️⃣ Hénon Map",
                            value=True,
                            help="2D - Medium complexity"
                        )
                    
                    with col_c:
                        use_lorenz = st.checkbox(
                            "3️⃣ Lorenz System",
                            value=True,
                            help="3D - Most complex"
                        )
                    
                    # Build system chain in order
                    if use_logistic:
                        selected_systems.append(ChaosSystem.LOGISTIC)
                    if use_henon:
                        selected_systems.append(ChaosSystem.HENON)
                    if use_lorenz:
                        selected_systems.append(ChaosSystem.LORENZ)
                    
                    if selected_systems:
                        # Show chain info
                        chain_info = get_chain_complexity(selected_systems)
                        
                        st.markdown("---")
                        st.markdown(f"**Chain:** {chain_info['description']}")
                        
                        col_x, col_y, col_z = st.columns(3)
                        with col_x:
                            st.metric("Systems", chain_info['num_systems'])
                        with col_y:
                            st.metric("Total Layers", chain_info['total_layers'])
                        with col_z:
                            st.metric("Dimensions", f"{chain_info['total_dimensions']}D")
                        
                        st.caption(f"🔒 Security Level: **{chain_info['complexity']}**")
                    else:
                        st.warning("⚠️ Please select at least one chaos system")
                
                st.markdown("---")
                
                forward_primer = st.text_input(
                    "Forward Primer (Key 1 - 20 nt)",
                    value="ACACGACGCTCTTCCGATCT",
                    help="DNA sequence used as part of the encryption key",
                    max_chars=50
                )
                
                reverse_primer = st.text_input(
                    "Reverse Primer (Key 2 - 21nt)", 
                    value="AGATCGGAAGAGCACACGTCT",
                    help="DNA sequence used as part of the encryption key",
                    max_chars=50
                )
                
                if forward_primer and reverse_primer:
                    if not all(c in 'ATGCatgc' for c in forward_primer):
                        st.error("❌ Forward primer contains invalid characters")
                    elif not all(c in 'ATGCatgc' for c in reverse_primer):
                        st.error("❌ Reverse primer contains invalid characters")
            
            # Apply randomization
            st.markdown("---")
            if st.button("🔀 Apply Randomization", type="primary", use_container_width=True):
                if randomization_option == "None":
                    st.session_state.randomized_dna = dna_sequence
                    st.session_state.chaos_systems = None
                    st.success("✅ No randomization applied. DNA sequence passed through.")
                else:
                    if not forward_primer or not reverse_primer:
                        st.error("❌ Please enter both primer sequences")
                    elif not selected_systems:
                        st.error("❌ Please select at least one chaos system")
                    else:
                        chain_desc = get_chain_description(selected_systems)
                        with st.spinner(f"Applying {chain_desc} scrambling..."):
                            randomized = randomize_dna(
                                dna_sequence, 
                                forward_primer, 
                                reverse_primer, 
                                systems=selected_systems
                            )
                            st.session_state.randomized_dna = randomized
                            
                            # Store primers and systems for decode reference
                            st.session_state.randomization_primers = {
                                'forward': forward_primer.upper(),
                                'reverse': reverse_primer.upper()
                            }
                            st.session_state.chaos_systems = selected_systems
                            
                            st.success("✅ Randomization completed!")
    
    with col2:
        st.subheader("📤 Output")
        
        if st.session_state.randomized_dna:
            randomized_dna = st.session_state.randomized_dna
            
            # Preview
            st.markdown("##### 🔬 Randomized DNA Preview")
            preview_length = min(500, len(randomized_dna))
            preview_text = randomized_dna[:preview_length]
            if len(randomized_dna) > preview_length:
                preview_text += f"\n... ({len(randomized_dna) - preview_length:,} more nucleotides)"
            
            st.text_area(
                "Randomized DNA Sequence",
                value=preview_text,
                height=200,
                disabled=True
            )
            
            # Characteristics after randomization
            st.markdown("##### 📊 Randomized DNA Characteristics")
            rand_chars = calculate_dna_characteristics(randomized_dna)
            
            char_col1, char_col2 = st.columns(2)
            with char_col1:
                st.metric("Length", f"{rand_chars['length']:,} nt")
                st.metric("Max Homopolymer", f"{rand_chars['max_homopolymer']} nt")
            with char_col2:
                st.metric("GC Ratio", f"{rand_chars['gc_ratio']:.4f}")
                st.metric("Shannon Entropy", f"{rand_chars['shannon_entropy']:.4f}")
            
            # Download
            st.markdown("##### 💾 Download")
            st.download_button(
                label="📥 Download Randomized DNA (.txt)",
                data=randomized_dna,
                file_name="randomized_dna.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("👈 Load a DNA sequence and apply randomization to see results here.")


def decode_mode():
    """Decode mode: Convert DNA sequence back to original file."""
    st.header("📥 Decode Mode")
    st.markdown("Decode DNA sequences back to their original file format.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input")
        
        # Data source selection
        input_source = st.radio(
            "DNA Sequence Source",
            options=["From Randomization Mode", "Upload DNA File"],
            horizontal=True,
            key="decode_source"
        )
        
        dna_sequence = None
        
        if input_source == "From Randomization Mode":
            if st.session_state.randomized_dna:
                dna_sequence = st.session_state.randomized_dna
                st.success(f"✅ Loaded DNA sequence ({len(dna_sequence):,} nt) from Randomization mode")
            else:
                st.warning("⚠️ No DNA sequence available. Please complete Encode and Randomization first.")
        else:
            uploaded_dna = st.file_uploader(
                "Upload DNA sequence file (.txt)",
                type=['txt'],
                key="decode_upload"
            )
            if uploaded_dna:
                dna_sequence = uploaded_dna.read().decode('utf-8').strip()
                dna_sequence = ''.join(c for c in dna_sequence.upper() if c in 'ATGC')
                if validate_dna_sequence(dna_sequence):
                    st.success(f"✅ Loaded valid DNA sequence ({len(dna_sequence):,} nt)")
                else:
                    st.error("❌ Invalid DNA sequence")
                    dna_sequence = None
        
        if dna_sequence:
            st.markdown("---")
            
            # Re-randomization option
            st.markdown("##### 🔓 Re-randomization (Decryption)")
            
            was_randomized = st.checkbox(
                "Sequence was randomized with Chaos Map",
                value=hasattr(st.session_state, 'randomization_primers')
            )
            
            forward_primer = ""
            reverse_primer = ""
            decode_systems = []
            
            if was_randomized:
                st.info("🔐 Enter the same primer sequences and chaos system(s) used during randomization")
                
                # Pre-fill if available
                default_forward = ""
                default_reverse = ""
                default_systems = None
                if hasattr(st.session_state, 'randomization_primers'):
                    default_forward = st.session_state.randomization_primers.get('forward', '')
                    default_reverse = st.session_state.randomization_primers.get('reverse', '')
                if hasattr(st.session_state, 'chaos_systems') and st.session_state.chaos_systems:
                    default_systems = st.session_state.chaos_systems
                
                forward_primer = st.text_input(
                    "Forward Primer (Key 1)",
                    value=default_forward,
                    key="decode_forward"
                )
                
                reverse_primer = st.text_input(
                    "Reverse Primer (Key 2)",
                    value=default_reverse,
                    key="decode_reverse"
                )
                
                # Mode selection for decode
                decode_chain_mode = st.radio(
                    "Decryption Mode",
                    options=["Single System", "Multi-System Chain"],
                    horizontal=True,
                    index=1 if (default_systems and len(default_systems) > 1) else 0,
                    key="decode_chain_mode"
                )
                
                if decode_chain_mode == "Single System":
                    # Determine default index
                    default_idx = 1  # Default to Hénon
                    if default_systems and len(default_systems) == 1:
                        system_to_idx = {
                            ChaosSystem.LOGISTIC: 0,
                            ChaosSystem.HENON: 1,
                            ChaosSystem.LORENZ: 2
                        }
                        default_idx = system_to_idx.get(default_systems[0], 1)
                    
                    decode_chaos_options = {
                        "Logistic Map (1D - Simple)": ChaosSystem.LOGISTIC,
                        "Hénon Map (2D - Medium)": ChaosSystem.HENON,
                        "Lorenz System (3D - Complex)": ChaosSystem.LORENZ
                    }
                    
                    decode_system_name = st.selectbox(
                        "Chaos System Used",
                        options=list(decode_chaos_options.keys()),
                        index=default_idx,
                        help="Must match the system used during randomization",
                        key="decode_chaos_system"
                    )
                    decode_systems = [decode_chaos_options[decode_system_name]]
                
                else:  # Multi-System Chain
                    st.markdown("**Select the same systems used during randomization:**")
                    
                    # Determine defaults
                    default_logistic = default_systems and ChaosSystem.LOGISTIC in default_systems
                    default_henon = default_systems and ChaosSystem.HENON in default_systems
                    default_lorenz = default_systems and ChaosSystem.LORENZ in default_systems
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        decode_use_logistic = st.checkbox(
                            "1️⃣ Logistic",
                            value=default_logistic,
                            key="decode_logistic"
                        )
                    
                    with col_b:
                        decode_use_henon = st.checkbox(
                            "2️⃣ Hénon",
                            value=default_henon,
                            key="decode_henon"
                        )
                    
                    with col_c:
                        decode_use_lorenz = st.checkbox(
                            "3️⃣ Lorenz",
                            value=default_lorenz,
                            key="decode_lorenz"
                        )
                    
                    # Build system chain in order
                    if decode_use_logistic:
                        decode_systems.append(ChaosSystem.LOGISTIC)
                    if decode_use_henon:
                        decode_systems.append(ChaosSystem.HENON)
                    if decode_use_lorenz:
                        decode_systems.append(ChaosSystem.LORENZ)
                    
                    if decode_systems:
                        chain_desc = get_chain_description(decode_systems)
                        st.caption(f"🔓 Chain: **{chain_desc}**")
            
            st.markdown("---")
            
            # Decode button
            if st.button("🔓 Decode DNA Sequence", type="primary", use_container_width=True):
                with st.spinner("Decoding in progress..."):
                    try:
                        # Apply de-randomization if needed
                        if was_randomized and forward_primer and reverse_primer and decode_systems:
                            dna_to_decode = derandomize_dna(
                                dna_sequence, 
                                forward_primer, 
                                reverse_primer, 
                                systems=decode_systems
                            )
                        else:
                            dna_to_decode = dna_sequence
                        
                        # Decode DNA to data
                        decoded_data, metadata = decode_from_dna(dna_to_decode)
                        
                        # Apply decompression
                        final_data = decompress_data(
                            decoded_data,
                            metadata['compression_type'],
                            metadata['extension']
                        )
                        
                        # Store results
                        st.session_state.decoded_data = final_data
                        st.session_state.decoded_metadata = metadata
                        
                        st.success("✅ Decoding completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Decoding failed: {str(e)}")
    
    with col2:
        st.subheader("📤 Output")
        
        if st.session_state.decoded_data is not None:
            data = st.session_state.decoded_data
            meta = st.session_state.decoded_metadata
            
            # Metadata display
            st.markdown("##### 📋 Decoded File Information")
            
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("File Size", f"{len(data):,} bytes")
                st.metric("Original Extension", meta.get('extension', 'Unknown'))
            with info_col2:
                st.metric("Compression Used", meta.get('compression_type', 'none').upper())
                st.metric("Data Length (from header)", f"{meta.get('data_length', 0):,} bytes")
            
            # Preview based on type
            st.markdown("##### 👁️ Preview")
            
            extension = meta.get('extension', '').lower()
            
            if extension in TEXT_EXTENSIONS or extension == '':
                try:
                    text_preview = data[:2000].decode('utf-8', errors='replace')
                    if len(data) > 2000:
                        text_preview += "\n... (truncated)"
                    st.text_area("Text Preview", value=text_preview, height=200, disabled=True)
                except:
                    st.info("Binary data - cannot preview as text")
            
            elif extension in IMAGE_EXTENSIONS:
                try:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(data))
                    st.image(img, caption="Decoded Image", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not preview image: {e}")
            
            else:
                st.info(f"Binary file with extension: {extension}")
                st.code(f"File size: {len(data):,} bytes")
            
            # Download
            st.markdown("##### 💾 Download")
            
            # Generate filename
            ext = meta.get('extension', '.bin')
            if not ext.startswith('.'):
                ext = '.' + ext
            output_filename = f"decoded_file{ext}"
            
            # Determine MIME type
            mime_types = {
                '.txt': 'text/plain',
                '.json': 'application/json',
                '.csv': 'text/csv',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'application/octet-stream')
            
            st.download_button(
                label=f"📥 Download Decoded File ({ext})",
                data=data,
                file_name=output_filename,
                mime=mime_type,
                use_container_width=True
            )
        else:
            st.info("👈 Load a DNA sequence and click 'Decode' to see results here.")


def comparison_mode():
    """Comparison mode: Compare original and reconstructed data."""
    st.header("📊 Comparison Mode")
    st.markdown("Compare original and reconstructed data to verify encoding/decoding accuracy.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input Files")
        
        # Original file
        st.markdown("##### 📄 Original File")
        original_source = st.radio(
            "Original data source",
            options=["From Encode Mode", "Upload File"],
            horizontal=True,
            key="compare_orig_source"
        )
        
        original_data = None
        original_filename = ""
        
        if original_source == "From Encode Mode":
            if st.session_state.original_data:
                original_data = st.session_state.original_data
                original_filename = st.session_state.original_filename or "original_file"
                st.success(f"✅ Loaded: {original_filename} ({len(original_data):,} bytes)")
            else:
                st.warning("⚠️ No original file available from Encode mode")
        else:
            uploaded_orig = st.file_uploader(
                "Upload original file",
                type=None,
                key="compare_orig_upload"
            )
            if uploaded_orig:
                original_data = uploaded_orig.read()
                original_filename = uploaded_orig.name
                st.success(f"✅ Loaded: {original_filename} ({len(original_data):,} bytes)")
        
        st.markdown("---")
        
        # Reconstructed file
        st.markdown("##### 🔄 Reconstructed File")
        recon_source = st.radio(
            "Reconstructed data source",
            options=["From Decode Mode", "Upload File"],
            horizontal=True,
            key="compare_recon_source"
        )
        
        reconstructed_data = None
        
        if recon_source == "From Decode Mode":
            if st.session_state.decoded_data:
                reconstructed_data = st.session_state.decoded_data
                st.success(f"✅ Loaded decoded data ({len(reconstructed_data):,} bytes)")
            else:
                st.warning("⚠️ No decoded data available from Decode mode")
        else:
            uploaded_recon = st.file_uploader(
                "Upload reconstructed file",
                type=None,
                key="compare_recon_upload"
            )
            if uploaded_recon:
                reconstructed_data = uploaded_recon.read()
                st.success(f"✅ Loaded: {uploaded_recon.name} ({len(reconstructed_data):,} bytes)")
        
        st.markdown("---")
        
        # Compare button
        if original_data and reconstructed_data:
            if st.button("📊 Compare Files", type="primary", use_container_width=True):
                with st.spinner("Comparing data..."):
                    results = compare_data(original_data, reconstructed_data, original_filename)
                    st.session_state.comparison_results = results
                    st.success("✅ Comparison completed!")
        else:
            st.info("⬆️ Load both original and reconstructed files to enable comparison")
    
    with col2:
        st.subheader("📊 Comparison Results")
        
        if hasattr(st.session_state, 'comparison_results') and st.session_state.comparison_results:
            results = st.session_state.comparison_results
            data_type = results.get('data_type', 'unknown')
            
            # Data type badge
            type_colors = {'text': '🔵', 'image': '🟢', 'pdf': '📄', 'binary': '🟠'}
            st.markdown(f"### {type_colors.get(data_type, '⚪')} {data_type.upper()} Comparison")
            
            if data_type == 'text':
                # Text metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    accuracy = results.get('accuracy', 0)
                    st.metric(
                        "Accuracy",
                        f"{accuracy:.4f}%",
                        delta="Perfect" if accuracy == 100 else None
                    )
                with metric_col2:
                    ber = results.get('bit_error_rate', 0)
                    st.metric(
                        "Bit Error Rate",
                        f"{ber:.2e}",
                        delta="Zero errors" if ber == 0 else None
                    )
                
                st.markdown("##### 📝 Details")
                st.write(f"**Character Matches:** {results.get('character_matches', 0):,} / {results.get('total_characters', 0):,}")
                st.write(f"**Original Length:** {results.get('original_length', 0):,} characters")
                st.write(f"**Reconstructed Length:** {results.get('reconstructed_length', 0):,} characters")
            
            elif data_type == 'image':
                if results.get('error'):
                    st.error(f"Error: {results['error']}")
                else:
                    # Image metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        ssim = results.get('ssim')
                        if ssim is not None:
                            st.metric(
                                "SSIM",
                                f"{ssim:.6f}",
                                delta="Perfect" if ssim == 1.0 else None
                            )
                    with metric_col2:
                        psnr = results.get('psnr')
                        if psnr is not None:
                            if psnr == float('inf'):
                                st.metric("PSNR", "∞ dB", delta="Identical")
                            else:
                                st.metric("PSNR", f"{psnr:.2f} dB")
                    with metric_col3:
                        mse = results.get('mse')
                        if mse is not None:
                            st.metric(
                                "MSE",
                                f"{mse:.6f}",
                                delta="Zero" if mse == 0 else None
                            )
                    
                    st.markdown("##### 🖼️ Image Details")
                    st.write(f"**Original Size:** {results.get('original_size', 'N/A')}")
                    st.write(f"**Reconstructed Size:** {results.get('reconstructed_size', 'N/A')}")
            
            elif data_type == 'pdf':
                # PDF-specific comparison metrics
                if results.get('error'):
                    st.error(f"Error: {results['error']}")
                else:
                    # Page count
                    orig_pages = results.get('page_count_original', 0)
                    recon_pages = results.get('page_count_reconstructed', 0)
                    st.write(f"**📄 Pages:** {orig_pages} original → {recon_pages} reconstructed")
                    
                    st.markdown("---")
                    
                    # Text Preservation metrics
                    st.markdown("##### 📝 Text Accuracy")
                    text_pres = results.get('text_preservation')
                    cer = results.get('cer')
                    
                    if text_pres is not None and cer is not None:
                        cer_percent = cer * 100
                        
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric(
                                "Text Preservation",
                                f"{text_pres:.1f}%",
                                delta="Perfect" if text_pres == 100 else None
                            )
                        with metric_col2:
                            st.metric(
                                "Character Error Rate (CER)",
                                f"{cer_percent:.2f}%",
                                delta="Excellent" if cer_percent < 1 else None
                            )
                        
                        # CER interpretation
                        if cer_percent < 1:
                            st.success("✅ **Excellent** - CER < 1% indicates excellent text fidelity")
                        elif cer_percent < 5:
                            st.info("ℹ️ **Good** - Minor text differences detected")
                        else:
                            st.warning("⚠️ **Fair** - Noticeable text differences")
                    else:
                        st.info("📝 Text Preservation: N/A (no text content detected)")
                    
                    st.markdown("---")
                    
                    # Visual Similarity metrics
                    st.markdown("##### 🖼️ Visual Similarity")
                    visual_ssim = results.get('visual_ssim')
                    has_images = results.get('has_images', False)
                    
                    if has_images:
                        if visual_ssim is not None:
                            st.metric(
                                "Visual Similarity (SSIM)",
                                f"{visual_ssim:.2f}",
                                delta="Excellent" if visual_ssim > 0.98 else ("Very Good" if visual_ssim > 0.95 else None)
                            )
                            
                            # SSIM interpretation
                            if visual_ssim > 0.98:
                                st.success("✅ **Excellent** (>0.98) - Visually identical")
                            elif visual_ssim > 0.95:
                                st.success("✅ **Very Good** (>0.95) - Minor visual differences")
                            elif visual_ssim > 0.90:
                                st.info("ℹ️ **Good** (>0.90) - Some visual differences")
                            else:
                                st.warning("⚠️ **Fair** - Noticeable visual differences")
                        else:
                            st.info("🖼️ Visual Similarity: Could not compute (install pdf2image & poppler)")
                    else:
                        st.info("🖼️ Visual Similarity: N/A (page contains no images/charts)")
            
            elif data_type == 'audio':
                # Audio-specific comparison metrics
                if results.get('error'):
                    st.error(f"Error: {results['error']}")
                else:
                    # Duration info
                    dur_orig = results.get('duration_original')
                    dur_recon = results.get('duration_reconstructed')
                    if dur_orig and dur_recon:
                        orig_str = f"{int(dur_orig // 60)}:{int(dur_orig % 60):02d}"
                        recon_str = f"{int(dur_recon // 60)}:{int(dur_recon % 60):02d}"
                        st.write(f"**⏱️ Duration:** {orig_str} original → {recon_str} reconstructed")
                    
                    st.markdown("---")
                    
                    # Compression type
                    is_lossless = results.get('is_lossless', False)
                    if is_lossless:
                        st.success("🎼 **LOSSLESS** - Perfect reconstruction, all audio data preserved")
                    else:
                        st.info("🎧 **LOSSY** - Some audio information removed for compression")
                    
                    st.markdown("---")
                    
                    # Audio quality metrics
                    st.markdown("##### 📊 Audio Quality")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        snr = results.get('snr')
                        if snr is not None:
                            if snr == float('inf'):
                                st.metric("Signal-to-Noise Ratio", "∞ dB", delta="Perfect")
                            else:
                                delta = "Excellent" if snr > 60 else ("Very Good" if snr > 40 else None)
                                st.metric("Signal-to-Noise Ratio", f"{snr:.1f} dB", delta=delta)
                    
                    with metric_col2:
                        corr = results.get('correlation')
                        if corr is not None:
                            delta = "Excellent" if corr > 0.99 else ("Very Good" if corr > 0.95 else None)
                            st.metric("Waveform Correlation", f"{corr:.4f}", delta=delta)
                    
                    # Interpretation
                    snr_val = results.get('snr', 0) or 0
                    corr_val = results.get('correlation', 0) or 0
                    
                    if is_lossless or snr_val == float('inf'):
                        st.success("✅ **Perfect** - Audio is bit-perfect identical")
                    elif snr_val > 40 and corr_val > 0.99:
                        st.success("✅ **Excellent** - Imperceptible quality difference")
                    elif snr_val > 20 and corr_val > 0.95:
                        st.info("ℹ️ **Good** - Minor differences, acceptable for most uses")
                    else:
                        st.warning("⚠️ **Fair** - Noticeable audio quality differences")
            
            elif data_type == 'video':
                # Video-specific comparison metrics
                if results.get('error'):
                    st.error(f"Error: {results['error']}")
                else:
                    # Duration and resolution info
                    dur_orig = results.get('duration_original')
                    dur_recon = results.get('duration_reconstructed')
                    if dur_orig and dur_recon:
                        orig_str = f"{int(dur_orig // 60)}:{int(dur_orig % 60):02d}"
                        recon_str = f"{int(dur_recon // 60)}:{int(dur_recon % 60):02d}"
                        st.write(f"**⏱️ Duration:** {orig_str} original → {recon_str} reconstructed")
                    
                    res_orig = results.get('resolution_original', '?x?')
                    res_recon = results.get('resolution_reconstructed', '?x?')
                    st.write(f"**📐 Resolution:** {res_orig} original → {res_recon} reconstructed")
                    
                    fps_orig = results.get('fps_original')
                    fps_recon = results.get('fps_reconstructed')
                    if fps_orig and fps_recon:
                        st.write(f"**🎬 Frame Rate:** {fps_orig} fps original → {fps_recon} fps reconstructed")
                    
                    st.markdown("---")
                    
                    # Compression type
                    is_lossless = results.get('is_lossless', False)
                    if is_lossless:
                        st.success("🎥 **VISUALLY LOSSLESS** - Excellent quality, no visible differences")
                    else:
                        st.info("📹 **LOSSY** - Some visual information removed for compression")
                    
                    st.markdown("---")
                    
                    # Video quality metrics
                    st.markdown("##### 📊 Video Quality")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        psnr = results.get('psnr')
                        if psnr is not None:
                            if psnr == float('inf'):
                                st.metric("PSNR", "∞ dB", delta="Perfect")
                            else:
                                delta = "Excellent" if psnr > 40 else ("Very Good" if psnr > 35 else None)
                                st.metric("PSNR", f"{psnr:.1f} dB", delta=delta)
                    
                    with metric_col2:
                        ssim = results.get('ssim')
                        if ssim is not None:
                            delta = "Excellent" if ssim > 0.98 else ("Very Good" if ssim > 0.95 else None)
                            st.metric("SSIM", f"{ssim:.4f}", delta=delta)
                    
                    # Interpretation
                    psnr_val = results.get('psnr', 0) or 0
                    ssim_val = results.get('ssim', 0) or 0
                    
                    if is_lossless or psnr_val == float('inf'):
                        st.success("✅ **Perfect** - Video is frame-perfect identical")
                    elif psnr_val > 40 and ssim_val > 0.98:
                        st.success("✅ **Excellent** - Imperceptible quality difference")
                    elif psnr_val > 30 and ssim_val > 0.95:
                        st.info("ℹ️ **Good** - Minor differences, acceptable for most uses")
                    else:
                        st.warning("⚠️ **Fair** - Noticeable video quality differences")
            
            else:  # binary
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    accuracy = results.get('accuracy', 0)
                    st.metric("Byte Accuracy", f"{accuracy:.4f}%")
                with metric_col2:
                    ber = results.get('bit_error_rate', 0)
                    st.metric("Bit Error Rate", f"{ber:.2e}")
                
                st.markdown("##### 📦 Details")
                st.write(f"**Byte Matches:** {results.get('byte_matches', 0):,} / {results.get('total_bytes', 0):,}")
                st.write(f"**Original Size:** {results.get('original_size', 0):,} bytes")
                st.write(f"**Reconstructed Size:** {results.get('reconstructed_size', 0):,} bytes")
            
            # Match status
            st.markdown("---")
            if results.get('match', False):
                st.success("✅ **Perfect Match!** The reconstructed data is identical to the original.")
            else:
                if data_type == 'pdf':
                    # For PDF, check if effectively equivalent
                    text_pres = results.get('text_preservation', 0) or 0
                    visual_ssim = results.get('visual_ssim') or 0
                    if text_pres > 99 and (visual_ssim is None or visual_ssim > 0.98):
                        st.success("✅ **Effectively Identical** - Text and visual content preserved.")
                    else:
                        st.warning("⚠️ **Data differs** - Some content changed due to compression.")
                elif data_type == 'video':
                    # For video, check if quality is preserved
                    psnr = results.get('psnr', 0) or 0
                    ssim = results.get('ssim', 0) or 0
                    if psnr > 40 and ssim > 0.98:
                        st.success("✅ **Effectively Identical** - Video quality preserved.")
                    elif psnr > 30 and ssim > 0.95:
                        st.info("ℹ️ **Good Quality** - Minor visual compression artifacts.")
                    else:
                        st.warning("⚠️ **Data differs** - Visible differences due to compression.")
                else:
                    st.warning("⚠️ **Data differs** from the original. Check the metrics above for details.")
        
        else:
            st.info("👈 Load both files and click 'Compare' to see results here.")


def ngs_preparation_mode():
    """NGS Preparation mode: Fragment DNA and prepare for sequencing."""
    st.header("🧫 NGS Preparation Mode")
    st.markdown("Prepare DNA sequences for Next-Generation Sequencing with proper fragment structure.")
    
    # Constants for NGS structure
    DEFAULT_FORWARD_PRIMER = "ACACGACGCTCTTCCGATCT"  # 20 nt
    DEFAULT_REVERSE_PRIMER = "AGATCGGAAGAGCACACGTCT"  # 21 nt
    INDEX_LENGTH = 8  # 8 nt strand index
    PRIMER_OVERHEAD = len(DEFAULT_FORWARD_PRIMER) + INDEX_LENGTH + len(DEFAULT_REVERSE_PRIMER)  # 49 nt
    MIN_TOTAL_LENGTH = 100
    MAX_TOTAL_LENGTH = 150
    MIN_PAYLOAD = MIN_TOTAL_LENGTH - PRIMER_OVERHEAD  # 51 nt
    MAX_PAYLOAD = MAX_TOTAL_LENGTH - PRIMER_OVERHEAD  # 101 nt
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input")
        
        # Data source selection
        input_source = st.radio(
            "DNA Sequence Source",
            options=["From Encode Mode", "From Randomization Mode", "Upload DNA File", "Enter Directly"],
            horizontal=False,
            key="ngs_input_source"
        )
        
        dna_sequence = None
        
        if input_source == "From Encode Mode":
            if st.session_state.encoded_dna:
                dna_sequence = st.session_state.encoded_dna
                st.success(f"✅ Loaded DNA sequence ({len(dna_sequence):,} nt) from Encode mode")
            else:
                st.warning("⚠️ No DNA sequence available from Encode mode. Please encode a file first.")
        
        elif input_source == "From Randomization Mode":
            if st.session_state.randomized_dna:
                dna_sequence = st.session_state.randomized_dna
                st.success(f"✅ Loaded DNA sequence ({len(dna_sequence):,} nt) from Randomization mode")
            else:
                st.warning("⚠️ No DNA sequence available from Randomization mode.")
        
        elif input_source == "Upload DNA File":
            uploaded_dna = st.file_uploader(
                "Upload DNA sequence file (.txt)",
                type=['txt'],
                key="ngs_upload"
            )
            if uploaded_dna:
                dna_sequence = uploaded_dna.read().decode('utf-8').strip()
                dna_sequence = ''.join(c for c in dna_sequence.upper() if c in 'ATGC')
                if validate_dna_sequence(dna_sequence):
                    st.success(f"✅ Loaded valid DNA sequence ({len(dna_sequence):,} nt)")
                else:
                    st.error("❌ Invalid DNA sequence. Only A, T, G, C characters allowed.")
                    dna_sequence = None
        
        else:  # Enter Directly
            dna_input = st.text_area(
                "Enter DNA sequence",
                height=150,
                placeholder="Paste your DNA sequence here (A, T, G, C only)...",
                key="ngs_direct_input"
            )
            if dna_input:
                dna_sequence = ''.join(c for c in dna_input.upper() if c in 'ATGC')
                if len(dna_sequence) > 0:
                    st.success(f"✅ Valid DNA sequence ({len(dna_sequence):,} nt)")
                else:
                    st.warning("⚠️ No valid nucleotides found.")
                    dna_sequence = None
        
        if dna_sequence:
            st.markdown("---")
            st.markdown("##### 📊 Input Sequence Statistics")
            
            input_chars = calculate_dna_characteristics(dna_sequence)
            
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.metric("Total Length", f"{input_chars['length']:,} nt")
                st.metric("GC Ratio", f"{input_chars['gc_ratio']:.4f}")
            with stat_col2:
                st.metric("Max Homopolymer", f"{input_chars['max_homopolymer']} nt")
                st.metric("Shannon Entropy", f"{input_chars['shannon_entropy']:.4f}")
            
            st.markdown("---")
            
            # Fragmentation settings
            st.markdown("##### ⚙️ Fragmentation Settings")
            
            # Explain the structure
            with st.expander("📐 NGS Fragment Structure", expanded=False):
                st.markdown(f"""
                Each fragment has the following structure (5' → 3'):
                
                | Component | Length | Sequence |
                |-----------|--------|----------|
                | Forward Primer | 20 nt | `{DEFAULT_FORWARD_PRIMER}` |
                | Strand Index | 8 nt | Unique per fragment |
                | **Payload** | **X nt** | Your encoded data |
                | Reverse Primer | 21 nt | `{DEFAULT_REVERSE_PRIMER}` |
                
                **Total length = 49 + X nucleotides**
                
                For optimal NGS (100-150 nt total), payload should be **{MIN_PAYLOAD}-{MAX_PAYLOAD} nt**.
                """)
            
            # Payload length selection
            suggested_payload = min(MAX_PAYLOAD, max(MIN_PAYLOAD, 80))  # Default to 80 nt
            
            payload_length = st.slider(
                "Payload Length (X) per Fragment",
                min_value=MIN_PAYLOAD,
                max_value=MAX_PAYLOAD,
                value=suggested_payload,
                step=1,
                help=f"Each fragment will contain X nucleotides of your data. Total fragment length will be {PRIMER_OVERHEAD} + X."
            )
            
            total_fragment_length = PRIMER_OVERHEAD + payload_length
            num_full_fragments = len(dna_sequence) // payload_length
            remainder = len(dna_sequence) % payload_length
            
            # Show fragment calculation details
            st.markdown("##### 📊 Fragmentation Calculation")
            
            calc_col1, calc_col2 = st.columns(2)
            with calc_col1:
                st.metric("Full Fragments", f"{num_full_fragments}")
                st.metric("Standard Fragment Length", f"{total_fragment_length} nt")
            with calc_col2:
                if remainder > 0:
                    st.metric("Remainder", f"{remainder} nt")
                    last_frag_length = PRIMER_OVERHEAD + remainder
                    st.metric("Last Fragment Length", f"{last_frag_length} nt")
                else:
                    st.metric("Remainder", "0 nt")
                    st.metric("Last Fragment Length", f"{total_fragment_length} nt (same)")
            
            # Handle remainder option
            if remainder > 0:
                st.markdown("---")
                st.markdown("##### ⚠️ Last Fragment Handling")
                st.warning(f"The DNA sequence ({len(dna_sequence):,} nt) does not divide evenly by payload length ({payload_length} nt). The last fragment will have only **{remainder} nt** of payload.")
                
                last_fragment_option = st.radio(
                    "How to handle the last fragment?",
                    options=[
                        f"Keep shorter (last fragment: {PRIMER_OVERHEAD + remainder} nt total)",
                        f"Pad to full length (add {payload_length - remainder} nt padding)"
                    ],
                    index=0,
                    key="last_fragment_option"
                )
                
                pad_last_fragment = "Pad" in last_fragment_option
                
                # Option to separate last fragment
                separate_last = st.checkbox(
                    "Download last fragment separately",
                    value=False,
                    help="Creates a separate file for the shorter/padded last fragment"
                )
            else:
                pad_last_fragment = False
                separate_last = False
                st.success("✅ DNA sequence divides evenly - all fragments will have the same length.")
            
            st.markdown("---")
            
            # Primer customization
            st.markdown("##### 🔬 Primer Sequences")
            
            # Check if primers available from Randomization mode
            randomization_primers_available = (
                hasattr(st.session_state, 'randomization_primers') and 
                st.session_state.randomization_primers and
                'forward' in st.session_state.randomization_primers and
                'reverse' in st.session_state.randomization_primers
            )
            
            primer_source = st.radio(
                "Primer Source",
                options=["Default (Illumina-compatible)", "From Randomization Mode", "Custom"],
                index=0,
                horizontal=True
            )
            
            if primer_source == "Default (Illumina-compatible)":
                forward_primer = DEFAULT_FORWARD_PRIMER
                reverse_primer = DEFAULT_REVERSE_PRIMER
                st.code(f"Forward (5'): {forward_primer}\nReverse (3'): {reverse_primer}", language=None)
            
            elif primer_source == "From Randomization Mode":
                if randomization_primers_available:
                    forward_primer = st.session_state.randomization_primers['forward'][:20]  # Take first 20 nt
                    reverse_primer = st.session_state.randomization_primers['reverse'][:21]  # Take first 21 nt
                    st.success("✅ Loaded primers from Randomization mode")
                    st.code(f"Forward (5'): {forward_primer}\nReverse (3'): {reverse_primer}", language=None)
                else:
                    st.error("❌ No primers found from Randomization mode. Please randomize a sequence first or choose another option.")
                    forward_primer = DEFAULT_FORWARD_PRIMER
                    reverse_primer = DEFAULT_REVERSE_PRIMER
            
            else:  # Custom
                primer_col1, primer_col2 = st.columns(2)
                with primer_col1:
                    forward_primer = st.text_input(
                        "Forward Primer (5')",
                        value=DEFAULT_FORWARD_PRIMER,
                        help="20 nucleotides recommended"
                    ).upper().replace(" ", "")
                with primer_col2:
                    reverse_primer = st.text_input(
                        "Reverse Primer (3')",
                        value=DEFAULT_REVERSE_PRIMER,
                        help="21 nucleotides recommended"
                    ).upper().replace(" ", "")
                
                # Validate primers
                if not all(c in 'ATGC' for c in forward_primer):
                    st.error("❌ Forward primer contains invalid characters")
                    forward_primer = DEFAULT_FORWARD_PRIMER
                if not all(c in 'ATGC' for c in reverse_primer):
                    st.error("❌ Reverse primer contains invalid characters")
                    reverse_primer = DEFAULT_REVERSE_PRIMER
            
            st.markdown("---")
            
            # Generate fragments button
            if st.button("🧬 Generate NGS Fragments", type="primary", use_container_width=True):
                with st.spinner("Fragmenting and analyzing sequences..."):
                    # Generate fragments
                    fragments, last_fragment_info = generate_ngs_fragments(
                        dna_sequence=dna_sequence,
                        payload_length=payload_length,
                        forward_primer=forward_primer,
                        reverse_primer=reverse_primer,
                        pad_last=pad_last_fragment
                    )
                    
                    # Store in session state
                    st.session_state.ngs_fragments = fragments
                    st.session_state.ngs_settings = {
                        'payload_length': payload_length,
                        'forward_primer': forward_primer,
                        'reverse_primer': reverse_primer,
                        'total_length': total_fragment_length,
                        'num_fragments': len(fragments),
                        'has_short_last': last_fragment_info['is_short'],
                        'last_fragment_padded': last_fragment_info['is_padded'],
                        'separate_last': separate_last if remainder > 0 else False
                    }
                    
                    st.success(f"✅ Generated {len(fragments)} NGS fragments!")
    
    with col2:
        st.subheader("📤 Output")
        
        if hasattr(st.session_state, 'ngs_fragments') and st.session_state.ngs_fragments:
            fragments = st.session_state.ngs_fragments
            settings = st.session_state.ngs_settings
            
            # Summary statistics
            st.markdown("##### 📊 Fragment Summary")
            
            sum_col1, sum_col2, sum_col3 = st.columns(3)
            with sum_col1:
                st.metric("Total Fragments", f"{len(fragments):,}")
            with sum_col2:
                st.metric("Fragment Length", f"{settings['total_length']} nt")
            with sum_col3:
                avg_score = sum(f['score_value'] for f in fragments) / len(fragments)
                st.metric("Avg. Score", f"{avg_score:.1f}/100")
            
            # Score distribution
            easy_count = sum(1 for f in fragments if f['difficulty'] == 'Easy')
            medium_count = sum(1 for f in fragments if f['difficulty'] == 'Medium')
            hard_count = sum(1 for f in fragments if f['difficulty'] == 'Hard')
            
            st.markdown("##### 🎯 Sequencing Difficulty Distribution")
            
            dist_col1, dist_col2, dist_col3 = st.columns(3)
            with dist_col1:
                st.success(f"✅ Easy: {easy_count} ({easy_count/len(fragments)*100:.1f}%)")
            with dist_col2:
                st.warning(f"⚠️ Medium: {medium_count} ({medium_count/len(fragments)*100:.1f}%)")
            with dist_col3:
                st.error(f"❌ Hard: {hard_count} ({hard_count/len(fragments)*100:.1f}%)")
            
            st.markdown("---")
            
            # Fragment table
            st.markdown("##### 📋 Fragment Table")
            
            # Create DataFrame for display
            import pandas as pd
            
            table_data = []
            for f in fragments:
                table_data.append({
                    'No.': f['number'],
                    'Name': f['name'],
                    'Length': f['length'],
                    'GC%': f"{f['gc_ratio']*100:.1f}%",
                    'Max Homo': f['max_homopolymer'],
                    'Entropy': f"{f['entropy']:.3f}",
                    'Score': f"{f['score_value']}/100",
                    'Difficulty': f['difficulty']
                })
            
            df_display = pd.DataFrame(table_data)
            
            # Display options
            show_sequence = st.checkbox("Show full sequences in table", value=False)
            
            if show_sequence:
                # Add sequence column
                for i, f in enumerate(fragments):
                    table_data[i]['Sequence'] = f['sequence']
                df_display = pd.DataFrame(table_data)
            
            # Display table with pagination
            rows_per_page = st.selectbox("Rows per page", options=[10, 25, 50, 100], index=0)
            
            total_pages = (len(df_display) + rows_per_page - 1) // rows_per_page
            
            if total_pages > 1:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
            else:
                page = 1
            
            start_idx = (page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(df_display))
            
            st.dataframe(
                df_display.iloc[start_idx:end_idx],
                use_container_width=True,
                hide_index=True
            )
            
            st.caption(f"Showing rows {start_idx + 1}-{end_idx} of {len(df_display)}")
            
            st.markdown("---")
            
            # Detailed view of specific fragment
            st.markdown("##### 🔍 Fragment Detail View")
            
            selected_fragment = st.selectbox(
                "Select fragment to view",
                options=[f['name'] for f in fragments],
                index=0
            )
            
            selected_f = next(f for f in fragments if f['name'] == selected_fragment)
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("**Sequence Structure:**")
                st.code(f"""
5' ─┬─ Forward Primer ({len(settings['forward_primer'])} nt)
    │   {settings['forward_primer']}
    ├─ Strand Index (8 nt)
    │   {selected_f['index']}
    ├─ Payload ({selected_f['payload_length']} nt)
    │   {selected_f['payload'][:30]}{'...' if len(selected_f['payload']) > 30 else ''}
    └─ Reverse Primer ({len(settings['reverse_primer'])} nt)
        {settings['reverse_primer']}
─── 3'
                """, language=None)
            
            with detail_col2:
                st.markdown("**Characteristics:**")
                
                # Color-coded metrics
                gc = selected_f['gc_ratio']
                homo = selected_f['max_homopolymer']
                score = selected_f['score_value']
                
                gc_color = "green" if 0.4 <= gc <= 0.6 else ("orange" if 0.3 <= gc <= 0.7 else "red")
                homo_color = "green" if homo <= 3 else ("orange" if homo <= 5 else "red")
                score_color = "green" if score >= 70 else ("orange" if score >= 50 else "red")
                
                st.markdown(f"""
                | Metric | Value | Status |
                |--------|-------|--------|
                | GC Ratio | {gc*100:.1f}% | {'✅' if gc_color == 'green' else '⚠️' if gc_color == 'orange' else '❌'} |
                | Max Homopolymer | {homo} nt | {'✅' if homo_color == 'green' else '⚠️' if homo_color == 'orange' else '❌'} |
                | Shannon Entropy | {selected_f['entropy']:.4f} | {'✅' if selected_f['entropy'] > 1.9 else '⚠️'} |
                | **Score** | **{score}/100** | {'✅' if score_color == 'green' else '⚠️' if score_color == 'orange' else '❌'} |
                | **Difficulty** | **{selected_f['difficulty']}** | |
                """)
            
            st.markdown("**Full Sequence:**")
            st.code(selected_f['sequence'], language=None)
            
            st.markdown("---")
            
            # Download section
            st.markdown("##### 💾 Download")
            
            # Show last fragment info if different
            if settings.get('has_short_last') or settings.get('last_fragment_padded'):
                last_frag = fragments[-1]
                if settings.get('has_short_last'):
                    st.info(f"ℹ️ Last fragment ({last_frag['name']}) has different length: **{last_frag['length']} nt** (payload: {last_frag['payload_length']} nt)")
                elif settings.get('last_fragment_padded'):
                    st.info(f"ℹ️ Last fragment ({last_frag['name']}) was padded to full length with balanced nucleotides.")
            
            # Separate main fragments and last fragment if requested
            if settings.get('separate_last') and len(fragments) > 1:
                main_fragments = fragments[:-1]
                last_fragment = [fragments[-1]]
                
                st.markdown("**Main Fragments (uniform length):**")
                main_col1, main_col2 = st.columns(2)
                
                with main_col1:
                    main_csv = create_ngs_csv(main_fragments, settings)
                    st.download_button(
                        label=f"📥 Download Main CSV ({len(main_fragments)} fragments)",
                        data=main_csv,
                        file_name="ngs_fragments_main.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with main_col2:
                    main_fasta = create_ngs_fasta(main_fragments)
                    st.download_button(
                        label=f"📥 Download Main FASTA",
                        data=main_fasta,
                        file_name="ngs_fragments_main.fasta",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                st.markdown("**Last Fragment (different length):**")
                last_col1, last_col2 = st.columns(2)
                
                with last_col1:
                    last_csv = create_ngs_csv(last_fragment, settings)
                    st.download_button(
                        label="📥 Download Last Fragment CSV",
                        data=last_csv,
                        file_name="ngs_fragment_last.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with last_col2:
                    last_fasta = create_ngs_fasta(last_fragment)
                    st.download_button(
                        label="📥 Download Last Fragment FASTA",
                        data=last_fasta,
                        file_name="ngs_fragment_last.fasta",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                st.markdown("---")
                st.markdown("**All Fragments (combined):**")
            
            # Create CSV data for all fragments
            csv_data = create_ngs_csv(fragments, settings)
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                st.download_button(
                    label="📥 Download All CSV",
                    data=csv_data,
                    file_name="ngs_fragments.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with download_col2:
                # Create FASTA format
                fasta_data = create_ngs_fasta(fragments)
                st.download_button(
                    label="📥 Download All FASTA",
                    data=fasta_data,
                    file_name="ngs_fragments.fasta",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Additional options
            with st.expander("📊 Export Options"):
                # Export only easy sequences
                easy_fragments = [f for f in fragments if f['difficulty'] == 'Easy']
                if easy_fragments:
                    easy_csv = create_ngs_csv(easy_fragments, settings)
                    st.download_button(
                        label=f"📥 Download Easy Sequences Only ({len(easy_fragments)} fragments)",
                        data=easy_csv,
                        file_name="ngs_fragments_easy.csv",
                        mime="text/csv"
                    )
                
                # Export sequences only (for synthesis ordering)
                sequences_only = "\n".join([f['sequence'] for f in fragments])
                st.download_button(
                    label="📥 Download Sequences Only (for ordering)",
                    data=sequences_only,
                    file_name="ngs_sequences.txt",
                    mime="text/plain"
                )
        
        else:
            st.info("👈 Configure settings and click 'Generate NGS Fragments' to see results here.")
            
            # Show expected output format
            with st.expander("📋 Expected Output Format"):
                st.markdown("""
                The output table will include:
                
                | Column | Description |
                |--------|-------------|
                | No. | Fragment number (1, 2, 3, ...) |
                | Name | Fragment identifier (seq_1, seq_2, ...) |
                | Sequence | Full fragment sequence with primers |
                | Length | Total nucleotide count |
                | GC% | Guanine + Cytosine percentage |
                | Max Homo | Longest homopolymer run |
                | Entropy | Shannon entropy (information density) |
                | Score | Sequencing compatibility score (0-100) |
                | Difficulty | Easy / Medium / Hard classification |
                """)


def generate_ngs_fragments(dna_sequence: str, payload_length: int,
                           forward_primer: str, reverse_primer: str,
                           pad_last: bool = False) -> tuple:
    """
    Generate NGS-ready fragments from a DNA sequence.
    
    Args:
        dna_sequence: Input DNA sequence to fragment
        payload_length: Length of payload per fragment
        forward_primer: Forward primer binding site
        reverse_primer: Reverse primer binding site
        pad_last: If True, pad the last fragment to full length; if False, keep it shorter
    
    Returns:
        Tuple of (fragments list, last_fragment_info dict)
    """
    import random
    
    fragments = []
    num_fragments = (len(dna_sequence) + payload_length - 1) // payload_length
    remainder = len(dna_sequence) % payload_length
    
    last_fragment_info = {
        'is_short': False,
        'is_padded': False,
        'original_payload_length': 0
    }
    
    for i in range(num_fragments):
        # Extract payload
        start = i * payload_length
        end = min(start + payload_length, len(dna_sequence))
        payload = dna_sequence[start:end]
        original_payload_len = len(payload)
        
        # Handle last fragment
        is_last_fragment = (i == num_fragments - 1)
        if is_last_fragment and len(payload) < payload_length:
            if pad_last:
                # Pad with balanced nucleotides
                padding_needed = payload_length - len(payload)
                padding = generate_balanced_padding(padding_needed)
                payload = payload + padding
                last_fragment_info['is_padded'] = True
            else:
                last_fragment_info['is_short'] = True
            last_fragment_info['original_payload_length'] = original_payload_len
        
        # Generate random index (unique 8-mer avoiding long homopolymers)
        index = generate_strand_index(i)
        
        # Assemble full fragment: Forward + Index + Payload + Reverse
        full_sequence = forward_primer + index + payload + reverse_primer
        
        # Calculate characteristics
        chars = calculate_fragment_characteristics(full_sequence)
        
        # Calculate score and difficulty
        score, difficulty = calculate_sequencing_score(chars)
        
        fragment = {
            'number': i + 1,
            'name': f"seq_{i+1}",
            'sequence': full_sequence,
            'length': len(full_sequence),
            'payload': payload,
            'payload_length': len(payload),
            'original_payload_length': original_payload_len,
            'index': index,
            'gc_ratio': chars['gc_ratio'],
            'max_homopolymer': chars['max_homopolymer'],
            'entropy': chars['entropy'],
            'score_value': score,
            'difficulty': difficulty,
            'is_padded': is_last_fragment and last_fragment_info['is_padded']
        }
        
        fragments.append(fragment)
    
    return fragments, last_fragment_info


def generate_strand_index(fragment_num: int) -> str:
    """Generate a random 8-nucleotide strand index avoiding long homopolymers."""
    import random
    
    # Generate random but reproducible index based on fragment number
    random.seed(fragment_num * 12345)
    bases = "ATGC"
    # Ensure no long homopolymers in index
    index = ""
    for j in range(8):
        if j > 0 and len(index) >= 2 and index[-1] == index[-2]:
            # Avoid 3+ consecutive same bases
            available = [b for b in bases if b != index[-1]]
            index += random.choice(available)
        else:
            index += random.choice(bases)
    return index


def generate_balanced_padding(length: int) -> str:
    """Generate balanced padding to avoid long homopolymers."""
    if length == 0:
        return ""
    
    # Use repeating pattern for balance
    pattern = "ATGC"
    padding = ""
    for i in range(length):
        padding += pattern[i % 4]
    return padding


def calculate_fragment_characteristics(sequence: str) -> dict:
    """Calculate DNA characteristics for a fragment."""
    length = len(sequence)
    
    # GC ratio
    gc_count = sequence.count('G') + sequence.count('C')
    gc_ratio = gc_count / length if length > 0 else 0
    
    # Max homopolymer
    max_homo = 1
    current = 1
    for i in range(1, length):
        if sequence[i] == sequence[i-1]:
            current += 1
            max_homo = max(max_homo, current)
        else:
            current = 1
    
    # Shannon entropy
    import math
    counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    for nt in sequence:
        if nt in counts:
            counts[nt] += 1
    
    entropy = 0
    for count in counts.values():
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)
    
    return {
        'gc_ratio': gc_ratio,
        'max_homopolymer': max_homo,
        'entropy': entropy,
        'length': length
    }


def calculate_sequencing_score(chars: dict) -> tuple:
    """
    Calculate a sequencing compatibility score (0-100) and difficulty rating.
    
    Scoring criteria:
    - GC ratio: 40-60% is ideal (30 points)
    - Homopolymer: ≤3 is ideal, >6 is problematic (35 points)
    - Entropy: Higher is better, >1.9 is good (20 points)
    - Length: 100-150 is ideal (15 points)
    
    Returns:
        Tuple of (score, difficulty_label)
    """
    score = 0
    
    # GC ratio scoring (max 30 points)
    gc = chars['gc_ratio']
    if 0.45 <= gc <= 0.55:
        score += 30  # Ideal
    elif 0.40 <= gc <= 0.60:
        score += 25  # Good
    elif 0.35 <= gc <= 0.65:
        score += 18  # Acceptable
    elif 0.30 <= gc <= 0.70:
        score += 10  # Marginal
    else:
        score += 0   # Problematic
    
    # Homopolymer scoring (max 35 points)
    homo = chars['max_homopolymer']
    if homo <= 2:
        score += 35  # Excellent
    elif homo <= 3:
        score += 30  # Very good
    elif homo <= 4:
        score += 22  # Good
    elif homo <= 5:
        score += 14  # Acceptable
    elif homo <= 6:
        score += 7   # Marginal
    else:
        score += 0   # Problematic
    
    # Entropy scoring (max 20 points)
    entropy = chars['entropy']
    if entropy >= 1.95:
        score += 20  # Excellent diversity
    elif entropy >= 1.90:
        score += 16  # Good
    elif entropy >= 1.80:
        score += 12  # Acceptable
    elif entropy >= 1.70:
        score += 6   # Marginal
    else:
        score += 0   # Low diversity
    
    # Length scoring (max 15 points)
    length = chars['length']
    if 100 <= length <= 150:
        score += 15  # Ideal NGS range
    elif 80 <= length <= 180:
        score += 10  # Acceptable
    elif 60 <= length <= 200:
        score += 5   # Marginal
    else:
        score += 0   # Outside typical range
    
    # Determine difficulty
    if score >= 70:
        difficulty = "Easy"
    elif score >= 50:
        difficulty = "Medium"
    else:
        difficulty = "Hard"
    
    return score, difficulty


def create_ngs_csv(fragments: list, settings: dict) -> str:
    """Create CSV content for NGS fragments."""
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'Number', 'Name', 'Sequence', 'Length', 'Payload_Length',
        'Index', 'GC_Ratio', 'Max_Homopolymer', 'Shannon_Entropy',
        'Score', 'Difficulty'
    ])
    
    # Data rows
    for f in fragments:
        writer.writerow([
            f['number'],
            f['name'],
            f['sequence'],
            f['length'],
            f['payload_length'],
            f['index'],
            f"{f['gc_ratio']:.4f}",
            f['max_homopolymer'],
            f"{f['entropy']:.4f}",
            f['score_value'],
            f['difficulty']
        ])
    
    # Add metadata at the end
    writer.writerow([])
    writer.writerow(['# NGS Fragment Generation Settings'])
    writer.writerow(['# Forward Primer', settings['forward_primer']])
    writer.writerow(['# Reverse Primer', settings['reverse_primer']])
    writer.writerow(['# Payload Length', settings['payload_length']])
    writer.writerow(['# Total Fragments', settings['num_fragments']])
    
    return output.getvalue()


def create_ngs_fasta(fragments: list) -> str:
    """Create FASTA format content for NGS fragments."""
    lines = []
    for f in fragments:
        lines.append(f">{f['name']} length={f['length']} gc={f['gc_ratio']:.3f} score={f['score_value']}")
        # Split sequence into 80-character lines (FASTA standard)
        seq = f['sequence']
        for i in range(0, len(seq), 80):
            lines.append(seq[i:i+80])
    
    return '\n'.join(lines)


def guide_mode():
    """Guide mode: Instructions, Examples, and Help sections."""
    st.header("📚 Guide")
    
    # Section selector with radio buttons for clear navigation
    guide_section = st.radio(
        "Select Section",
        options=["ℹ️ About", "📖 Instructions", "🧪 Examples", "❓ Help"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if guide_section == "ℹ️ About":
        render_about()
    elif guide_section == "📖 Instructions":
        render_instructions()
    elif guide_section == "🧪 Examples":
        render_examples()
    else:  # Help
        render_help()


def render_about():
    """Render the About section with platform introduction."""
    st.subheader("ℹ️ About This Platform")
    
    # Introduction paragraph
    st.markdown("""
    **DNA Data Storage Platform** is a comprehensive tool for encoding, storing, and retrieving digital 
    data using DNA sequences. This platform enables you to convert any digital file—text documents, 
    images, audio, video, or binary data—into synthetic DNA sequences that can be synthesized, stored, 
    and later decoded back to the original format.
    
    The platform uses **quaternary encoding** where each nucleotide (A, T, G, C) represents 2 bits of 
    information, achieving a baseline density of **2 bits per nucleotide**. With intelligent compression 
    algorithms tailored to each file type, the effective information density can reach **20-70+ bits 
    per nucleotide** for highly compressible content.
    
    **Key capabilities include:**
    - **Multi-format support**: Text, images (PNG, JPEG, WebP), audio (WAV, MP3, FLAC), video (MP4, AVI, MKV), PDF, and any binary file
    - **Smart compression**: Automatic file-type detection with optimized algorithms (Brotli for text, WebP for images, H.264/AV1 for video, FLAC/AAC for audio)
    - **Chaos-based encryption**: Secure your DNA sequences using Logistic, Hénon, or Lorenz chaos systems with primer-based keys
    - **NGS preparation**: Fragment sequences for Next-Generation Sequencing with proper primer structure and quality scoring
    - **Quality verification**: Compare original and reconstructed files using SSIM, PSNR, SNR, and correlation metrics
    
    This platform is designed for researchers, developers, and enthusiasts exploring DNA as a medium 
    for long-term, high-density data storage.
    """)
    
    st.markdown("---")
    
    # Visual workflow diagram
    st.markdown("### 🔄 Platform Workflow")
    
    # Create a visual pipeline using columns
    workflow_cols = st.columns(5)
    
    workflow_steps = [
        ("📤", "Encode", "Convert files to DNA", "#4CAF50"),
        ("🔀", "Randomize", "Encrypt with chaos maps", "#2196F3"),
        ("🧫", "NGS Prep", "Fragment for sequencing", "#9C27B0"),
        ("📥", "Decode", "Restore original files", "#FF9800"),
        ("📊", "Compare", "Verify data integrity", "#607D8B")
    ]
    
    for col, (icon, title, desc, color) in zip(workflow_cols, workflow_steps):
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                border: 2px solid {color};
                border-radius: 15px;
                padding: 20px 10px;
                text-align: center;
                min-height: 140px;
            ">
                <div style="font-size: 2.5rem;">{icon}</div>
                <div style="font-weight: bold; color: {color}; font-size: 1.1rem;">{title}</div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 5px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("---")
    
    # Key features highlight
    st.markdown("### ✨ Key Features")
    
    feat_cols = st.columns(2)
    
    with feat_cols[0]:
        st.success("**🗜️ Smart Compression**  \nAutomatic file-type detection with optimized algorithms (Brotli, WebP, H.264, FLAC)")
        st.info("**🔐 Chaos Encryption**  \nSecure your DNA with primer-based encryption using Logistic, Hénon, or Lorenz chaos systems")
    
    with feat_cols[1]:
        st.warning("**📊 Quality Metrics**  \nVerify reconstruction with SSIM, PSNR, SNR, and correlation analysis")
        st.error("**🧬 High Density**  \nAchieve 2+ bits per nucleotide with metadata embedding")


def render_instructions():
    """Render detailed instructions for each mode."""
    st.subheader("📖 Detailed Instructions")
    
    # Mode selector
    instruction_mode = st.selectbox(
        "Select Mode for Instructions",
        options=["Encode Mode", "Randomization Mode", "NGS Preparation Mode", "Decode Mode", "Comparison Mode"],
        index=0
    )
    
    if instruction_mode == "Encode Mode":
        render_encode_instructions()
    elif instruction_mode == "Randomization Mode":
        render_randomization_instructions()
    elif instruction_mode == "NGS Preparation Mode":
        render_ngs_instructions()
    elif instruction_mode == "Decode Mode":
        render_decode_instructions()
    else:
        render_comparison_instructions()


def render_encode_instructions():
    """Detailed encode mode instructions."""
    st.markdown("### 📤 Encode Mode Instructions")
    
    # Visual step-by-step guide
    steps = [
        {
            "num": "1",
            "title": "Upload Your File",
            "desc": "Click the file uploader and select any file from your computer. Supported formats include text, images, audio, video, PDF, and any binary file.",
            "tip": "💡 The platform automatically detects file type and suggests optimal compression.",
            "icon": "📁"
        },
        {
            "num": "2",
            "title": "Review File Information",
            "desc": "After upload, you'll see file details including name, size, extension, and detected type. This helps confirm the correct file was selected.",
            "tip": "💡 File type detection works by both extension and content analysis.",
            "icon": "📋"
        },
        {
            "num": "3",
            "title": "Choose Compression",
            "desc": "Select 'None' for raw encoding or 'Data-Type-Specific' for optimized compression. Each file type uses a specialized algorithm.",
            "tip": "💡 Compression can significantly reduce DNA sequence length, improving storage efficiency.",
            "icon": "⚙️"
        },
        {
            "num": "4",
            "title": "Adjust Quality Settings",
            "desc": "If compression is enabled, adjust quality sliders. Higher quality = larger file, lower quality = smaller file with some loss.",
            "tip": "💡 For archival, use high quality. For space efficiency, use medium quality.",
            "icon": "🎚️"
        },
        {
            "num": "5",
            "title": "Start Encoding",
            "desc": "Click 'Start Encoding' to convert your file to DNA. The process embeds metadata for seamless decoding later.",
            "tip": "💡 Encoding time depends on file size and compression settings.",
            "icon": "🚀"
        },
        {
            "num": "6",
            "title": "Download DNA Sequence",
            "desc": "Once complete, download the DNA sequence as a .txt file. This file contains all information needed to reconstruct your original data.",
            "tip": "💡 Keep this file safe - it's your data in DNA form!",
            "icon": "💾"
        }
    ]
    
    for step in steps:
        with st.expander(f"{step['icon']} Step {step['num']}: {step['title']}", expanded=False):
            st.markdown(step['desc'])
            st.info(step['tip'])
    
    # Supported file types table
    st.markdown("---")
    st.markdown("#### 📁 Supported File Types")
    
    st.markdown("""
    | Category | File Types | Compression Algorithm | Compression Type | Typical Ratio |
    |----------|------------|----------------------|------------------|---------------|
    | **1. Text/Binary** | .txt, .csv, .json, .xml, .html, .md, .py, .js, .c, .cpp, .java, .log, .dat, .bin, .docx | Brotli | Lossless | 2-10x |
    | **2. PDF** | .pdf | PDF Optimization (Ghostscript) | Mixed (lossless text, lossy images) | 1.5-3x |
    | **3. Image** | .png, .jpg, .jpeg, .bmp, .gif, .tiff, .webp, .ico | WebP | Lossy (configurable quality 1-100) | 5-20x |
    | **4. Audio** | .wav, .mp3, .flac, .aac, .ogg, .m4a, .aiff, .wma | FLAC (lossless) / AAC / MP3 (lossy) | Lossless or Lossy (selectable) | 2-10x |
    | **5. Video** | .mp4, .avi, .mkv, .mov, .webm, .flv, .wmv, .mpeg, .mpg, .m4v, .3gp | H.264 / AV1 | Lossy (configurable CRF) | 10-100x |
    """)


def render_randomization_instructions():
    """Detailed randomization mode instructions."""
    st.markdown("### 🔀 Randomization Mode Instructions")
    
    st.markdown("""
    Randomization applies **chaos map encryption** to your DNA sequence. This:
    - 🔐 Encrypts data using primer sequences as keys
    - 📊 Improves DNA synthesis compatibility
    - 🧬 Reduces homopolymer runs (repeated nucleotides)
    """)
    
    steps = [
        {
            "num": "1",
            "title": "Load DNA Sequence",
            "desc": "Choose 'From Encode Mode' to use the just-encoded sequence, or 'Upload DNA File' to load a previously saved sequence.",
            "icon": "📥"
        },
        {
            "num": "2",
            "title": "Review Original Characteristics",
            "desc": "Check the original DNA statistics: length, max homopolymer run, GC ratio, and Shannon entropy.",
            "icon": "📊"
        },
        {
            "num": "3",
            "title": "Choose Randomization Method",
            "desc": "Select 'None' to skip, or 'Chaos Map' for encryption. Chaos maps use mathematical systems for secure randomization.",
            "icon": "🔐"
        },
        {
            "num": "4",
            "title": "Configure Chaos System",
            "desc": "Choose Single System (Logistic, Hénon, or Lorenz) or Multi-System Chain for enhanced security.",
            "icon": "⚙️"
        },
        {
            "num": "5",
            "title": "Set Primer Keys",
            "desc": "Enter forward and reverse primer sequences. These act as encryption keys - you'll need them to decode!",
            "icon": "🔑"
        },
        {
            "num": "6",
            "title": "Apply & Download",
            "desc": "Click 'Apply Randomization' and download the encrypted sequence along with your primer keys.",
            "icon": "💾"
        }
    ]
    
    for step in steps:
        with st.expander(f"{step['icon']} Step {step['num']}: {step['title']}", expanded=False):
            st.markdown(step['desc'])
    
    # Chaos systems comparison
    st.markdown("---")
    st.markdown("#### 🌀 Chaos Systems Comparison")
    
    chaos_data = {
        "System": ["Logistic Map", "Hénon Map", "Lorenz System"],
        "Dimensions": ["1D", "2D", "3D"],
        "Complexity": ["Simple", "Medium", "Complex"],
        "Speed": ["Fastest", "Fast", "Moderate"],
        "Security": ["Basic", "Good", "Best"]
    }
    st.table(chaos_data)


def render_ngs_instructions():
    """Detailed NGS preparation mode instructions."""
    st.markdown("### 🧫 NGS Preparation Mode Instructions")
    
    st.markdown("""
    NGS (Next-Generation Sequencing) Preparation mode fragments your DNA sequence into 
    synthesis-ready oligos with proper structure for sequencing.
    """)
    
    # Fragment structure diagram
    st.markdown("#### 📐 Fragment Structure")
    st.code("""
    5' ─────────────────────────────────────────────────────────── 3'
    
    ┌──────────────┬───────────┬───────────────────┬───────────────┐
    │ Fwd Primer   │  Index    │     Payload       │  Rev Primer   │
    │   (20 nt)    │  (8 nt)   │    (51-101 nt)    │   (21 nt)     │
    └──────────────┴───────────┴───────────────────┴───────────────┘
    
    Total: 100-150 nucleotides (optimal NGS range)
    """, language=None)
    
    steps = [
        {
            "num": "1",
            "title": "Load DNA Sequence",
            "desc": "Import your DNA from Encode mode, Randomization mode, upload a file, or enter directly.",
            "icon": "📥"
        },
        {
            "num": "2",
            "title": "Review Input Statistics",
            "desc": "Check the total length, GC ratio, max homopolymer, and entropy of your input sequence.",
            "icon": "📊"
        },
        {
            "num": "3",
            "title": "Set Payload Length",
            "desc": "Choose payload size (51-101 nt) to achieve total fragment length of 100-150 nt. If the sequence doesn't divide evenly, you can choose to pad or keep the last fragment shorter.",
            "icon": "📏"
        },
        {
            "num": "4",
            "title": "Configure Primers",
            "desc": "Use default Illumina-compatible primers, load from Randomization mode, or enter custom sequences.",
            "icon": "🔬"
        },
        {
            "num": "5",
            "title": "Generate Fragments",
            "desc": "Click 'Generate NGS Fragments' to create all fragments. Each fragment gets a unique random 8-mer index.",
            "icon": "🧬"
        },
        {
            "num": "6",
            "title": "Review & Download",
            "desc": "Check the table, view difficulty scores, and download CSV/FASTA for synthesis ordering. You can also download the last fragment separately if it has different length.",
            "icon": "💾"
        }
    ]
    
    for step in steps:
        with st.expander(f"{step['icon']} Step {step['num']}: {step['title']}", expanded=False):
            st.markdown(step['desc'])
    
    # Scoring explanation
    st.markdown("---")
    st.markdown("#### 🎯 Sequencing Score Criteria")
    
    st.markdown("""
    Each fragment receives a score (0-100) based on synthesis/sequencing compatibility:
    
    | Criterion | Ideal Range | Max Points |
    |-----------|-------------|------------|
    | **GC Ratio** | 45-55% | 30 |
    | **Max Homopolymer** | ≤3 nt | 35 |
    | **Shannon Entropy** | >1.95 | 20 |
    | **Total Length** | 100-150 nt | 15 |
    """)
    
    st.markdown("""
    **Difficulty Ratings:**
    - 🟢 **Easy** (70-100): Straightforward synthesis and sequencing
    - 🟡 **Medium** (50-69): May require optimization
    - 🔴 **Hard** (<50): Challenging synthesis, consider redesign
    """)


def render_decode_instructions():
    """Detailed decode mode instructions."""
    st.markdown("### 📥 Decode Mode Instructions")
    
    st.markdown("""
    Decode mode reverses the encoding process to reconstruct your original file from DNA.
    """)
    
    steps = [
        {
            "num": "1",
            "title": "Load DNA Sequence",
            "desc": "Use 'From Randomization Mode' for recently processed sequences, or 'Upload DNA File' to load a saved sequence.",
            "icon": "📥"
        },
        {
            "num": "2",
            "title": "Check for Randomization",
            "desc": "If the sequence was randomized, you must derandomize first using the same primer keys and chaos settings.",
            "icon": "🔍"
        },
        {
            "num": "3",
            "title": "Derandomize (if needed)",
            "desc": "Enter the exact primer sequences and chaos system configuration used during randomization.",
            "icon": "🔓"
        },
        {
            "num": "4",
            "title": "Start Decoding",
            "desc": "Click 'Start Decoding' to convert DNA back to binary data. Metadata is automatically extracted.",
            "icon": "🚀"
        },
        {
            "num": "5",
            "title": "Verify Results",
            "desc": "Check the decoded file information: size, compression type, and original extension.",
            "icon": "✅"
        },
        {
            "num": "6",
            "title": "Download File",
            "desc": "Download your reconstructed file. It should match the original (or be semantically equivalent for lossy compression).",
            "icon": "💾"
        }
    ]
    
    for step in steps:
        with st.expander(f"{step['icon']} Step {step['num']}: {step['title']}", expanded=False):
            st.markdown(step['desc'])
    
    st.warning("⚠️ **Important:** If you randomized your sequence, you MUST use the exact same primer keys and chaos settings to decode. Without them, your data cannot be recovered!")


def render_comparison_instructions():
    """Detailed comparison mode instructions."""
    st.markdown("### 📊 Comparison Mode Instructions")
    
    st.markdown("""
    Compare original and reconstructed files to verify data integrity and measure quality.
    """)
    
    steps = [
        {
            "num": "1",
            "title": "Load Original File",
            "desc": "Choose 'From Session' to use the file from Encode mode, or 'Upload File' to load from disk.",
            "icon": "📁"
        },
        {
            "num": "2",
            "title": "Load Reconstructed File",
            "desc": "Similarly, load the decoded/reconstructed file for comparison.",
            "icon": "📂"
        },
        {
            "num": "3",
            "title": "Run Comparison",
            "desc": "Click 'Compare Files' to analyze both files using type-appropriate metrics.",
            "icon": "🔬"
        },
        {
            "num": "4",
            "title": "Review Metrics",
            "desc": "Examine quality metrics specific to your file type (SSIM for images, SNR for audio, etc.).",
            "icon": "📊"
        }
    ]
    
    for step in steps:
        with st.expander(f"{step['icon']} Step {step['num']}: {step['title']}", expanded=False):
            st.markdown(step['desc'])
    
    # Metrics by type
    st.markdown("---")
    st.markdown("#### 📈 Quality Metrics by File Type")
    
    metrics_data = {
        "File Type": ["Text", "Image", "Audio", "Video", "PDF", "Binary"],
        "Primary Metric": ["Accuracy %", "SSIM", "SNR (dB)", "PSNR (dB)", "Text Preservation %", "Byte Accuracy %"],
        "Secondary Metric": ["Bit Error Rate", "PSNR (dB)", "Correlation", "SSIM", "Visual SSIM", "Bit Error Rate"],
        "Perfect Score": ["100%", "1.0", "∞ dB", "∞ dB", "100%", "100%"]
    }
    st.table(metrics_data)


def render_examples():
    """Render interactive examples with pre-loaded data."""
    st.subheader("🧪 Interactive Examples")
    st.markdown("Try these pre-loaded examples to see how DNA encoding works!")
    
    # Example selector
    example_type = st.selectbox(
        "Choose Example Type",
        options=["📝 Text Encoding", "🖼️ Image Concept", "🎵 Audio Concept", "🔀 Randomization Demo", "📊 Full Pipeline Demo"],
        index=0
    )
    
    if example_type == "📝 Text Encoding":
        render_text_example()
    elif example_type == "🖼️ Image Concept":
        render_image_concept()
    elif example_type == "🎵 Audio Concept":
        render_audio_concept()
    elif example_type == "🔀 Randomization Demo":
        render_randomization_example()
    else:
        render_pipeline_demo()


def render_text_example():
    """Interactive text encoding example."""
    st.markdown("### 📝 Text to DNA Encoding Example")
    
    st.markdown("""
    This example shows how text is converted to DNA sequences using quaternary encoding.
    """)
    
    # Input text
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Input")
        sample_text = st.text_area(
            "Enter text to encode",
            value="Hello, DNA!",
            height=100,
            key="example_text"
        )
        
        if st.button("🧬 Encode to DNA", key="encode_example"):
            st.session_state.example_encoded = True
    
    with col2:
        st.markdown("#### DNA Mapping")
        st.code("""
Binary → DNA Mapping:
  00 → A (Adenine)
  01 → T (Thymine)
  10 → G (Guanine)
  11 → C (Cytosine)
        """, language=None)
    
    # Show encoding process
    if st.session_state.get('example_encoded', False) or sample_text:
        st.markdown("---")
        st.markdown("#### 🔬 Encoding Process")
        
        # Step 1: Text to Binary
        text_bytes = sample_text.encode('utf-8')
        binary_str = ''.join(format(b, '08b') for b in text_bytes)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Step 1: Text → Bytes**")
            st.code(f"'{sample_text}'\n↓\n{list(text_bytes)[:10]}{'...' if len(text_bytes) > 10 else ''}")
        
        with col2:
            st.markdown("**Step 2: Bytes → Binary**")
            display_binary = binary_str[:48] + ('...' if len(binary_str) > 48 else '')
            st.code(f"{display_binary}")
        
        with col3:
            st.markdown("**Step 3: Binary → DNA**")
            # Convert to DNA
            dna_map = {'00': 'A', '01': 'T', '10': 'G', '11': 'C'}
            dna = ''
            for i in range(0, len(binary_str) - 1, 2):
                dna += dna_map.get(binary_str[i:i+2], 'A')
            display_dna = dna[:24] + ('...' if len(dna) > 24 else '')
            st.code(f"{display_dna}")
        
        # Results
        st.markdown("---")
        st.markdown("#### 📊 Results")
        
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Original Size", f"{len(text_bytes)} bytes")
        res_col2.metric("Binary Length", f"{len(binary_str)} bits")
        res_col3.metric("DNA Length", f"{len(dna)} nt")
        res_col4.metric("Density", f"{len(binary_str)/len(dna):.2f} bits/nt")
        
        # Full DNA sequence
        st.markdown("**Full DNA Sequence:**")
        st.code(dna, language=None)
        
        st.success("✅ This DNA sequence contains all the information to reconstruct the original text!")


def render_image_concept():
    """Conceptual image encoding example."""
    st.markdown("### 🖼️ Image Encoding Concept")
    
    st.markdown("""
    Images are encoded similarly to text, but with optional WebP compression to reduce size.
    """)
    
    # Visual diagram
    st.markdown("#### 📐 Image Encoding Pipeline")
    
    pipeline_cols = st.columns(5)
    
    steps = [
        ("🖼️", "Image", "PNG/JPG"),
        ("🗜️", "Compress", "WebP"),
        ("💾", "Bytes", "Binary"),
        ("🧬", "Encode", "DNA"),
        ("📥", "Store", "A/T/G/C")
    ]
    
    for col, (icon, label, detail) in zip(pipeline_cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-weight: bold;">{label}</div>
                <div style="font-size: 0.8rem; color: #666;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Example calculations
    st.markdown("#### 📊 Example: 100x100 RGB Image")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        st.markdown("**Without Compression:**")
        st.markdown("""
        - Pixels: 100 × 100 = 10,000
        - Bytes: 10,000 × 3 (RGB) = 30,000
        - Bits: 30,000 × 8 = 240,000
        - DNA: 240,000 ÷ 2 = **120,000 nt**
        """)
    
    with calc_col2:
        st.markdown("**With WebP Compression (85% quality):**")
        st.markdown("""
        - Compressed size: ~3,000 bytes (typical)
        - Bits: 3,000 × 8 = 24,000
        - DNA: 24,000 ÷ 2 = **12,000 nt**
        - **Savings: 90% reduction!**
        """)
    
    st.info("💡 **Tip:** WebP compression is lossy but visually imperceptible at quality 85+. Use quality 100 for lossless.")


def render_audio_concept():
    """Conceptual audio encoding example."""
    st.markdown("### 🎵 Audio Encoding Concept")
    
    st.markdown("""
    Audio files can be compressed using lossless (FLAC) or lossy (AAC/MP3) algorithms.
    """)
    
    # Comparison table
    st.markdown("#### 🎛️ Audio Compression Options")
    
    audio_options = {
        "Method": ["FLAC", "AAC 256kbps", "AAC 128kbps", "MP3 320kbps", "MP3 192kbps"],
        "Type": ["Lossless", "Lossy", "Lossy", "Lossy", "Lossy"],
        "Quality": ["Perfect", "Excellent", "Very Good", "Excellent", "Good"],
        "Typical Reduction": ["50-70%", "80-90%", "90-95%", "75-85%", "85-90%"],
        "Best For": ["Archival", "High quality", "Streaming", "Compatibility", "Small files"]
    }
    st.table(audio_options)
    
    # Visual example
    st.markdown("---")
    st.markdown("#### 📊 Example: 3-minute WAV Audio (44.1kHz, Stereo)")
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        st.metric("Original WAV", "31.5 MB")
        st.caption("Uncompressed PCM")
    
    with example_col2:
        st.metric("FLAC", "~18 MB")
        st.caption("Lossless, perfect quality")
    
    with example_col3:
        st.metric("AAC 128kbps", "~2.9 MB")
        st.caption("Lossy, very good quality")
    
    st.success("✅ All formats can be decoded back, but lossy formats have slight quality reduction.")


def render_randomization_example():
    """Interactive randomization demonstration."""
    st.markdown("### 🔀 Randomization Demo")
    
    st.markdown("""
    See how chaos map randomization transforms DNA sequences to improve synthesis compatibility.
    """)
    
    # Sample DNA input
    sample_dna = "AAAATTTTGGGGCCCCAAAATTTT"
    
    st.markdown("#### Original Sequence (High Homopolymer)")
    st.code(sample_dna, language=None)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Max Homopolymer", "4 nt")
        st.caption("(4 consecutive identical bases)")
    with col2:
        st.metric("GC Content", "50%")
        st.caption("(balanced)")
    
    st.markdown("---")
    
    # Simulated randomization
    st.markdown("#### After Chaos Map Randomization")
    
    # Simple demonstration of what randomization does
    import random
    random.seed(42)  # Reproducible
    bases = list("ATGC")
    randomized = ''.join(random.choice(bases) for _ in sample_dna)
    
    st.code(randomized, language=None)
    
    # Calculate max homopolymer
    max_homo = 1
    current = 1
    for i in range(1, len(randomized)):
        if randomized[i] == randomized[i-1]:
            current += 1
            max_homo = max(max_homo, current)
        else:
            current = 1
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Max Homopolymer", f"{max_homo} nt", delta=f"-{4-max_homo} nt")
    with col2:
        gc = (randomized.count('G') + randomized.count('C')) / len(randomized) * 100
        st.metric("GC Content", f"{gc:.0f}%")
    
    st.success("✅ Randomization reduces homopolymers, making DNA easier to synthesize!")
    
    st.markdown("---")
    st.markdown("#### 🔐 How It Works")
    
    st.markdown("""
    1. **Primer Keys** act as encryption keys (forward + reverse sequences)
    2. **Chaos System** generates pseudo-random permutation based on keys
    3. **Nucleotide Shuffling** rearranges bases while preserving information
    4. **Reversible** with the same keys and settings
    """)
    
    st.warning("⚠️ **Important:** Without the correct primer keys, randomized DNA cannot be decoded!")


def render_pipeline_demo():
    """Full pipeline demonstration."""
    st.markdown("### 📊 Full Pipeline Demo")
    
    st.markdown("""
    This demo shows the complete encode → randomize → decode pipeline with a sample message.
    """)
    
    # Sample message
    demo_message = "DNA stores data!"
    
    st.markdown("#### 🔄 Complete Pipeline")
    
    # Create pipeline visualization
    stages = [
        {
            "stage": "1. Original",
            "icon": "📝",
            "content": demo_message,
            "info": f"{len(demo_message)} characters"
        },
        {
            "stage": "2. Encoded",
            "icon": "🧬",
            "content": "ATGC" * 8 + "...",  # Simplified
            "info": f"~{len(demo_message) * 4} nucleotides"
        },
        {
            "stage": "3. Randomized",
            "icon": "🔀",
            "content": "GTAC" * 8 + "...",  # Simplified
            "info": "Encrypted with primers"
        },
        {
            "stage": "4. Decoded",
            "icon": "📥",
            "content": demo_message,
            "info": "Perfect reconstruction"
        }
    ]
    
    # Display as columns
    cols = st.columns(4)
    for col, stage in zip(cols, stages):
        with col:
            st.markdown(f"**{stage['stage']}** {stage['icon']}")
            st.code(stage['content'], language=None)
            st.caption(stage['info'])
    
    st.success("✅ Data integrity verified - original message perfectly reconstructed!")
    
    # Statistics
    st.markdown("---")
    st.markdown("#### 📈 Pipeline Statistics")
    
    stat_cols = st.columns(4)
    stat_cols[0].metric("Encoding Density", "2.0 bits/nt")
    stat_cols[1].metric("Compression", "None (demo)")
    stat_cols[2].metric("Encryption", "Hénon Map")
    stat_cols[3].metric("Data Integrity", "100%")


def render_help():
    """Render comprehensive help section."""
    st.subheader("❓ Help & FAQ")
    
    # Help topic selector
    help_topic = st.selectbox(
        "Select Help Topic",
        options=[
            "🧬 About DNA Data Storage",
            "📁 Supported File Types",
            "🗜️ Compression Algorithms",
            "🔐 Chaos Map Encryption",
            "🧫 NGS Preparation",
            "📊 Quality Metrics",
            "⚠️ Troubleshooting",
            "❓ Frequently Asked Questions"
        ],
        index=0
    )
    
    if help_topic == "🧬 About DNA Data Storage":
        render_about_help()
    elif help_topic == "📁 Supported File Types":
        render_filetypes_help()
    elif help_topic == "🗜️ Compression Algorithms":
        render_compression_help()
    elif help_topic == "🔐 Chaos Map Encryption":
        render_encryption_help()
    elif help_topic == "🧫 NGS Preparation":
        render_ngs_help()
    elif help_topic == "📊 Quality Metrics":
        render_metrics_help()
    elif help_topic == "⚠️ Troubleshooting":
        render_troubleshooting_help()
    else:
        render_faq_help()


def render_about_help():
    """About DNA data storage."""
    st.markdown("### 🧬 About DNA Data Storage")
    
    st.markdown("""
    DNA data storage is a revolutionary technology that encodes digital information into 
    synthetic DNA sequences. This platform implements a practical system for converting 
    any digital file into DNA.
    """)
    
    st.markdown("#### 🔬 How It Works")
    
    st.markdown("""
    DNA consists of four nucleotide bases: **A**denine, **T**hymine, **G**uanine, and **C**ytosine.
    
    Since there are 4 possible bases, each nucleotide can represent **2 bits** of information:
    """)
    
    mapping_col1, mapping_col2 = st.columns(2)
    with mapping_col1:
        st.code("""
Binary  →  DNA
  00    →   A
  01    →   T
  10    →   G
  11    →   C
        """, language=None)
    
    with mapping_col2:
        st.markdown("""
        **Theoretical density:** 2 bits/nucleotide
        
        **With compression:** Up to 20-70+ bits/nucleotide
        
        **1 gram of DNA** can store ~215 petabytes!
        """)
    
    st.markdown("#### ✨ Advantages of DNA Storage")
    
    adv_cols = st.columns(3)
    with adv_cols[0]:
        st.success("**📦 Density**  \n1 million times denser than flash storage")
    with adv_cols[1]:
        st.info("**⏳ Durability**  \nStable for thousands of years in proper conditions")
    with adv_cols[2]:
        st.warning("**🔌 Energy**  \nNo power needed for long-term storage")


def render_filetypes_help():
    """Supported file types help."""
    st.markdown("### 📁 Supported File Types")
    
    st.markdown("This platform supports virtually any file type. Here's how different types are handled:")
    
    # File types table
    file_types = {
        "Category": ["Text", "Image", "Audio", "Video", "PDF", "Binary"],
        "Extensions": [
            ".txt, .csv, .json, .xml, .html, .md, .py, .js",
            ".png, .jpg, .jpeg, .bmp, .gif, .tiff, .webp",
            ".wav, .mp3, .flac, .aac, .ogg, .m4a, .aiff",
            ".mp4, .avi, .mkv, .mov, .webm, .flv, .wmv",
            ".pdf",
            "Any other file"
        ],
        "Compression": [
            "Brotli (lossless)",
            "WebP (lossy/lossless)",
            "FLAC/AAC/MP3",
            "H.264/AV1",
            "PDF optimization",
            "Brotli (lossless)"
        ],
        "Detection": [
            "Extension + UTF-8 check",
            "Magic bytes + extension",
            "Magic bytes + extension",
            "Magic bytes + extension",
            "%PDF header",
            "Fallback"
        ]
    }
    st.table(file_types)
    
    st.info("💡 **Tip:** The platform auto-detects file types even if the extension is wrong or missing, using file signature (magic bytes) analysis.")


def render_compression_help():
    """Compression algorithms help."""
    st.markdown("### 🗜️ Compression Algorithms")
    
    # Brotli
    with st.expander("📝 Brotli (Text/Binary)", expanded=True):
        st.markdown("""
        **Type:** Lossless  
        **Best for:** Text, JSON, XML, code files
        
        Brotli is a modern compression algorithm developed by Google that achieves 
        excellent compression ratios for text data.
        
        **Quality Settings (0-11):**
        - 0-3: Fast compression, larger output
        - 4-7: Balanced
        - 8-11: Best compression, slower
        
        **Typical compression:** 2-10x for text files
        """)
    
    # WebP
    with st.expander("🖼️ WebP (Images)"):
        st.markdown("""
        **Type:** Lossy or Lossless  
        **Best for:** Photos, graphics, screenshots
        
        WebP is an image format that provides superior compression compared to JPEG and PNG.
        
        **Quality Settings (1-100):**
        - 1-50: High compression, visible artifacts
        - 51-85: Balanced (recommended)
        - 86-99: High quality, minimal loss
        - 100: Lossless mode
        
        **Typical compression:** 5-20x compared to PNG
        """)
    
    # Audio
    with st.expander("🎵 Audio Codecs (FLAC/AAC/MP3)"):
        st.markdown("""
        **FLAC (Free Lossless Audio Codec)**
        - Type: Lossless
        - Compression: 50-70% of original
        - Best for: Archival, studio recordings
        
        **AAC (Advanced Audio Coding)**
        - Type: Lossy
        - Better than MP3 at same bitrate
        - Best for: Efficient storage, streaming
        
        **MP3 (MPEG Audio Layer III)**
        - Type: Lossy
        - Universal compatibility
        - Best for: Maximum compatibility
        """)
    
    # Video
    with st.expander("🎬 Video Codecs (H.264/AV1)"):
        st.markdown("""
        **H.264/AVC**
        - Type: Lossy
        - Universal playback support
        - Fast encoding
        - CRF 18-28 recommended
        
        **AV1**
        - Type: Lossy
        - 30-50% better compression than H.264
        - Slower encoding
        - Best for: Maximum compression
        """)


def render_encryption_help():
    """Chaos map encryption help."""
    st.markdown("### 🔐 Chaos Map Encryption")
    
    st.markdown("""
    This platform uses **deterministic chaos systems** to encrypt DNA sequences. 
    The encryption is reversible when you have the correct keys (primer sequences).
    """)
    
    # Chaos systems
    with st.expander("🌀 Logistic Map (1D)", expanded=True):
        st.markdown("""
        **Equation:** xₙ₊₁ = r × xₙ × (1 - xₙ)
        
        - Simplest chaos system
        - Fast computation
        - Good for basic encryption
        - Single parameter (r)
        """)
    
    with st.expander("🔄 Hénon Map (2D)"):
        st.markdown("""
        **Equations:**  
        xₙ₊₁ = 1 - a × xₙ² + yₙ  
        yₙ₊₁ = b × xₙ
        
        - Two-dimensional system
        - Better security than Logistic
        - Default choice for most uses
        - Parameters: a, b
        """)
    
    with st.expander("🌊 Lorenz System (3D)"):
        st.markdown("""
        **Equations:**  
        dx/dt = σ(y - x)  
        dy/dt = x(ρ - z) - y  
        dz/dt = xy - βz
        
        - Three-dimensional system
        - Highest security
        - More computationally intensive
        - Parameters: σ, ρ, β
        """)
    
    st.markdown("#### 🔑 Primer Keys")
    st.markdown("""
    Primer sequences act as encryption keys:
    
    - **Forward Primer:** Seeds the chaos system initial conditions
    - **Reverse Primer:** Additional entropy for enhanced security
    - **Multi-System Chain:** Applies multiple chaos systems in sequence
    
    ⚠️ **Store your primers safely!** Without the exact primers and chaos settings, 
    your encrypted data cannot be recovered.
    """)


def render_ngs_help():
    """NGS preparation help."""
    st.markdown("### 🧫 NGS Preparation")
    
    st.markdown("""
    The NGS (Next-Generation Sequencing) Preparation mode prepares your DNA sequences 
    for real-world synthesis and sequencing experiments.
    """)
    
    # Fragment structure
    with st.expander("📐 Fragment Structure", expanded=True):
        st.markdown("""
        Each fragment follows the standard NGS oligo structure:
        
        ```
        5' ── Forward Primer ── Index ── Payload ── Reverse Primer ── 3'
              (20 nt)          (8 nt)   (51-101 nt)  (21 nt)
        ```
        
        **Default Primers (Illumina-compatible):**
        - Forward: `ACACGACGCTCTTCCGATCT` (20 nt)
        - Reverse: `AGATCGGAAGAGCACACGTCT` (21 nt)
        
        **Total Length:** 100-150 nucleotides (optimal for NGS)
        """)
    
    # Index generation
    with st.expander("🏷️ Strand Indices"):
        st.markdown("""
        Each fragment has a unique 8-nucleotide index for identification:
        
        **Random Index Generation**
        - Unique random 8-mers for each fragment
        - Designed to avoid long homopolymers (no 3+ consecutive same bases)
        - Reproducible based on fragment number
        - Better for error detection and fragment identification
        
        Example indices: `ATGCTAGT`, `GCATACGT`, `TGCAGTAC`, ...
        """)
    
    # Scoring system
    with st.expander("🎯 Scoring System"):
        st.markdown("""
        Each fragment receives a synthesis/sequencing compatibility score (0-100):
        
        | Factor | Points | Ideal Range |
        |--------|--------|-------------|
        | GC Ratio | 30 | 45-55% |
        | Max Homopolymer | 35 | ≤3 nt |
        | Shannon Entropy | 20 | >1.95 |
        | Length | 15 | 100-150 nt |
        
        **Difficulty Ratings:**
        - **Easy (70-100):** Straightforward synthesis
        - **Medium (50-69):** May need optimization
        - **Hard (<50):** Consider redesigning
        """)
    
    # Synthesis tips
    with st.expander("💡 Synthesis Tips"):
        st.markdown("""
        **For Better Synthesis Success:**
        
        1. **Avoid Long Homopolymers**
           - Use randomization to reduce consecutive repeats
           - Homopolymers >6 nt often cause synthesis errors
        
        2. **Balance GC Content**
           - Aim for 40-60% GC
           - Extreme GC causes secondary structures
        
        3. **Use Appropriate Length**
           - 100-150 nt is ideal for most NGS platforms
           - Longer oligos are more expensive and error-prone
        
        4. **Order Easy Fragments First**
           - Test with high-score fragments before ordering all
        """)
    
    # Export formats
    with st.expander("📥 Export Formats"):
        st.markdown("""
        **CSV Format:**
        - Complete data with all characteristics
        - Good for analysis and record-keeping
        
        **FASTA Format:**
        - Standard bioinformatics format
        - Compatible with most analysis tools
        
        **Sequences Only:**
        - Just the sequences, one per line
        - Direct input for synthesis ordering
        """)


def render_metrics_help():
    """Quality metrics help."""
    st.markdown("### 📊 Quality Metrics")
    
    st.markdown("Different file types use different metrics to measure reconstruction quality:")
    
    # General metrics
    with st.expander("📏 General Metrics", expanded=True):
        st.markdown("""
        **Byte Accuracy (%):** Percentage of bytes that match exactly
        - 100% = Perfect reconstruction
        
        **Bit Error Rate (BER):** Proportion of bits that differ
        - 0 = Perfect
        - Lower is better
        """)
    
    # Image metrics
    with st.expander("🖼️ Image Metrics"):
        st.markdown("""
        **SSIM (Structural Similarity Index)**
        - Range: -1 to 1
        - 1.0 = Identical
        - >0.95 = Excellent
        - >0.90 = Good
        
        **PSNR (Peak Signal-to-Noise Ratio)**
        - Unit: decibels (dB)
        - ∞ = Identical
        - >40 dB = Excellent
        - >30 dB = Good
        
        **MSE (Mean Squared Error)**
        - 0 = Perfect
        - Lower is better
        """)
    
    # Audio metrics
    with st.expander("🎵 Audio Metrics"):
        st.markdown("""
        **SNR (Signal-to-Noise Ratio)**
        - Unit: decibels (dB)
        - ∞ = Identical
        - >60 dB = Excellent (imperceptible)
        - >40 dB = Very Good
        - >20 dB = Good
        
        **Correlation**
        - Range: -1 to 1
        - 1.0 = Perfect correlation
        - >0.99 = Excellent
        """)
    
    # Video metrics
    with st.expander("🎬 Video Metrics"):
        st.markdown("""
        **PSNR (Peak Signal-to-Noise Ratio)**
        - ∞ = Identical
        - >40 dB = Excellent
        - >35 dB = Very Good
        - >30 dB = Good
        
        **SSIM (Structural Similarity)**
        - 1.0 = Identical
        - >0.95 = Very Good
        """)


def render_troubleshooting_help():
    """Troubleshooting help."""
    st.markdown("### ⚠️ Troubleshooting")
    
    # Common issues
    with st.expander("❌ Decoding fails with 'Invalid DNA sequence'", expanded=True):
        st.markdown("""
        **Possible causes:**
        1. The sequence contains invalid characters (only A, T, G, C allowed)
        2. The file was corrupted during transfer
        3. The sequence was truncated
        
        **Solutions:**
        - Verify the DNA file contains only valid nucleotides
        - Re-download the original DNA file
        - Check for any extra whitespace or line breaks
        """)
    
    with st.expander("🔓 Cannot decode randomized sequence"):
        st.markdown("""
        **Possible causes:**
        1. Wrong primer sequences entered
        2. Wrong chaos system selected
        3. Wrong chain order (for multi-system)
        
        **Solutions:**
        - Double-check forward and reverse primer sequences
        - Verify the exact chaos system configuration used
        - Ensure primers are entered in the correct order
        
        ⚠️ **Note:** Without the correct keys, recovery is mathematically impossible.
        """)
    
    with st.expander("📊 Reconstructed file looks different"):
        st.markdown("""
        **For lossy compression (expected):**
        - WebP images may have slight artifacts
        - AAC/MP3 audio may have minor quality differences
        - H.264 video may have compression artifacts
        
        **Unexpected differences:**
        - Check if the correct compression settings were used
        - Verify the full DNA sequence was used (not truncated)
        - Compare using the Comparison tab to see metrics
        """)
    
    with st.expander("⏱️ Encoding/Decoding is very slow"):
        st.markdown("""
        **Possible causes:**
        1. Large file size
        2. Complex compression settings
        3. AV1 video encoding (inherently slow)
        
        **Solutions:**
        - Use faster presets for video encoding
        - Reduce quality settings slightly
        - For AV1, consider using H.264 instead
        - For very large files, expect longer processing times
        """)


def render_faq_help():
    """Frequently asked questions."""
    st.markdown("### ❓ Frequently Asked Questions")
    
    faqs = [
        {
            "q": "Can I actually synthesize this DNA?",
            "a": "Yes! The DNA sequences are valid and can be synthesized. However, synthesis costs vary by length. Use randomization to reduce homopolymer runs for better synthesis compatibility."
        },
        {
            "q": "How long will DNA storage last?",
            "a": "Properly stored DNA can last thousands of years. The information is stable as long as the DNA molecule remains intact. Store in cool, dry conditions."
        },
        {
            "q": "What happens if I lose my primer keys?",
            "a": "If you randomized your DNA and lose the primer keys, the data cannot be recovered. The chaos encryption is mathematically secure. Always backup your primers!"
        },
        {
            "q": "Why is my compressed file larger than the original?",
            "a": "This can happen with already-compressed files (like JPEG or MP3), very small files, or random/encrypted data. The platform automatically uses uncompressed encoding in such cases."
        },
        {
            "q": "What's the maximum file size supported?",
            "a": "There's no hard limit, but very large files will take longer to process. For files over 100MB, consider splitting them or using maximum compression."
        },
        {
            "q": "Can I encode multiple files at once?",
            "a": "Currently, the platform handles one file at a time. For multiple files, consider creating a ZIP archive first, then encoding the archive."
        },
        {
            "q": "Is the DNA sequence encrypted?",
            "a": "By default, no. The basic encoding is reversible by anyone with the DNA sequence. Use the Randomization mode with primer keys for encryption."
        },
        {
            "q": "What's the difference between H.264 and AV1?",
            "a": "H.264 is faster and more compatible. AV1 provides 30-50% better compression but is much slower to encode. Use H.264 for speed, AV1 for maximum compression."
        }
    ]
    
    for faq in faqs:
        with st.expander(f"**Q:** {faq['q']}"):
            st.markdown(f"**A:** {faq['a']}")


if __name__ == "__main__":
    main()
