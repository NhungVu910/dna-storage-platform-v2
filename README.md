# 🧬 DNA Data Storage Platform v2.0

A comprehensive, user-friendly platform for encoding, encrypting, preparing, and decoding digital data using synthetic DNA sequences. Built with Streamlit for accessible deployment and research collaboration.

## ✨ Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Multi-format Encoding** | Convert text, images, audio, video, PDF to DNA | ✅ |
| **Chaos Encryption** | Secure with Logistic/Hénon/Lorenz chaos systems | ✅ |
| **NGS Fragment Generation** | Prepare sequences for DNA synthesis | ✅ |
| **Quality Verification** | Compare original and reconstructed data | ✅ |
| **Cloud Deployment** | One-click deployment to Streamlit Cloud | ✅ |

## ⚙️ Installation

### **Local Setup**
```bash
# 1. Clone repository
git clone https://github.com/username/dna-storage-platform.git
cd dna-storage-platform

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
streamlit run app.py
```

## File Structure
```
dna_storage/
├── app.py              # Main Streamlit application
├── dna_codec.py        # DNA encoding/decoding logic
├── compression.py      # Compression algorithms
├── randomization.py    # Henon chaos map implementation
├── comparison.py       # Quality metrics calculation
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage Examples

### Complete Workflow
1. **Encode**: Upload file → Select compression → Encode to DNA
2. **Randomize**: Load DNA → Select chaos system → Enter primers → Randomize
3. **NGS Prep**: Load DNA → Set fragment length → Generate fragments
4. **Decode**: Load randomized DNA → Enter primers → Decode to file
5. **Compare**: Load original and decoded → Verify integrity

## License
MIT License

## Author
DNA Data Storage Platform v2.0
