import base64
with open('/home/deist/Downloads/OCR/anh/parseq_mech_data/val/images/2aoboqvxz29j4spv6cz0mr3xuwcmdubqqmnkapsw182.jpg', 'rb') as f:
    print(base64.b64encode(f.read()).decode()[:50])
