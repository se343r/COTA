import json
with open('/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/.system_generated/logs/transcript_full.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        content = data.get('content', '')
        if 'def api_testocr_infer' in content and 'M = cv2.getPerspectiveTransform' in content:
            idx = content.find('def api_testocr_infer')
            print(content[idx:idx+4000])
            break
