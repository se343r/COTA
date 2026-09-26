import json
with open('/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/.system_generated/logs/transcript_full.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        content = data.get('content', '')
        if '<!-- testocr.html -->' in content or '<div class="digit-card" id="card-${i}">' in content:
            idx = content.find('<div class="digit-card" id="card-${i}">')
            if idx != -1:
                print(content[idx:idx+1000])
                break
