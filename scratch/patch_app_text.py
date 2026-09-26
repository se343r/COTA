import json

def get_text(digits):
    res = ""
    for d in digits:
        lbl = d.get("label", "")
        if not lbl: continue
        
        t = "_" if lbl == "unreadable" else lbl
        if d.get("is_between"): t += "↕"
        if d.get("is_decimal"): t = "." + t
        res += t
    return res

