with open("templates/testocr.html", "r") as f:
    html = f.read()

bad_js = """        res.digits.forEach(d => {
            finalStr += d.char;
            if (d.ocr_conf !== undefined) {"""

good_js = """        res.digits.forEach(d => {
            finalStr += d.char;
            
            if (d.box_orig) {
                ctx.strokeStyle = "#ff4da3";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(d.box_orig[0][0], d.box_orig[0][1]);
                for(let j=1; j<4; j++) ctx.lineTo(d.box_orig[j][0], d.box_orig[j][1]);
                ctx.closePath();
                ctx.stroke();
                
                ctx.fillStyle = "#ff4da3";
                ctx.font = "bold 24px Arial";
                ctx.fillText(d.char, d.box_orig[0][0], d.box_orig[0][1] - 5);
            }

            if (d.ocr_conf !== undefined) {"""

if "d.box_orig" not in html:
    html = html.replace(bad_js, good_js)
    with open("templates/testocr.html", "w") as f:
        f.write(html)
