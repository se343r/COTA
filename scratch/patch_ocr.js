function getNextOcrIdx(startIdx) {
    for (let i = startIdx; i < ocrImages.length; i++) {
        const item = ocrImages[i];
        if (ocrFilter === "minority" && !item.minority) continue;
        if (ocrFilter === "dientu" && item.meter_class !== "dientu") continue;
        if (ocrFilter === "co" && item.meter_class !== "co") continue;
        return i;
    }
    return -1;
}

function getPrevOcrIdx(startIdx) {
    for (let i = startIdx; i >= 0; i--) {
        const item = ocrImages[i];
        if (ocrFilter === "minority" && !item.minority) continue;
        if (ocrFilter === "dientu" && item.meter_class !== "dientu") continue;
        if (ocrFilter === "co" && item.meter_class !== "co") continue;
        return i;
    }
    return -1;
}
