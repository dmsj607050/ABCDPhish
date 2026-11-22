"""Optional offline Tesseract adapter: append normalized OCR regions to a JSONL."""
import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--tesseract-cmd")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()
    try:
        import pytesseract
    except ImportError as exc:
        raise SystemExit("Install optional dependency: python -m pip install 'aidr-net[ocr]' (also install Tesseract executable)") from exc
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    source, destination = Path(args.manifest).resolve(), Path(args.output).resolve()
    if source == destination:
        raise ValueError("Choose a different output manifest")
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("html_path", "screenshot_path"):
            if row.get(key):
                row[key] = str((source.parent / row[key]).resolve())
        if row.get("screenshot_path"):
            with Image.open(row["screenshot_path"]) as image:
                width, height = image.size
                data = pytesseract.image_to_data(image.convert("RGB"), lang=args.lang, output_type=pytesseract.Output.DICT)
            regions = row.setdefault("visual_regions", [])
            for i, text in enumerate(data["text"]):
                confidence = float(data["conf"][i]) / 100
                if not text.strip() or confidence < args.min_confidence:
                    continue
                x, y, w, h = (int(data[key][i]) for key in ("left", "top", "width", "height"))
                box = [max(0., x/width), max(0., y/height), min(1., (x+w)/width), min(1., (y+h)/height)]
                if box[0] < box[2] and box[1] < box[3]:
                    regions.append({"box": box, "text": text, "confidence": confidence, "source": "tesseract"})
        rows.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} enriched records to {destination}")


if __name__ == "__main__":
    main()
