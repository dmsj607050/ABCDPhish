"""Generate synthetic local fixtures. These are wiring examples, not a benchmark."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(8):
        label = i % 2
        brand = "Example Bank" if label else "Example Store"
        # Reserved .test names; no visits or network requests.
        url = f"https://sample-{i}.test/login"
        target = f"https://collector-{i}.test/submit" if label else "/session"
        html = f'<html><head><title>{brand}</title></head><body><h1>{brand}</h1><form action="{target}"><label>Account</label><input name="account"><input type="password" name="password"><button>Sign in</button></form></body></html>'
        (root / f"page-{i}.html").write_text(html, encoding="utf-8")
        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), brand, fill="black")
        draw.rectangle((20, 65, 205, 95), outline="gray")
        draw.rectangle((20, 110, 205, 140), outline="gray")
        draw.rectangle((20, 160, 205, 195), fill="#2457c5")
        draw.text((80, 170), "Sign in", fill="white")
        image.save(root / f"page-{i}.png")
        records.append({"id": f"demo-{i}", "url": url, "html_path": f"page-{i}.html", "screenshot_path": f"page-{i}.png", "label": label,
                        "visual_regions": [{"box": [0.08, 0.05, 0.95, 0.22], "role": "appearance", "confidence": 0.9},
                                           {"box": [0.08, 0.28, 0.95, 0.9], "role": "intent", "confidence": 0.9}]})
    for name, items in (("train", records[:6]), ("val", records[6:]), ("predict", [{k: v for k, v in r.items() if k != "label"} for r in records[6:]])):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items), encoding="utf-8")
    print(f"Wrote synthetic fixtures to {root.resolve()}")


if __name__ == "__main__":
    main()
