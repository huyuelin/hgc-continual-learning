#!/usr/bin/env python3
"""
Generate paper overview figure using yyds168 API (Gemini image model).
Runs entirely on remote server to avoid sshfs+network conflicts.
"""

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from io import BytesIO

try:
    import httpx
except ImportError:
    print("[INFO] Installing httpx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

try:
    from PIL import Image
except ImportError:
    print("[INFO] Installing Pillow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
    from PIL import Image

# ===== Configuration =====
API_BASE = "https://api.yyds168.net/v1"
API_KEY = "sk-QYEIPnvyhVy6lSkWMuoKd7BdADtW78VQ9HkJQfNKo9HX2Soy"

# Best available: gemini-3.0-pro-image-landscape-2k (Pro, landscape, 2K)
IMAGE_MODEL = "gemini-3.0-pro-image-landscape-2k"

OUTPUT_DIR = Path("/home/agent_user/workspace/yuelin/continuous_learning/eassy/hgc_paper/figures_banana")

# ===== The optimized prompt =====
FIGURE_PROMPT = r"""Create a publication-quality main overview figure for a NeurIPS paper.
This is Figure 1: Problem + Method Overview.
Do NOT place "Figure 1." or the full paper title inside the image.
Only keep compact panel titles.

Style:
clean flat vector scientific illustration, white background, minimalist academic aesthetic, crisp readable text, no 3D, no glossy effects, no poster style, no slide-deck feel.
Wide landscape layout, about 2:1 ratio, suitable for a two-column conference paper.

Main design goal:
A reviewer should understand the figure in 5-10 seconds.
The figure must communicate one dominant idea:
vanilla multi-frequency HOPE does NOT automatically preserve optimizer-state persistence,
while LAOSP restores the hierarchy by applying stronger optimizer-state protection to slower levels.

================================
GLOBAL LAYOUT
================================
Use a clean left-to-right structure with 3 zones:

LEFT: Vanilla HOPE (Problem)
CENTER: one short conceptual sentence only
RIGHT: LAOSP (Solution)

Add one thin bottom takeaway strip across the full width.

Important:
- The center zone must be narrow.
- No "CENTER DIVIDER" text.
- No large title at the top.
- The figure should feel compressed, sharp, and publication-ready.

Visual hierarchy:
1. On the RIGHT, the OGP shields are the strongest visual signal.
2. The four CMS levels are second.
3. The momentum buffers are third.
4. Formula, CAM, and CLGD are weak annotations only, not visual centers.

================================
LEFT PANEL - VANILLA HOPE (PROBLEM)
================================
Panel title:
Vanilla HOPE (Problem)

Draw a vertical stack of four CMS levels as rounded blue rectangles:
Fast level (C = 1)
Mid level (C = 4)
Slow level (C = 32)
Ultra-slow level (C = 128)

Use progressively darker blue from top to bottom to indicate intended persistence.

To the right of each level, place a small momentum buffer block in teal/cyan.

Show the problem clearly:
- red new-task gradient arrows go directly into all four momentum buffers
- inside each buffer, blue old-task memory is reduced to the SAME visible proportion
- make this "same decay" visually obvious without requiring the viewer to read text
- use identical fade bars / identical blue memory sizes across all four buffers
- the message should be: all four buffers lose old memory similarly

Keep the arrows tidy and simple, not tangled.

Add only one compact annotation near the buffers:
"same decay in all buffers: beta^T"

Add one small gray callout near the lower-left side:
"Parameter-level protection acts above this point, but forgetting happens inside the momentum buffer."
This callout should be subtle, not dominant.

Bottom caption under the left panel:
"Same decay at all levels"

Optional tiny sub-caption in smaller gray text:
"frequency hierarchy is not preserved in optimizer state"

================================
CENTER ZONE
================================
No box title, no arrow label, no extra divider words.

Only place one short, visually memorable sentence in the middle, slightly larger than normal annotations:

Parameter update frequency != optimizer-state persistence

Optionally add a very thin vertical gray divider line behind it, but keep it subtle.

================================
RIGHT PANEL - LAOSP (SOLUTION)
================================
Panel title:
LAOSP (Solution)

Repeat the same four CMS levels and momentum buffers for direct comparison.

Between each incoming red gradient arrow and each momentum buffer, place an OGP protection shield.

The shield size must scale strongly with level frequency:
- Fast level (C = 1): almost invisible shield or faint dotted outline, alpha ~ 0
- Mid level (C = 4): small amber shield, alpha ~ 0.57
- Slow level (C = 32): medium orange shield, alpha ~ 0.83
- Ultra-slow level (C = 128): large gold shield, alpha ~ 0.86

This shield scaling is the single most important visual element in the whole figure.

The red gradient arrows should be increasingly deflected by the shields:
- almost no deflection at fast level
- moderate deflection at mid level
- strong deflection at slow level
- very strong deflection at ultra-slow level

Inside the right-side momentum buffers:
- preserve the blue old-task memory clearly
- especially make slow and ultra-slow buffers look visibly more preserved than on the left panel

Add one tiny annotation near the right-middle area:
"project before entering momentum buffer"

If a formula is shown, it must be very small, light gray, and visually secondary:
g_tilde = g - alpha * U * Sigma_hat * U^T * g

Do NOT let the formula become a second visual center.

================================
OPTIONAL EXTENSIONS - VERY SUBTLE
================================
CAM and CLGD should NOT be prominent.
They are optional tiny accents only.

If included:
- CAM = a very small, pale green dashed self-loop near the slow-level momentum buffer
- CLGD = a very thin, pale purple dashed arrow near the lower-right edge, away from the shield center

Make both 30-50% smaller and lighter than before.
They should feel almost ignorable at first glance.
Do not place them near the main shield visual center.

It is acceptable to omit CAM and CLGD entirely from this figure.

Bottom caption under the right panel:
"Stronger protection for slower levels"

Optional tiny sub-caption in smaller gray text:
"frequency hierarchy is restored"

================================
BOTTOM STRIP
================================
Create a thin, elegant summary strip.

Keep it minimal and polished.
It should only do ONE job: restate the takeaway.

Left side of strip:
small icon pair:
parameter-level intervention = red X
optimizer-state intervention = green check

Right side of strip:
short text only:
"alpha^(l) increases with C^(l)"

Optionally add four tiny alpha boxes:
0, 0.57, 0.83, 0.86
aligned with C = 1, 4, 32, 128

Do not add extra explanatory sentences.
Keep the strip compressed and academic.

================================
COLOR PALETTE
================================
Background: white
CMS levels: light-to-dark blue
Momentum buffers: teal / cyan
Old-task memory: blue
New-task gradients: red
OGP shields: amber / orange / gold
CAM: pale green dashed
CLGD: pale purple dashed
Primary text: dark gray or black
Secondary annotations: medium gray
Formula / optional details: light gray

================================
NEGATIVE CONSTRAINTS
================================
Do NOT use a dark background.
Do NOT add a large figure title at the top.
Do NOT include "Figure 1." inside the image.
Do NOT use "CENTER DIVIDER" text.
Do NOT make CAM or CLGD visually important.
Do NOT make the formula large or central.
Do NOT make the figure look crowded.
Do NOT add biological or neuroscience imagery.
Do NOT create tangled arrows.
Do NOT turn the image into a generic ML pipeline.
Do NOT overload with text.

================================
FINAL PRIORITY
================================
At first glance, the reviewer must instantly see:
LEFT = all buffers lose old memory similarly
CENTER = frequency != optimizer-state persistence
RIGHT = larger shields for slower levels

The figure should feel like a compressed, high-confidence, top-tier conference overview figure:
less explanation, more immediate structure."""


async def generate_image(prompt: str, model: str = IMAGE_MODEL, num_candidates: int = 3):
    """Generate images using yyds168 OpenAI-compatible API with Gemini image model."""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    results = []

    for i in range(num_candidates):
        print(f"\n[INFO] Generating candidate {i+1}/{num_candidates} with model: {model}", flush=True)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert scientific diagram illustrator. Generate high-quality scientific diagrams based on user requests. Note that do not include figure titles in the image."
                },
                {
                    "role": "user",
                    "content": f"Render an image based on the following detailed description:\n\n{prompt}\n\nDiagram:"
                }
            ],
            "temperature": 1.0,
            "max_tokens": 8192,
        }

        for attempt in range(5):
            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    resp = await client.post(
                        f"{API_BASE}/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                if resp.status_code != 200:
                    print(f"  [WARN] HTTP {resp.status_code}: {resp.text[:300]}", flush=True)
                    if attempt < 4:
                        delay = min(10 * (2 ** attempt), 60)
                        print(f"  Retrying in {delay}s...", flush=True)
                        await asyncio.sleep(delay)
                        continue
                    break

                data = resp.json()
                choices = data.get("choices", [])

                if not choices:
                    print(f"  [WARN] No choices returned", flush=True)
                    if attempt < 4:
                        await asyncio.sleep(10)
                        continue
                    break

                message = choices[0].get("message", {})
                content = message.get("content", "")

                # Try to extract base64 image from various response formats
                b64_data = None

                # Format 1: inline_data in content list
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if "inline_data" in part:
                                b64_data = part["inline_data"].get("data", "")
                                break
                            elif part.get("type") == "image_url":
                                url = part.get("image_url", {}).get("url", "")
                                if "base64," in url:
                                    b64_data = url.split("base64,", 1)[1]
                                    break

                # Format 2: data URL in string content
                if not b64_data and isinstance(content, str):
                    if "base64," in content:
                        b64_data = content.split("base64,", 1)[1].split('"')[0].split("'")[0].strip()
                    elif content.startswith("/9j/") or content.startswith("iVBOR"):
                        b64_data = content.strip()

                # Format 3: images field
                if not b64_data:
                    images = message.get("images", [])
                    if images:
                        img = images[0]
                        if isinstance(img, dict):
                            url = img.get("image_url", {}).get("url", "")
                        else:
                            url = str(img)
                        if "base64," in url:
                            b64_data = url.split("base64,", 1)[1]

                if b64_data:
                    results.append(b64_data)
                    print(f"  [OK] Candidate {i+1} generated successfully ({len(b64_data)} chars)", flush=True)
                    break
                else:
                    if isinstance(content, str) and len(content) > 100:
                        print(f"  [INFO] Model returned text ({len(content)} chars), not image.", flush=True)
                        print(f"  First 200 chars: {content[:200]}", flush=True)
                    else:
                        print(f"  [WARN] Could not extract image from response", flush=True)

                    if attempt < 4:
                        await asyncio.sleep(10)
                        continue
                    break

            except Exception as e:
                print(f"  [ERROR] Attempt {attempt+1}: {e}", flush=True)
                if attempt < 4:
                    await asyncio.sleep(10)
                    continue

    return results


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 60, flush=True)
    print(f"PaperBanana Figure Generator (Remote Execution)", flush=True)
    print(f"Model: {IMAGE_MODEL}", flush=True)
    print(f"API: {API_BASE}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print("=" * 60, flush=True)

    num_candidates = 3
    results = await generate_image(FIGURE_PROMPT, num_candidates=num_candidates)

    if not results:
        print("\n[ERROR] No images were generated. Check API key and model availability.", flush=True)
        sys.exit(1)

    saved_files = []
    for idx, b64_data in enumerate(results):
        try:
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(BytesIO(img_bytes))

            # Save as PNG
            filename = f"laosp_overview_{timestamp}_candidate_{idx+1}.png"
            filepath = OUTPUT_DIR / filename
            img.save(str(filepath), "PNG", optimize=True)
            saved_files.append(filepath)
            print(f"\n[SAVED] {filepath} ({img.size[0]}x{img.size[1]})", flush=True)

            # Also save as JPEG
            jpg_filename = f"laosp_overview_{timestamp}_candidate_{idx+1}.jpg"
            jpg_filepath = OUTPUT_DIR / jpg_filename
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            img.save(str(jpg_filepath), "JPEG", quality=95)
            print(f"[SAVED] {jpg_filepath}", flush=True)

        except Exception as e:
            print(f"\n[ERROR] Failed to save candidate {idx+1}: {e}", flush=True)
            raw_path = OUTPUT_DIR / f"laosp_overview_{timestamp}_candidate_{idx+1}.raw.bin"
            with open(raw_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            print(f"[SAVED] Raw data: {raw_path}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Done! {len(saved_files)} figures saved to {OUTPUT_DIR}/", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
