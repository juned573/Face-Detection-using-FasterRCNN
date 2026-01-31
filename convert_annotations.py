import os
import json

def convert_txt_to_json(txt_path, output_json):
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    data = []
    i = 0
    while i < len(lines):
        filename = lines[i]
        i += 1
        if i >= len(lines):
            print(f" Missing face count after filename: {filename}")
            break

        try:
            face_count = int(lines[i])
        except ValueError:
            print(f"Invalid face count on line {i + 1}: {lines[i]}")
            continue
        i += 1

        boxes = []
        for _ in range(face_count):
            if i >= len(lines):
                print(f" Unexpected end of file after face count at: {filename}")
                break

            try:
                parts = list(map(float, lines[i].split()[:4]))
                x, y, w, h = parts
                boxes.append([x, y, x + w, y + h])
            except Exception as e:
                print(f" Bad box data at line {i + 1}: {lines[i]} | Error: {e}")
            i += 1

        data.append({"filename": filename, "boxes": boxes})

    with open(output_json, 'w') as out:
        json.dump(data, out, indent=2)

    print(f"Done! Saved {len(data)} entries to {output_json}")

convert_txt_to_json(
    'C:/Users/jeqba/wider_face_split/wider_face_train_bbx_gt.txt',
    'train_annotations.json'
)
