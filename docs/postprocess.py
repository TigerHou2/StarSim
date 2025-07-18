import os

# Get the directory this script is in
script_dir = os.path.dirname(os.path.abspath(__file__))

target_line = '<div class="cm-editor cm-s-jupyter">'
replacement_line = '<div class="cm-editor cm-s-jupyter" style="overflow-x: scroll;">'

for root, _, files in os.walk(os.path.join(script_dir, "assets")):
    for file in files:
        if file.endswith(".html"):
            file_path = os.path.join(root, file)
            
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            for line in lines:
                if line.strip() == target_line:
                    new_lines.append(line.replace(target_line, replacement_line))
                    modified = True
                else:
                    new_lines.append(line)
            
            if modified:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
