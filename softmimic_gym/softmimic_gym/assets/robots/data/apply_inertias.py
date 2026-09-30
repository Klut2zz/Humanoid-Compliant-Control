import xml.etree.ElementTree as ET
import re

# Parse the MuJoCo XML model
mjcf_file = "../../../../../softmimic_gym_deploy/softmimic_gym_deploy/src/assets/h1_2_description/h1_2_handless_no_ghost_fixed_base_torsoroot2.xml"        # your MuJoCo file
# usda_file = "h1_2_minimal_simplefoot.usda"       # your input USDA
# output_file = "h1_2_minimal_simplefoot_inertias.usda" # where the patched USDA will be written
usda_file = "h1_2_minimal_simplefoot_torsoroot.usda"       # your input USDA
output_file = "h1_2_minimal_simplefoot_torsoroot_inertias.usda" # where the patched USDA will be written

# 1) Extract inertial parameters from the MuJoCo file
mj = ET.parse(mjcf_file)
body_params = {}
for body in mj.findall('.//body'):
    name = body.get('name')
    inertial = body.find('inertial')
    if name and inertial is not None:
        mass = float(inertial.get('mass'))
        com = [float(x) for x in inertial.get('pos').split()]
        diag = [float(x) for x in inertial.get('diaginertia').split()]
        body_params[name] = (mass, com, diag)

# 2) Read the USDA, strip old physics lines, and inject new ones
with open(usda_file) as f:
    lines = f.readlines()

output = []
i = 0
while i < len(lines):
    line = lines[i]
    output.append(line)
    # Detect the start of a rigid-body Xform block
    m = re.match(r'^(\s*)def Xform "([^"]+)"', line)
    if m:
        base_indent, prim_name = m.groups()
        # Copy until the opening brace {
        i += 1
        while i < len(lines) and '{' not in lines[i]:
            output.append(lines[i])
            i += 1
        if i >= len(lines):
            break
        # Add the '{' line
        output.append(lines[i])
        i += 1
        # Now skip any existing physics:mass/centerOfMass/diagonalInertia lines
        while i < len(lines) and re.match(r'^\s*(float physics:mass|point3f physics:centerOfMass|float3 physics:diagonalInertia)', lines[i]):
            i += 1
        # If we have parameters for this prim, insert them
        if prim_name in body_params:
            mass, com, diag = body_params[prim_name]
            indent = base_indent + '    '  # one extra tab (4 spaces) beyond def indent
            output.append(f'{indent}float physics:mass             = {mass}\n')
            output.append(f'{indent}point3f physics:centerOfMass   = ({com[0]}, {com[1]}, {com[2]})\n')
            output.append(f'{indent}float3 physics:diagonalInertia = ({diag[0]}, {diag[1]}, {diag[2]})\n')
        continue
    i += 1

# 3) Write the patched USDA
with open(output_file, 'w') as f:
    f.writelines(output)

print(f"Patched USDA written to {output_file}")
