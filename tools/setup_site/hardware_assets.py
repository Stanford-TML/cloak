from pathlib import Path
import base64

ROOT = Path(__file__).resolve().parents[2] / 'docs/setup'

def diagram(name, image, title, points, caption):
    encoded = base64.b64encode((ROOT/'images'/image).read_bytes()).decode()
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 1200"><title>{title}</title><rect width="1400" height="1200" fill="#faf9f7"/><image href="data:image/jpeg;base64,{encoded}" width="1400" height="1050"/>'
    for i,(x,y) in enumerate(points,1):
        svg += f'<circle cx="{x}" cy="{y}" r="27" fill="none" stroke="#fff" stroke-width="12"/><circle cx="{x}" cy="{y}" r="27" fill="none" stroke="#bd2538" stroke-width="6"/><circle cx="{x+40}" cy="{y-30}" r="20" fill="#bd2538"/><text x="{x+40}" y="{y-23}" text-anchor="middle" font-family="sans-serif" font-size="22" fill="white">{i}</text>'
    svg += f'<text x="35" y="1100" font-family="sans-serif" font-size="30" font-weight="bold" fill="#26232d">{title}</text><text x="35" y="1150" font-family="sans-serif" font-size="23" fill="#554c60">{caption}</text></svg>'
    (ROOT/'images'/name).write_text(svg)

def prepare():
    diagram('robotiq-holes.svg','view7565.jpg','Robotiq: original DROID camera bracket',[(876,568),(768,676)],'1 + 2: camera bracket screws. Match the wrist housing and connector, not the photo rotation.')
    diagram('umi-holes.svg','view7572.jpg','UMI: same DROID bracket, different mounting position',[(561,578),(649,688)],'1 + 2: camera bracket screws. Camera USB end and wrist connector identify the orientation.')
    diagram('sharpa-holes.svg','view7581.jpg','Sharpa: dedicated bracket, Robotiq mounting holes',[(699,323),(655,423)],'1 + 2: camera bracket screws. Robot wrist is rotated relative to the Robotiq photo.')
    diagram('sharpa-adapter.svg','view7583.jpg','Sharpa: angled adapter and right-hand orientation',[(504,482),(630,560),(856,447)],'1: flange face. 2: hand interface. 3: camera lenses face toward the hand workspace.')

def downloads(slug):
    rows = [('droid_camera_mount.stl','Camera mount - part 2','1')] if slug in ('robotiq','umi') else [('cloak_sharpa_mount.stl','Camera mount - part 2','1'),('sharpa_angled_adapter_mm.stl','Hand adapter','1')]
    rows = [('droid_camera_closing_part.stl','Camera mount - part 1','1')] + rows
    html='<h3>STL downloads</h3><p>These files are included in the repository under <code>docs/setup/stl/</code>. Import them as millimeters, at 100% scale.</p><table><tr><th>Part</th><th>Preview</th><th>Quantity</th></tr>'
    html+=''.join(f'<tr><td><a download href="stl/{f}">{label}</a></td><td><img width="150" height="110" src="images/{Path(f).stem}.svg" alt="Mesh preview of {label}"></td><td>{qty}</td></tr>' for f,label,qty in rows)+'</table>'
    html+='<p>Part 1 closes the camera holder; part 2 attaches it to the robot. Print one of each. Check the slicer preview before printing.</p><p><a href="stl/README.md">Part sources and dimensions</a></p>'
    html+='<h3>Fasteners</h3><p>Camera holder: <strong>one M3 × 10 mm screw and an M3 nut</strong>. Camera bracket to wrist: <strong>two M3 × 30 mm screws</strong>. Reuse these fasteners from your existing DROID camera installation. The <a href="https://droid-dataset.github.io/droid/hardware-setup/assembly.html#mounting-hand-camera-on-robot">DROID assembly guide</a> specifies these lengths. The M3 size is inferred from the 3.0 mm screw holes in the DROID mount STL files and the confirmed use of the same thread size for both connections.</p>'
    if slug=='sharpa':html+='<p>Sharpa hand to angled adapter: <strong>four M4 × 16 mm screws</strong>. Adapter to Franka flange: <strong>four M6 × 10 mm screws</strong> (the DROID Robotiq coupling-to-flange fasteners). See the <a href="https://droid-dataset.github.io/droid/hardware-setup/assembly.html#mount-procedure">DROID mounting instructions</a> for the flange fasteners.</p>'
    elif slug=='umi':html+='<p>UMI fingertips: <strong>two M6 × 10 mm screws</strong> (the DROID Robotiq coupling-to-flange fasteners, using two rather than four). See the <a href="https://droid-dataset.github.io/droid/hardware-setup/assembly.html#mount-procedure">DROID mounting instructions</a> for the original fasteners.</p>'
    else:html+='<p>Coupling to Franka flange: <strong>four M6 × 10 mm screws</strong>. Robotiq gripper to coupling: <strong>four M5 × 30 mm screws</strong>. Follow the <a href="https://droid-dataset.github.io/droid/hardware-setup/assembly.html#mount-procedure">DROID Robotiq mounting instructions</a>.</p>'
    if slug=='umi':html+='<p>UMI fingertip print files will be added when available.</p>'
    html+=f'<h3>Mounting holes and orientation</h3><figure><a href="images/{slug}-holes.svg"><img src="images/{slug}-holes.svg" alt="{slug} camera bracket mounting screws marked 1 and 2"></a><figcaption>The circles mark the two camera-mount screws. Use the wrist housing and connector to match the orientation shown.</figcaption></figure>'
    if slug=='sharpa':html+='<figure><img src="images/sharpa-adapter.svg" alt="Sharpa angled adapter and camera orientation"><figcaption>Installed right hand: the camera sits above the hand in this side view. Match the adapter slope and hand orientation shown. End-effector bolt holes hidden by the installed adapter are not numbered.</figcaption></figure>'
    elif slug=='robotiq':html+='<p>Keep the standard DROID coupling orientation shown below. The camera bracket and Robotiq coupling have separate mounting fasteners.</p>'
    else:html+='<p>Fit the native Franka Hand with the opposing UMI fingertips as shown below. The camera bracket and native hand have separate mounting fasteners.</p>'
    return html
