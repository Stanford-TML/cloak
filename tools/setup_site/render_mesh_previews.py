from pathlib import Path
import struct, math
root=Path(__file__).resolve().parents[2]
# Render the actual STL triangles with an orthographic projection.
for f in (root/'docs/setup/stl').glob('*.stl'):
 if f.stem.endswith('_model'): continue
 data=f.read_bytes(); count=struct.unpack_from('<I',data,80)[0]; faces=[]; points=[]
 for i in range(count):
  vals=struct.unpack_from('<12fH',data,84+50*i)
  tri=[]
  for k in (3,6,9):
   x,y,z=vals[k:k+3]; p=((x-y)*.7071,(x+y)*.4082-z*.8165,(x+y+z)*.5774);tri.append(p);points.append(p)
  faces.append(tri)
 lo=[min(p[j] for p in points) for j in (0,1)]; hi=[max(p[j] for p in points) for j in (0,1)]
 scale=min(270/(hi[0]-lo[0]),190/(hi[1]-lo[1])); cx=(lo[0]+hi[0])/2; cy=(lo[1]+hi[1])/2
 svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 220"><rect width="300" height="220" fill="#f3f1ee"/>']
 for tri in sorted(faces,key=lambda t:sum(p[2] for p in t)):
  a,b,c=tri;u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)];n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];norm=math.sqrt(sum(x*x for x in n)) or 1;light=abs((n[0]*.3-n[1]*.4+n[2]*.866)/norm);shade=int(85+100*light)
  coords=' '.join(f'{150+(p[0]-cx)*scale:.2f},{110+(p[1]-cy)*scale:.2f}' for p in tri)
  svg.append(f'<polygon points="{coords}" fill="rgb({shade},{shade},{shade+10})"/>')
 (root/'docs/setup/images'/f'{f.stem}.svg').write_text(''.join(svg)+'</svg>')
