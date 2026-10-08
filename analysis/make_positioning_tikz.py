"""Write the positioning map (Figure 1) as TikZ, so that it uses the fonts of the manuscript."""
import re
src = open("make_fig_positioning.py").read()
W = eval(src[src.index("W = ["):src.index("]\n", src.index("W = ["))+1].split("=", 1)[1])
COL = {"ML for MILP": "blue!70!black", "OR reduction": "orange!85!black", "Unit commitment": "green!50!black",
       "Conformal / distribution-free": "violet!80!black"}
CW = [7.4, 3.4, 3.9, 5.6]; CH = [3.25, 1.3, 0.9, 2.05, 1.95]
XS = [sum(CW[:i]) for i in range(5)]; YS = [sum(CH[:j]) for j in range(6)]
XL = ["Hand-set", "Tuned or\\\\searched", "Learned or\\\\structural", "Calibrated\\\\(conformal)"]
YL = ["None", "Indirect or\\\\partial bound", "Via a parallel\\\\full solve", "Finite-sample, on\\\\another quantity",
      "Finite-sample, on\\\\the optimality gap"]
o = ["\\begin{tikzpicture}[font=\\scriptsize, x=1cm, y=1cm]"]
for i in range(4):
    for j in range(5):
        shade = "gray!6" if (i + j) % 2 == 0 else "gray!2"
        o.append(f"\\fill[{shade}] ({XS[i]},{YS[j]}) rectangle ({XS[i+1]},{YS[j+1]});")
o.append(f"\\fill[orange!12] ({XS[3]},{YS[4]}) rectangle ({XS[4]},{YS[5]});")
for x in XS: o.append(f"\\draw[gray!40] ({x},0) -- ({x},{YS[5]});")
for y in YS: o.append(f"\\draw[gray!40] (0,{y}) -- ({XS[4]},{y});")
o.append(f"\\draw[orange!80!black, line width=1.2pt] ({XS[3]},{YS[4]}) rectangle ({XS[4]},{YS[5]});")
cells = {}
for lab, st, x, y in W: cells.setdefault((x, y), []).append((lab, st))
LINE = 0.36
for (x, y), items in cells.items():
    n = len(items); two = n > 8; per = 8 if two else n
    for k, (lab, st) in enumerate(items):
        c, r = (k // per, k % per) if two else (0, k)
        x0 = XS[x] + 0.12 + c * CW[x] / 2; y0 = YS[y + 1] - 0.25 - r * LINE
        lab = lab.replace("&", "\\&")
        o.append(f"\\fill[{COL[st]}] ({x0:.2f},{y0-0.07:.2f}) rectangle ++(0.12,0.14);")
        o.append(f"\\node[anchor=west, inner sep=0pt] at ({x0+0.2:.2f},{y0:.2f}) {{{lab}}};")
cx = (XS[3] + XS[4]) / 2
o.append(f"\\node[draw=orange!80!black, fill=orange!80!black, text=white, rounded corners, align=center, font=\\scriptsize\\bfseries, text width={CW[3]-0.4}cm] at ({cx},{YS[4]+0.45}) {{CRISP (this paper):\\\\ calibrated level of variable fixing}};")
for i in range(4): o.append(f"\\node[align=center, anchor=north, font=\\footnotesize] at ({(XS[i]+XS[i+1])/2},-0.12) {{{XL[i]}}};")
for j in range(5): o.append(f"\\node[align=right, anchor=east, font=\\footnotesize] at (-0.12,{(YS[j]+YS[j+1])/2}) {{{YL[j]}}};")
o.append(f"\\node[font=\\small] at ({XS[4]/2},-1.3) {{How the reduction level (or the calibrated parameter) is chosen}};")
o.append(f"\\node[font=\\small, rotate=90] at (-3.2,{YS[5]/2}) {{Guarantee on the returned solution}};")
lx = 3.5
for k, (st, c) in enumerate(COL.items()):
    o.append(f"\\fill[{c}] ({lx},{YS[5]+0.35}) rectangle ++(0.2,0.2); \\node[anchor=west, font=\\footnotesize] at ({lx+0.28},{YS[5]+0.45}) {{{st}}};")
    lx += [3.0, 3.0, 3.4, 4.0][k]
o.append("\\end{tikzpicture}")
open("figures/fig_positioning_tikz.tex", "w").write("\n".join(o) + "\n")
print("ok", len(W))
