#!/usr/bin/env python3
"""Rebuild vector figures using only Python stdlib and pdfLaTeX (TikZ/PGFPlots).
Summary values are transcribed from manuscript Tables 1-3, Methods, and Results.
Fig. overlap (a) and the reader Bland-Altman figure are drawn from per-image CSVs in data/
(BC-VMamba aligned lambda=0.2, seed 42; reader values from R1_observer_matched.py).
Style follows the SW-YOLO reference roadmap: Times-like serif text, coloured rounded
"Step" title pills, light-tinted rounded boxes, and thick block arrows.
Run from any directory: python3 build_figures.py. Original PDF must remain at root.
"""
from pathlib import Path
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]/'figures'
SRC=Path(__file__).resolve().parent
DATA=Path(__file__).resolve().parents[1]/'data'
PRE=r'''\documentclass[tikz,border=3pt]{standalone}
\usepackage{newtxtext,newtxmath}
\usepackage{graphicx,pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,positioning,calc,patterns,decorations.pathreplacing,shapes.misc}
\definecolor{swblue}{HTML}{3D72B4}
\definecolor{swgreen}{HTML}{2E9A4E}
\definecolor{sworange}{HTML}{E8822A}
\definecolor{swpurple}{HTML}{7E62A8}
\definecolor{swnavy}{HTML}{2C5F8C}
\definecolor{swgray}{HTML}{7A7A7A}
% Okabe-Ito colourblind-safe palette for per-source scatter points
\definecolor{oiblue}{HTML}{0072B2}\definecolor{oiverm}{HTML}{D55E00}\definecolor{oigreen}{HTML}{009E73}
\definecolor{oipink}{HTML}{CC79A7}\definecolor{oiorange}{HTML}{E69F00}\definecolor{oisky}{HTML}{56B4E9}
\tikzset{
 stepbox/.style={draw=#1,line width=1pt,rounded corners=6pt,fill=#1!5},
 pill/.style={fill=#1,draw=none,rounded corners=6pt,text=white,align=center,
  inner xsep=6pt,inner ysep=2.5pt,font=\fontsize{8.5}{10}\selectfont},
 flow/.style={-{Triangle[length=2.6mm,width=3.2mm]},line width=2.2pt,draw=swnavy},
 thin flow/.style={-{Stealth[length=2mm]},line width=0.9pt,draw=swnavy},
}
\pgfplotsset{every axis/.append style={
 font=\fontsize{8}{10}\selectfont,
 tick label style={font=\fontsize{8}{10}\selectfont},
 label style={font=\fontsize{8}{10}\selectfont},
 title style={font=\fontsize{8.5}{10}\selectfont\bfseries},
 axis x line*=bottom,axis y line*=left,tick align=outside,
 xmajorgrids,grid style={gray!20},
 /pgf/number format/.cd,set thousands separator={{{{,}}}}}}
\pgfplotsset{discard if not/.style 2 args={x filter/.append code={%
 \edef\tempa{\thisrow{#1}}\edef\tempb{#2}\ifx\tempa\tempb\else\def\pgfmathresult{inf}\fi}},
 discard if empty/.style={x filter/.append code={%
 \edef\tempa{\thisrow{#1}}\ifx\tempa\empty\def\pgfmathresult{inf}\fi}},
 filter discard warning=false}
\hyphenpenalty=10000\exhyphenpenalty=10000
\begin{document}
'''
def build(name,body):
 p=SRC/(name+'.tex');p.write_text(PRE+body+'\n\\end{document}\n')
 result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={OUT}',p.name],cwd=SRC,capture_output=True,text=True)
 if result.returncode: raise RuntimeError(result.stdout[-4000:])
 for ext in ['aux','log']: (OUT/(name+'.'+ext)).unlink(missing_ok=True)
 if shutil.which('pdftoppm'):  # PNG preview for visual checks only
  subprocess.run(['pdftoppm','-r','200','-png','-singlefile',str(OUT/(name+'.pdf')),str(OUT/name)],check=True)

# Preserve complete empirical figures as vector PDF crops; no redigitization.
# Overlap (b) and the reader figure are redrawn below from per-image data.
# The source manuscript PDF supplies the vector crops; override its location with CUBS_SOURCE_PDF.
SRCPDF=os.environ.get('CUBS_SOURCE_PDF',str(ROOT / 'latex_source_files_srep_revision.pdf'))
for name,page,trim in [('original_uncertainty',7,'74 372 74 265')]:
 build(name,r'\begin{tikzpicture}\node[inner sep=0]{'+rf'\includegraphics[page={page},trim={trim},clip]'+ '{'+SRCPDF+'}'+r'};\end{tikzpicture}')

# Dice-CIMT scatter (a) from per-image data; ultrasound examples (b) cropped from source p. 5.
ov=list(csv.DictReader(open(DATA/'overlap_scatter.csv')))
assert len(ov)==401
dice=[float(x['mean_dice_5ckpt']) for x in ov];err=[float(x['abs_cimt_err_ensemble_mm']) for x in ov]
assert round(statistics.correlation(dice,err),3)==-0.456
assert max(err)<1.5
q1=statistics.quantiles(dice,n=4,method='inclusive')[0]  # Dice first quartile, computed here
EXP=json.loads((SRC.parent/'analysis'/'exploratory_numbers.json').read_text())
QUAD=EXP['discordance']['counts']
assert QUAD['highDice_fail']==sum(d>=q1 and e>0.1686 for d,e in zip(dice,err))
sources=[('Cyprus','Cyprus, clinical','oiblue','*',''),('Pisa_clin','Pisa, clinical','oiverm','square*',''),
         ('Munich','Munich, technical','oigreen','triangle*',''),('Pisa_tech','Pisa, technical','oipink','diamond*',''),
         ('Porto','Porto, technical','oiorange','pentagon*',''),('Torino','Torino, technical','oisky','triangle*',',rotate=180'),
         ('Toronto','Toronto, technical','black','x',',line width=.6pt')]
assert {x['center'] for x in ov}=={c for c,*_ in sources}
sc=[]
for c,lab,col,mk,extra in sources:
 sc.append(rf'\addplot[only marks,mark={mk},mark size=1.5pt,color={col},mark options={{fill opacity=.55,draw opacity=.9{extra}}},discard if not={{center}}{{{c}}}] table[col sep=comma,x=mean_dice_5ckpt,y=abs_cimt_err_ensemble_mm] {{../data/overlap_scatter.csv}};')
 sc.append(rf'\addlegendentry{{{lab}}}')
build('original_overlap',r'''\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]
\begin{axis}[name=a,width=11.9cm,height=5.2cm,scale only axis,xmin=0.1,xmax=1.0,ymin=0,ymax=1.5,
 xlabel={Mean Dice of five checkpoints},ylabel={Absolute CIMT error (mm)},ymajorgrids,
 xtick={0.1,0.2,...,1.0},ytick={0,0.3,...,1.5},
 xticklabel style={/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=1},
 yticklabel style={/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=1},
 legend style={at={(1.03,1)},anchor=north west,draw=none,fill=none,font=\fontsize{8}{10}\selectfont,row sep=1pt},
 legend cell align=left]
'''+'\n'.join(sc)+rf'''
\fill[sworange,opacity=.07] (axis cs:{q1:.4f},0.1686) rectangle (axis cs:1.0,1.5);
\draw[swgray,dashed,line width=.7pt] (axis cs:{q1:.4f},0)--(axis cs:{q1:.4f},1.5);
\draw[swgray,densely dotted,line width=.7pt] (axis cs:0.1,0.1686)--(axis cs:1.0,0.1686);
\node[anchor=north east,font=\fontsize{{7.5}}{{9}}\selectfont,text=swgray,align=right] at (axis cs:{q1-0.005:.4f},1.05) {{Dice first quartile = {q1:.3f}\\(computed from per-image data)}};
\node[anchor=south west,font=\fontsize{{7.5}}{{9}}\selectfont,text=swgray] at (axis cs:0.1,0.1686) {{CIMT failure threshold 0.1686 mm}};
\node[anchor=north east,font=\fontsize{{7.5}}{{9}}\selectfont,text=sworange!80!black,align=right] at (axis cs:0.995,0.75) {{Dice $\geq$ Q1 and error $>$ 0.1686 mm:\\{QUAD['highDice_fail']} of {QUAD['highDice_fail']+QUAD['lowDice_fail']} CIMT failures}};
\node[anchor=north west,inner sep=2pt] at (axis cs:0.115,1.47) {{Pearson $r=-0.456$ (cluster-bootstrap 95\% CI $-0.627$ to $-0.368$)}};
\end{{axis}}
\node[anchor=south east,font=\fontsize{{9}}{{11}}\selectfont\bfseries] at ([xshift=-0.9cm]a.north west) {{(a)}};
\node[anchor=north,inner sep=0] (b) at ($(a.south west)+(7.4,-1.15)$) {{\includegraphics[width=15.6cm,page=5,trim=79 164 79 328,clip]{{{SRCPDF}}}}};
\node[anchor=north east,font=\fontsize{{9}}{{11}}\selectfont\bfseries] at ([xshift=-0.1cm]b.north west) {{(b)}};
\end{{tikzpicture}}''')

# Reader Bland-Altman from per-image procedure-matched CIMT values.
ba=list(csv.DictReader(open(DATA/'reader_bland_altman.csv')))
def num(v):
 try:
  x=float(v);return None if math.isnan(x) else x
 except ValueError: return None
pairs=[('(a) Model $-$ A1','model_mm','msk_A1',(0.0639,0.2708,401)),(r'(b) A1$^{\prime}$ $-$ A1','msk_A1p','msk_A1',(0.0486,0.3232,401)),
       ('(c) A2 $-$ A1','msk_A2','msk_A1',(0.0660,0.4084,401)),('(d) A3 $-$ A2','msk_A3','msk_A2',(-0.1354,0.4258,118))]
body=r'\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]'+'\n'
for j,(title,u,v,expect) in enumerate(pairs):
 d=[num(x[u])-num(x[v]) for x in ba if num(x[u]) is not None and num(x[v]) is not None]
 bias=statistics.mean(d);half=1.96*statistics.stdev(d)
 assert (round(bias,4),round(half,4),len(d))==expect,(title,bias,half,len(d))
 opts=rf'''at={{({j*4.3}cm,0)}},width=3.5cm,height=4.3cm,scale only axis,xmin=0.2,xmax=1.8,ymin=-0.8,ymax=1.6,
 title={{{title}}},xlabel={{Mean of pair (mm)}},ymajorgrids,xtick={{0.4,0.8,1.2,1.6}},ytick={{-0.8,-0.4,0,0.4,0.8,1.2,1.6}},
 xticklabel style={{/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=1}},
 yticklabel style={{/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=1}}'''
 opts+=',ylabel={Difference (mm)}' if j==0 else ',yticklabels={}'
 body+=r'\begin{axis}['+opts+']\n'
 body+=rf'\addplot[only marks,mark=*,mark size=1.1pt,swblue,mark options={{fill opacity=.4,draw opacity=.7}},discard if empty={u},discard if empty={v}] table[col sep=comma,x expr=(\thisrow{{{u}}}+\thisrow{{{v}}})/2,y expr=\thisrow{{{u}}}-\thisrow{{{v}}}] {{../data/reader_bland_altman.csv}};'+'\n'
 body+=rf'\draw[swgray,line width=.4pt] (axis cs:0.2,0)--(axis cs:1.8,0);'+'\n'
 body+=rf'\draw[swnavy,line width=1.1pt] (axis cs:0.2,{bias:.4f})--(axis cs:1.8,{bias:.4f});'+'\n'
 for y in (bias-half,bias+half):
  body+=rf'\draw[sworange,dashed,line width=1pt] (axis cs:0.2,{y:.4f})--(axis cs:1.8,{y:.4f});'+'\n'
 fmt=lambda z:f'{z:.4f}'.replace('-','$-$')
 body+=rf'\node[anchor=north east,fill=white,fill opacity=.85,text opacity=1,inner sep=1.5pt,align=right,font=\fontsize{{7}}{{8.5}}\selectfont] at (rel axis cs:0.98,0.985) {{$n={len(d)}$; bias {fmt(bias)}\\LoA {fmt(bias-half)} to {fmt(bias+half)}}};'+'\n'
 body+='\\end{axis}\n'
body+=r'\end{tikzpicture}'
build('original_reader',body)

# Study roadmap (full width). Counts from Table 1 and Methods.
build('study_workflow',r'''\begin{tikzpicture}[x=1cm,y=1cm,font=\fontsize{8}{10}\selectfont,
 box/.style={stepbox=#1,minimum width=5.5cm,minimum height=2.45cm,align=center,text width=5.3cm}]
\node[box=swblue] (data) at (0,0) {Clinical: 2,176 images, 1,088 patients\\(Cyprus 1,388; Pisa 788)\\Technical: 500 images from 5 sources\\\textbf{Total: 2,676 calibrated images}};
\node[pill=swblue] at (data.north) {Step 1: Public multicenter CUBS data};
\node[box=swblue] (split) at (6.05,0) {\textbf{Development pool: 2,275 images}\\\textbf{Fixed test set: 401 images}\\326 clinical (163 patients); 75 technical\\238 independent resampling clusters};
\node[pill=swblue] at (split.north) {Step 2: Patient-aware partition};
\node[box=swgreen] (prep) at (12.1,0) {Annotation-derived ROI crop\\Square padding; $256\times256$ resize\\Development-only normalization\\A1 delineation as reference mask};
\node[pill=swgreen] at (prep.north) {Step 3: Common measurement pipeline};
\draw[flow] (data)--(split);\draw[flow] (split)--(prep);
\node[box=sworange] (train) at (0,-3.55) {2,275 images in five common folds\\Eight architectures; region-only objective\\Seed 42; trained from scratch\\Best validation-Dice checkpoint per fold};
\node[pill=sworange] at (train.north) {Step 4: Controlled model development};
\node[box=swpurple] (measure) at (6.05,-3.55) {\textbf{Overlap} (Dice) and \textbf{CIMT error} (mm)\\Rank agreement; paired contrasts\\Failure tails; reader agreement\\Uncertainty-based triage};
\node[pill=swpurple] at (measure.north) {Step 5: Evaluate both endpoints};
\node[box=swgreen] (infer) at (12.1,-3.55) {5,000 cluster-bootstrap replicates\\Bilateral clinical images resampled together\\Each technical image is one cluster\\Exploratory aligned loss: seeds 42 and 1};
\node[pill=swgreen] at (infer.north) {Step 6: Uncertainty and sensitivity};
\draw[flow] (prep.south)--++(0,-.35)-|([yshift=.24cm]train.north);
\draw[flow] (train)--(measure);\draw[flow] (measure)--(infer);
\draw[swgray,dashed,rounded corners=4pt,line width=.7pt] (-2.75,-5.05) rectangle (14.85,-5.6);
\node[font=\fontsize{8}{10}\selectfont\itshape] at (6.05,-5.325) {Fixed-test images were excluded from normalization, training, and model selection. The ROI is supplied; automatic wall localization is not evaluated.};
\end{tikzpicture}''')

# CIMT extraction schematic (full width). Operations from Methods (preprocessing, loss).
# Column masks for the illustrative grid: rows marked 1 are foreground (top row first).
cols=['00011100','00011100','00111100','00011110','00110100','00011100','00001000','00011100','00001110','00001110']
grid=[]
cw=0.19
for c,col in enumerate(cols):
 for r,v in enumerate(col):
  x=-0.95+c*cw;y=-0.85-r*cw
  fill='swpurple!70' if v=='1' else 'white'
  if c==0 and v=='1': fill='sworange!85'
  grid.append(rf'\filldraw[fill={fill},draw=gray!45,line width=.3pt] ({x:.3f},{y:.3f}) rectangle ++({cw},{-cw});')
grid='\n'.join(grid)
wall=r'''\fill[gray!60] (-1.4,-0.88) .. controls (-0.5,-0.78) and (0.5,-0.78) .. (1.4,-0.88) -- (1.4,-1.5) .. controls (0.5,-1.4) and (-0.5,-1.4) .. (-1.4,-1.5) -- cycle;
\fill[sworange!65] (-1.4,-1.5) .. controls (-0.5,-1.4) and (0.5,-1.4) .. (1.4,-1.5) -- (1.4,-1.74) .. controls (0.5,-1.64) and (-0.5,-1.64) .. (-1.4,-1.74) -- cycle;
\fill[gray!30] (-1.4,-1.74) .. controls (-0.5,-1.64) and (0.5,-1.64) .. (1.4,-1.74) -- (1.4,-2.0) .. controls (0.5,-1.9) and (-0.5,-1.9) .. (-1.4,-2.0) -- cycle;'''
build('measurement_pipeline',r'''\begin{tikzpicture}[x=1cm,y=1cm,font=\fontsize{8}{10}\selectfont,
 sbox/.style={stepbox=#1,minimum width=3.2cm,minimum height=4.25cm},
 note/.style={font=\fontsize{7.5}{9}\selectfont,align=center,text width=3.0cm},
 tiny/.style={font=\fontsize{7}{8}\selectfont}]
% boxes and pills
\foreach \cx/\col/\lab [count=\i] in {0/swblue/{Step 1: ROI crop},3.65/swgreen/{Step 2: Square \& resize},7.3/sworange/{Step 3: Binary mask},10.95/swpurple/{Step 4: Column thickness},14.6/swnavy/{Step 5: CIMT (mm)}}{
 \node[sbox=\col,anchor=north] (b\i) at (\cx,0) {};
 \node[pill=\col] at (b\i.north) {\lab};}
\foreach \i/\j in {1/2,2/3,3/4,4/5}{\draw[flow] (b\i.east|-0,-1.45)--(b\j.west|-0,-1.45);}
% Step 1: abstract longitudinal frame with wall layers
\begin{scope}[shift={(0,0)}]
\begin{scope}\clip (-1.4,-0.55) rectangle (1.4,-2.35);
\fill[gray!14] (-1.4,-0.55) rectangle (1.4,-2.35);
'''+wall+r'''
\end{scope}
\draw[gray!60] (-1.4,-0.55) rectangle (1.4,-2.35);
\draw[swblue,densely dashed,line width=.7pt] (-0.95,-1.4) rectangle (0.95,-1.74);
\draw[swblue,line width=1pt] (-1.15,-1.1) rectangle (1.15,-2.04);
\node[tiny,text=white] at (0,-1.08) {lumen};
\node[tiny,anchor=west] at (-1.38,-0.68) {B-mode frame};
\node[note,anchor=north] at (0,-2.5) {Annotated extent (dashed) padded by 100\% vertically and 10\% horizontally (solid)};
\end{scope}
% Step 2: padded square
\begin{scope}[shift={(3.65,0)}]
\fill[pattern=north east lines,pattern color=gray!55] (-0.82,-0.62) rectangle (0.82,-2.26);
\begin{scope}\clip (-0.82,-1.12) rectangle (0.82,-1.76);
\fill[gray!14] (-0.82,-1.12) rectangle (0.82,-1.76);
\fill[gray!60] (-0.82,-1.12) rectangle (0.82,-1.33);
\fill[sworange!65] (-0.82,-1.33) .. controls (-0.3,-1.29) and (0.3,-1.29) .. (0.82,-1.33) -- (0.82,-1.55) .. controls (0.3,-1.51) and (-0.3,-1.51) .. (-0.82,-1.55) -- cycle;
\fill[gray!30] (-0.82,-1.55) .. controls (-0.3,-1.51) and (0.3,-1.51) .. (0.82,-1.55) -- (0.82,-1.76) -- (-0.82,-1.76) -- cycle;
\end{scope}
\draw[swgreen,line width=1pt] (-0.82,-0.62) rectangle (0.82,-2.26);
\node[tiny,fill=white,inner sep=1pt] at (0,-0.86) {pad};
\node[tiny,fill=white,inner sep=1pt] at (0,-2.02) {pad};
\draw[{Stealth[length=1.3mm]}-{Stealth[length=1.3mm]},swgray] (-0.82,-2.4)--(0.82,-2.4) node[midway,below,tiny,inner sep=1pt] {$s\to256$ px};
\node[note,anchor=north] at (0,-2.62) {Crop padded to an $s\times s$ square, resized to $256\times256$};
\end{scope}
% Step 3: two delineation sources, one binarization rule
\begin{scope}[shift={(7.3,0)}]
\node[tiny] at (-0.75,-0.72) {Reader A1};
\node[tiny] at (0.75,-0.72) {Model};
\fill[gray!10] (-1.4,-0.9) rectangle (-0.1,-2.2);
\fill[pattern=north west lines,pattern color=swblue] (-1.4,-1.42) .. controls (-1,-1.36) and (-0.5,-1.36) .. (-0.1,-1.42) -- (-0.1,-1.7) .. controls (-0.5,-1.64) and (-1,-1.64) .. (-1.4,-1.7) -- cycle;
\draw[line width=.9pt] (-1.4,-1.42) .. controls (-1,-1.36) and (-0.5,-1.36) .. (-0.1,-1.42);
\draw[line width=.9pt] (-1.4,-1.7) .. controls (-1,-1.64) and (-0.5,-1.64) .. (-0.1,-1.7);
\draw[gray!60] (-1.4,-0.9) rectangle (-0.1,-2.2);
\fill[gray!10] (0.1,-0.9) rectangle (1.4,-2.2);
\shade[top color=gray!10,bottom color=gray!10,middle color=swpurple!80,shading angle=0] (0.1,-1.25) rectangle (1.4,-1.87);
\draw[gray!60] (0.1,-0.9) rectangle (1.4,-2.2);
\node[tiny,anchor=north] at (-0.75,-2.2) {rasterized};
\node[tiny,anchor=north] at (0.75,-2.2) {$p\ge0.5$};
\node[note,anchor=north] at (0,-2.62) {Same crop and grid for both; no tuned threshold or post-processing};
\end{scope}
% Step 4: per-column first-to-last extent
\begin{scope}[shift={(10.95,0)}]
'''+grid+r'''
\draw[{Stealth[length=1.2mm]}-{Stealth[length=1.2mm]},line width=.8pt,sworange!80!black] (-1.06,-0.85-3*0.19)--(-1.06,-0.85-6*0.19);
\node[tiny,anchor=east,inner sep=1pt] at (-1.1,-0.85-4.5*0.19) {$\tau_c$};
\node[tiny,text=swnavy] at (-0.95+6.5*0.19,-2.5) {$\times$ $<$2 px};
\node[tiny,anchor=south] (gk) at (-0.95+4.5*0.19,-0.83) {gap kept};
\node[note,anchor=north] at (0,-2.68) {$\tau_c$: first-to-last foreground row; columns with $<$2 pixels ignored};
\end{scope}
% Step 5: conversion to millimetres
\begin{scope}[shift={(14.6,0)}]
\node[font=\fontsize{8.5}{10}\selectfont] at (0,-1.0) {$\displaystyle\mathrm{CIMT}=\kappa\,\frac{s}{256}\,\frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}}\tau_c$};
\node[fill=swnavy,text=white,rounded corners=4pt,font=\fontsize{8}{10}\selectfont\bfseries,inner sep=4pt] at (0,-1.85) {CIMT in mm};
\node[note,anchor=north] at (0,-2.42) {$\mathcal{C}$: valid columns; $s$: square side (original px); $\kappa$: calibration (mm/pixel); empty mask $\to$ 0 mm};
\end{scope}
% Lower row: shared operator and exploratory training term
\node[stepbox=swgray,minimum width=6.85cm,minimum height=1.7cm,anchor=north west,align=center,text width=6.5cm] (shared) at (-1.6,-4.75)
 {Identical crop, resize, and extraction for model and reader delineations; no connected-component filtering. Rasterizing A1 alone shifts CIMT by $-0.0560$ mm.};
\node[pill=swgray] at (shared.north) {Shared measurement operator};
\node[stepbox=swpurple,minimum width=10.5cm,minimum height=1.7cm,anchor=north east] (loss) at (16.2,-4.75) {};
\node[font=\fontsize{8.5}{10}\selectfont] at ([yshift=-0.7cm]loss.north) {$\displaystyle\mathcal{L}_{\mathrm{CIMT}}=\frac{1}{HW}\sum_{c=1}^{W}\bigl|\hat{\tau}_c-\tau_c\bigr|,\qquad \hat{\tau}_c=\sum_r p_{r,c},\qquad \tau_c=\sum_r t_{r,c}$};
\node[font=\fontsize{8}{10}\selectfont] at ([yshift=0.26cm]loss.south) {Weight 0.2, added to the region objective (Dice + weighted cross-entropy); seeds 42 and 1};
\node[pill=swpurple] at (loss.north) {Exploratory objective (training only)};
\draw[thin flow,dashed] (b4.south)--([yshift=.24cm]b4.south|-loss.north);
\end{tikzpicture}''')

# Dataset composition (single column). Values from Table 1 and Methods.
src=[('Cyprus, clinical',1388,'swblue'),('Pisa, clinical',788,'swblue'),('Munich, technical',100,'sworange'),('Pisa, technical',100,'sworange'),('Porto, technical',100,'sworange'),('Torino, technical',100,'sworange'),('Toronto, technical',100,'sworange')]
bars='\n'.join(rf'\addplot[xbar,bar shift=0pt,bar width=7pt,fill={c}!70,draw={c}] coordinates {{({v},{k+1})}};' for k,(n,v,c) in enumerate(src))
S=7.0/2676  # cm per image for the partition bar
dev,tcl,tte=2275*S,326*S,75*S
build('dataset_composition',r'''\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]
\begin{axis}[name=a,width=5.4cm,height=4.3cm,scale only axis,xmin=0,xmax=1650,ymin=.4,ymax=7.6,y dir=reverse,
 ytick={1,...,7},yticklabels={'''+','.join('{'+n+'}' for n,_,_ in src)+r'''},
 xtick={0,500,1000,1500},xticklabel style={/pgf/number format/set thousands separator={{{{,}}}}},xlabel={Images},title={(a) Images per acquisition source},
 nodes near coords,point meta=x,every node near coord/.append style={font=\fontsize{7.5}{9}\selectfont,/pgf/number format/set thousands separator={{{{,}}}}}]
'''+bars+r'''
\end{axis}
\begin{scope}[shift={($(a.south west)+(-1.9,-2.25)$)},x=1cm,y=1cm]
\node[font=\fontsize{8.5}{10}\selectfont\bfseries,anchor=west] at (0,0.62) {(b) Patient-aware partition of 2,676 images};
\fill[swgray!35,draw=swgray] (0,0) rectangle ('''+f'{dev:.3f}'+r''',-0.45);
\fill[swgreen!70,draw=swgreen] ('''+f'{dev:.3f}'+r''',0) rectangle ('''+f'{dev+tcl:.3f}'+r''',-0.45);
\fill[sworange!75,draw=sworange] ('''+f'{dev+tcl:.3f}'+r''',0) rectangle ('''+f'{dev+tcl+tte:.3f}'+r''',-0.45);
\node at ('''+f'{dev/2:.3f}'+r''',-0.225) {Development pool: 2,275 (five folds)};
\draw[decorate,decoration={brace,amplitude=4pt},line width=.7pt] ('''+f'{dev:.3f}'+r''',0.05)--('''+f'{dev+tcl+tte:.3f}'+r''',0.05) node[midway,above=3pt,font=\fontsize{8}{10}\selectfont\bfseries] {401};
\draw[swgreen!80!black] ('''+f'{dev+tcl/2:.3f}'+r''',-0.45)--++(0,-0.3)--(4.3,-0.75) node[anchor=east,font=\fontsize{7.5}{9}\selectfont,text=black] {326 clinical (163 patients)};
\draw[sworange!80!black] ('''+f'{dev+tcl+tte/2:.3f}'+r''',-0.45)--++(0,-0.62)--(4.3,-1.07) node[anchor=east,font=\fontsize{7.5}{9}\selectfont,text=black] {75 technical images};
\node[anchor=east,font=\fontsize{7.5}{9}\selectfont\itshape] at ('''+f'{dev+tcl+tte:.3f}'+r''',-1.42) {Fixed test: 238 resampling clusters};
\end{scope}
\end{tikzpicture}''')

# Rank shift (single column). Ranks from Table 2; rho and CI from Results.
rk=[('BC-VMamba',1,1),('U-Mamba',2,5),('ResU-Net',3,6),('Attention U-Net',4,4),('TransUNet',5,2),('SegFormer',6,7),('U-Net',7,8),('Swin-UNet',8,3)]
dy=0.44
lines=[]
for n,d,m in rk:
 col='swgreen' if n=='BC-VMamba' else ('swblue' if m<d else ('sworange' if m>d else 'swgray'))
 lw='2.2pt' if n=='BC-VMamba' else '1.3pt'
 lines.append(rf'\draw[{col},line width={lw}] (2.75,{-d*dy:.2f})--(5.35,{-m*dy:.2f});')
 lines.append(rf'\node[circle,fill={col},text=white,inner sep=0pt,minimum size=11pt,font=\fontsize{{7.5}}{{8}}\selectfont\bfseries] at (2.75,{-d*dy:.2f}) {{{d}}};')
 lines.append(rf'\node[circle,fill={col},text=white,inner sep=0pt,minimum size=11pt,font=\fontsize{{7.5}}{{8}}\selectfont\bfseries] at (5.35,{-m*dy:.2f}) {{{m}}};')
 bf=r'\bfseries' if n=='BC-VMamba' else ''
 lines.append(rf'\node[anchor=east,font=\fontsize{{8}}{{10}}\selectfont{bf}] at (2.5,{-d*dy:.2f}) {{{n}}};')
 lines.append(rf'\node[anchor=west,font=\fontsize{{8}}{{10}}\selectfont{bf}] at (5.6,{-m*dy:.2f}) {{{n}}};')
build('rank_shift',r'''\begin{tikzpicture}[x=1cm,y=1cm,font=\fontsize{8}{10}\selectfont]
\node[pill=swblue] at (2.75,0.2) {Rank by Dice};
\node[pill=sworange] at (5.35,0.2) {Rank by MAE};
'''+'\n'.join(lines)+r'''
\begin{scope}[shift={(0.7,-4.25)},font=\fontsize{7.5}{9}\selectfont]
\draw[swblue,line width=1.3pt] (0,0)--(0.4,0) node[right,text=black] {better by MAE};
\draw[sworange,line width=1.3pt] (2.35,0)--(2.75,0) node[right,text=black] {worse by MAE};
\draw[swgray,line width=1.3pt] (4.65,0)--(5.05,0) node[right,text=black] {unchanged};
\end{scope}
\node[font=\fontsize{8}{10}\selectfont] at (4.05,-4.72) {Spearman $\rho=0.357$ (95\% CI $-0.048$ to $0.833$)};
\end{tikzpicture}''')

# Paired differences forest plot (single column). Values from Table 3, sorted by mean.
pd=[('U-Net',.1287,.0807,.1854,.0050,r'$<\!10^{-6}$','29/3'),('SegFormer',.0791,.0181,.1977,.0225,r'$<\!10^{-6}$','4/3'),('ResU-Net',.0745,.0431,.1124,-.0039,'0.288','26/3'),('U-Mamba',.0525,.0224,.0929,.0073,r'$<\!10^{-4}$','17/3'),('Attention U-Net',.0519,.0245,.0842,.0035,'0.003','17/3'),('Swin-UNet',.0432,-.0024,.1275,-.0059,'0.521','6/3'),('TransUNet',.0413,.0107,.0841,.0034,'0.036','9/3')]
fp=[]
for k,(n,m,l,h,md,q,t) in enumerate(pd):
 y=k+1
 fp.append(rf'\addplot[swblue,line width=1.1pt] coordinates {{({l},{y}) ({h},{y})}};')
 fp.append(rf'\addplot[only marks,mark=*,mark size=2.1pt,swblue] coordinates {{({m},{y})}};')
 fp.append(rf'\addplot[only marks,mark=diamond*,mark size=2.4pt,sworange,mark options={{draw=white,line width=.3pt}}] coordinates {{({md},{y})}};')
 fp.append(rf'\node[anchor=center] at (axis cs:0.247,{y}) {{{q}}};')
 fp.append(rf'\node[anchor=center] at (axis cs:0.300,{y}) {{{t}}};')
build('paired_differences',r'''\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]
\begin{axis}[width=4.0cm,height=4.6cm,scale only axis,clip=false,xmin=-0.02,xmax=0.21,ymin=.1,ymax=7.6,y dir=reverse,
 ytick={1,...,7},yticklabels={'''+','.join(p[0] for p in pd)+r'''},
 xtick={0,0.05,0.10,0.15,0.20},xticklabel style={/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=2},
 xlabel={Paired CIMT error difference (mm)},axis y line*=left,
 title={Baseline minus BC-VMamba}]
\fill[swgreen!8] (axis cs:0,.1) rectangle (axis cs:0.21,7.6);
\draw[swgray,dashed,line width=.8pt] (axis cs:0,.1)--(axis cs:0,7.6);
'''+'\n'.join(fp)+r'''
\node[font=\fontsize{8}{10}\selectfont\bfseries] at (axis cs:0.247,-0.25) {$q$};
\node[font=\fontsize{8}{10}\selectfont\bfseries] at (axis cs:0.300,-0.25) {$>$0.5 mm};
\node[font=\fontsize{7}{8}\selectfont\itshape,text=swgreen!70!black,anchor=north east] at (axis cs:0.207,0.12) {favors BC-VMamba $\rightarrow$};
\end{axis}
\begin{scope}[shift={(0.0,-1.35)},font=\fontsize{7.5}{9}\selectfont]
\draw[swblue,line width=1.1pt] (0,0)--(0.5,0);\fill[swblue] (0.25,0) circle (2.1pt);
\node[right] at (0.55,0) {mean, 95\% cluster-bootstrap CI};
\node[sworange,font=\fontsize{9}{9}\selectfont] at (4.45,0) {$\blacklozenge$};\node[right] at (4.55,0) {median};
\end{scope}
\end{tikzpicture}''')

# Architecture endpoints (full width). Values from Table 2.
rows=[('BC-VMamba',.8442,.8380,.8502,.1241,.1150,.1335,3),('U-Mamba',.8369,.8307,.8429,.1765,.1461,.2170,17),('ResU-Net',.8287,.8223,.8347,.1986,.1659,.2381,26),('Attention U-Net',.8258,.8193,.8317,.1760,.1481,.2107,17),('TransUNet',.8257,.8191,.8319,.1654,.1339,.2113,9),('SegFormer',.8224,.8152,.8292,.2031,.1407,.3210,4),('U-Net',.8021,.7953,.8084,.2528,.2037,.3108,29),('Swin-UNet',.7956,.7886,.8023,.1673,.1193,.2561,6)]
body=r'\begin{tikzpicture}'+'\n'
panels=[('(a) Segmentation overlap','Dice (higher is better)',.78,.86,'swblue','{0.78,0.80,0.82,0.84,0.86}',2),
        ('(b) Measurement error','CIMT MAE (mm; lower is better)',0,.35,'sworange','{0,0.1,0.2,0.3}',1),
        ('(c) Large-error tail','Images with error $>$0.5 mm',0,33,'swpurple','{0,10,20,30}',0)]
for j,(title,label,lo,hi,col,ticks,prec) in enumerate(panels):
 fmt=rf'/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision={prec}' if prec else ''
 opts=rf'''at={{({j*5.25}cm,0)}},anchor=west,width=3.75cm,height=4.6cm,scale only axis,xmin={lo},xmax={hi},ymin=.5,ymax=8.5,y dir=reverse,ytick={{1,2,3,4,5,6,7,8}},title={{{title}}},xlabel={{{label}}},xtick={ticks},xticklabel style={{{fmt}}}'''
 if j==0:opts+=',yticklabels={'+','.join(r[0] for r in rows)+'}'
 else:opts+=',yticklabels={,,,,,,,}'
 body+='\\begin{axis}['+opts+']\n'
 if j<2:
  for k,r in enumerate(rows):
   v,l,h=(r[1:4] if j==0 else r[4:7]);c=col if k else 'swgreen'
   body+=rf'\addplot[{c},line width=1.1pt] coordinates {{({l},{k+1}) ({h},{k+1})}};'+'\n'
   body+=rf'\addplot[only marks,mark={"diamond*" if k==0 else "*"},mark size={"3" if k==0 else "2.1"}pt,{c}] coordinates {{({v},{k+1})}};'+'\n'
 else:
  for k,r in enumerate(rows):
   c=col if k else 'swgreen'
   body+=rf'\addplot[xbar,bar shift=0pt,bar width=7pt,fill={c}!70,draw={c},nodes near coords,point meta=x,every node near coord/.append style={{font=\fontsize{{7.5}}{{9}}\selectfont}}] coordinates {{({r[7]},{k+1})}};'+'\n'
 body+='\\end{axis}\n'
body+='\\end{tikzpicture}'
build('architecture_endpoints',body)

# Seed sensitivity (single column). (a) Values from Results; (b) per-image errors from data/run_pairs.csv.
RP=EXP['runs']
build('seed_sensitivity',r'''\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]
\begin{axis}[name=a,width=4.8cm,height=3.2cm,scale only axis,clip=false,xmin=0,xmax=.155,ymin=.45,ymax=3.55,y dir=reverse,
 ytick={1,2,3},yticklabels={Region-only / seed 42,Aligned / seed 42,Aligned / seed 1},
 xlabel={Fixed-test CIMT MAE (mm)},xtick={0,.05,.10,.15},
 xticklabel style={/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=2},
 xbar,bar shift=0pt,bar width=10pt,nodes near coords,point meta=x,
 every node near coord/.append style={font=\fontsize{8}{10}\selectfont,/pgf/number format/fixed,/pgf/number format/precision=4,/pgf/number format/fixed zerofill}]
\addplot[fill=swblue!70,draw=swblue] coordinates {(.1241,1)};
\addplot[fill=swgreen!70,draw=swgreen] coordinates {(.1200,2)};
\addplot[fill=sworange!70,draw=sworange] coordinates {(.1307,3)};
\draw[swnavy,line width=.8pt] (axis cs:.004,1.3)--(axis cs:0.002,1.3)--(axis cs:0.002,1.7)--(axis cs:.004,1.7);
\node[anchor=west,fill=white,fill opacity=.85,text opacity=1,inner sep=1.5pt,font=\fontsize{7.5}{9}\selectfont,align=left] at (axis cs:.006,1.5)
 {Matched seed 42: paired change $-0.0040$ mm\\(95\% CI $-0.0065$ to $-0.0014$ mm)};
\end{axis}
\node[anchor=south east,font=\fontsize{9}{11}\selectfont\bfseries] at ([xshift=-2.6cm]a.north west) {(a)};
\begin{axis}[name=b,at={($(a.south west)+(0,-1.5cm)$)},anchor=north west,width=4.8cm,height=3.6cm,scale only axis,
 xmode=log,ymode=log,xmin=0.0008,xmax=1.5,ymin=0.0008,ymax=1.5,ymajorgrids,
 xlabel={Absolute error, aligned / seed 42 (mm)},ylabel={Aligned / seed 1 (mm)},
 log ticks with fixed point,xtick={0.001,0.01,0.1,1},ytick={0.001,0.01,0.1,1}]
\addplot[swgray,dashed,line width=.7pt,domain=0.0008:1.5,samples=2] {x};
\addplot[only marks,mark=*,mark size=1.1pt,swpurple,mark options={fill opacity=.4,draw opacity=.7}] table[col sep=comma,x=aligned_s42,y=aligned_s1] {../data/run_pairs.csv};
\node[anchor=north west,fill=white,fill opacity=.85,text opacity=1,inner sep=1.5pt,font=\fontsize{7.5}{9}\selectfont,align=left] at (rel axis cs:0.02,0.98)
 {Spearman $\rho=$ '''+f"{RP['rho_err_aligned42_aligned1'][0]:.3f}"+r'''};
\end{axis}
\node[anchor=south east,font=\fontsize{9}{11}\selectfont\bfseries] at ([xshift=-2.6cm]b.north west) {(b)};
\end{tikzpicture}''')

# ---------- Post hoc exploratory figures (numbers from analysis/exploratory_numbers.json) ----------
FMT2=r'/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=2'
FMT1=r'/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=1'
def lab2(t,ax):
 return rf'\node[anchor=north east,font=\fontsize{{9}}{{11}}\selectfont\bfseries] at ([xshift=-0.08cm]{ax}.north west) {{{t}}};'
def lab(t,ax):
 return rf'\node[anchor=south east,font=\fontsize{{9}}{{11}}\selectfont\bfseries] at ([xshift=-0.75cm,yshift=0.05cm]{ax}.north west) {{{t}}};'
fail_split=r'''\addplot[only marks,mark=*,mark size=1.1pt,swblue,mark options={fill opacity=.35,draw opacity=.6},x filter/.expression={\thisrow{abs_err}>0.1686 ? nan : x}] table[col sep=comma,x=XX,y=YY] {../data/geometry_points.csv};
\addplot[only marks,mark=triangle*,mark size=1.6pt,sworange,mark options={fill opacity=.6,draw opacity=.9},x filter/.expression={\thisrow{abs_err}>0.1686 ? x : nan}] table[col sep=comma,x=XX,y=YY] {../data/geometry_points.csv};'''
G=EXP['geometry']['region_s42']
iso=''.join(rf'\addplot[swgray,line width=.5pt,domain={dd/2+0.006:.3f}:1,samples=60] {{{dd}*x/(2*x-{dd})}};\node[font=\fontsize{{6.5}}{{8}}\selectfont,text=swgray,anchor=south,inner sep=1pt] at (axis cs:{dd/(2-dd):.4f},1) {{{dd:.2f}}};' for dd in (0.75,0.80,0.85,0.90))
build('geometry_mechanism',r'''\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]
\begin{axis}[name=a,width=4.3cm,height=4.3cm,scale only axis,clip mode=individual,xmin=0.6,xmax=1,ymin=0.5,ymax=1,ymajorgrids,
 xlabel={Recall},ylabel={Precision},xticklabel style={'''+FMT1+r'''},yticklabel style={'''+FMT1+r'''},title={Size balance and Dice}]
'''+iso+r'''
\addplot[swnavy,dashed,line width=.6pt,domain=0.6:1,samples=2] {x};
'''+fail_split.replace('XX','recall').replace('YY','precision')+r'''
\node[anchor=north west,font=\fontsize{7}{8.5}\selectfont,fill=white,fill opacity=.85,text opacity=1,inner sep=1pt,align=left] at (rel axis cs:0.02,0.9) {Recall $>$ precision:\\'''+f"{G['share_recall_gt_precision'][0]*100:.1f}\\% of images"+r'''};
\end{axis}
'''+lab('(a)','a')+r'''
\begin{axis}[name=b,at={($(a.east)+(1.45cm,0)$)},anchor=west,width=4.3cm,height=4.3cm,scale only axis,xmin=-0.6,xmax=1.2,ymin=-0.6,ymax=1.2,ymajorgrids,
 xlabel={$\log_2$(recall / precision)},ylabel={Signed CIMT error (mm)},xticklabel style={'''+FMT1+r'''},yticklabel style={'''+FMT1+r'''},title={Size ratio and signed error}]
\draw[swgray,line width=.4pt] (axis cs:-0.6,0)--(axis cs:1.2,0);\draw[swgray,line width=.4pt] (axis cs:0,-0.6)--(axis cs:0,1.2);
'''+fail_split.replace('XX','log2rp').replace('YY','signed')+r'''
\node[anchor=north west,font=\fontsize{7}{8.5}\selectfont,fill=white,fill opacity=.85,text opacity=1,inner sep=1pt,align=left] at (rel axis cs:0.02,0.98) {Mean signed error\\'''+f"$+{G['mean_signed_mm'][0]:.3f}$ mm ($+{G['mean_signed_px'][0]:.2f}$ px)"+r'''};
\end{axis}
'''+lab('(b)','b')+r'''
\begin{axis}[name=c,at={($(b.east)+(1.45cm,0)$)},anchor=west,width=4.3cm,height=4.3cm,scale only axis,xmin=0,xmax=14,ymin=0.6,ymax=0.95,ymajorgrids,
 xlabel={Reference thickness (grid pixels)},ylabel={Dice},yticklabel style={'''+FMT2+r'''},title={Dice and wall thickness}]
'''+fail_split.replace('XX','ref_px').replace('YY','dice')+r'''
\node[anchor=south east,font=\fontsize{7}{8.5}\selectfont,fill=white,fill opacity=.85,text opacity=1,inner sep=1pt,align=right] at (rel axis cs:0.98,0.02) {Spearman $\rho=$ '''+f"{G['rho_dice_ref_px'][0]:.3f}"+r'''};
\end{axis}
'''+lab('(c)','c')+r'''
\begin{scope}[shift={($(a.south west)+(0.2,-1.25)$)},font=\fontsize{7.5}{9}\selectfont]
\fill[swblue,opacity=.6] (0,0) circle (1.6pt);\node[right] at (0.1,0) {CIMT error $\leq$ 0.1686 mm};
\fill[sworange] (3.2,-0.06)--(3.32,-0.06)--(3.26,0.07)--cycle;\node[right] at (3.35,0) {CIMT error $>$ 0.1686 mm};
\draw[swgray,line width=.5pt] (6.5,0)--(7.0,0);\node[right] at (7.05,0) {iso-Dice contour};
\draw[swnavy,dashed,line width=.6pt] (9.2,0)--(9.7,0);\node[right] at (9.75,0) {recall = precision};
\end{scope}
\end{tikzpicture}''')

# Error distributions of the architectures with archived per-image outputs.
AR={a['model']:a for a in EXP['architectures']}
cols={'BC-VMamba':'swgreen','TransUNet':'swblue','U-Net':'sworange'}
files={'BC-VMamba':'ecdf_bcvmamba.csv','TransUNet':'ecdf_transunet.csv','U-Net':'ecdf_unet.csv'}
ec=''.join(rf'\addplot[{cols[m]},line width=1.1pt,const plot] table[col sep=comma,x=abs_err,y=cdf] {{../data/{files[m]}}};\addlegendentry{{{m}}}' for m in cols)
dots=''
for k,m in enumerate(cols):
 a=AR[m];y=k+1
 for key,mk,off in [('median_abs_err','*',-0.13),('p90_abs_err','square*',0.13)]:
  v,l,h=a[key]
  dots+=rf'\addplot[{cols[m]},line width=1pt] coordinates {{({l},{y+off}) ({h},{y+off})}};\addplot[only marks,mark={mk},mark size=2pt,{cols[m]}] coordinates {{({v},{y+off})}};'
sb=''
for m in cols:
 a=AR[m];x,xl,xh=a['mean_log2_rp'];y,yl,yh=a['mean_signed_mm']
 sb+=rf'\addplot[{cols[m]},line width=.9pt] coordinates {{({xl},{y}) ({xh},{y})}};\addplot[{cols[m]},line width=.9pt] coordinates {{({x},{yl}) ({x},{yh})}};\addplot[only marks,mark=*,mark size=2.4pt,{cols[m]}] coordinates {{({x},{y})}};'
 sb+=rf'\node[anchor=west,font=\fontsize{{7.5}}{{9}}\selectfont,text={cols[m]}!80!black] at (axis cs:{xh+0.004},{y-0.012}) {{{m}}};'
build('architecture_distributions',r'''\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]
\begin{axis}[name=a,width=4.3cm,height=4.0cm,scale only axis,xmin=0,xmax=0.6,ymin=0,ymax=1.02,ymajorgrids,
 xlabel={Absolute CIMT error (mm)},ylabel={Cumulative share of images},xticklabel style={'''+FMT1+r'''},yticklabel style={'''+FMT1+r'''},
 title={Error distribution},legend style={at={(0.97,0.03)},anchor=south east,draw=none,fill=white,fill opacity=.85,text opacity=1,font=\fontsize{7.5}{9}\selectfont},legend cell align=left]
'''+ec+r'''
\end{axis}
'''+lab('(a)','a')+r'''
\begin{axis}[name=b,at={($(a.east)+(2.15cm,0)$)},anchor=west,width=3.9cm,height=4.0cm,scale only axis,clip=false,xmin=0,xmax=0.47,ymin=0.4,ymax=3.6,y dir=reverse,
 ytick={1,2,3},yticklabels={'''+','.join(f"{{{m}\\\\[-1pt]\\footnotesize MAE {AR[m]['mae']:.3f}}}" for m in cols)+r'''},yticklabel style={align=right},xlabel={Absolute CIMT error (mm)},xticklabel style={'''+FMT1+r'''},title={Typical and upper-tail error}]
'''+dots+r'''
\end{axis}
'''+lab('(b)','b')+r'''
\begin{axis}[name=c,at={($(b.east)+(1.75cm,0)$)},anchor=west,width=3.9cm,height=4.0cm,scale only axis,clip=false,xmin=0.12,xmax=0.30,ymin=0,ymax=0.26,ymajorgrids,
 xlabel={Mean $\log_2$(recall / precision)},ylabel={Mean signed CIMT error (mm)},xticklabel style={'''+FMT2+r'''},yticklabel style={'''+FMT2+r'''},title={Over-segmentation and bias}]
'''+sb+r'''
\end{axis}
'''+lab('(c)','c')+r'''
\begin{scope}[shift={($(b.south west)+(0.1,-1.2)$)},font=\fontsize{7.5}{9}\selectfont]
\fill[black] (0,0) circle (2pt);\node[right] at (0.1,0) {median};\fill[black] (1.25,-0.07) rectangle (1.39,0.07);\node[right] at (1.42,0) {90th percentile; lines are 95\% cluster-bootstrap CIs};
\end{scope}
\end{tikzpicture}''')

# Discordance: reference-dependent ROC curves and size ratio by quadrant (aligned s42 ensemble).
dp=list(csv.DictReader(open(DATA/'discordance_points.csv')))
assert len(dp)==401
lab01=[float(x['abs_err'])>0.1686 for x in dp]
def roc(score):
 pts=sorted(zip(score,lab01),key=lambda t:-t[0]);P=sum(lab01);N=len(lab01)-P;tp=fp=0;out=[(0.0,0.0)];auc=0.0;prev=None
 for s,l in pts:
  if l:tp+=1
  else:fp+=1;auc+=tp
  out.append((fp/N,tp/P))
 return out,auc/(P*N)
D=EXP['discordance']
rocs=''
for nm,key,col,sty in [('1 $-$ Dice','auc_1minus_dice_cimt_fail','swblue','dashed'),(r'$|\log_2(\mathrm{recall}/\mathrm{precision})|$','auc_abs_log2rp_cimt_fail','swpurple','densely dashed')]:
 sc=[1-float(x['dice']) for x in dp] if key.startswith('auc_1minus') else [abs(float(x['log2rp'])) for x in dp]
 pts,auc=roc(sc);assert round(auc,3)==round(D[key][0],3),(key,auc)
 rocs+=rf'\addplot[{col},{sty},line width=1.1pt] coordinates {{'+' '.join(f'({a:.4f},{b:.4f})' for a,b in pts)+'};'
 rocs+=rf'\addlegendentry{{{nm}: {D[key][0]:.3f}}}'
qs=[('highDice_ok','High Dice,\\\\no failure'),('highDice_fail','High Dice,\\\\CIMT failure'),('lowDice_ok','Low Dice,\\\\no failure'),('lowDice_fail','Low Dice,\\\\CIMT failure')]
qcol={'highDice_ok':'swblue','highDice_fail':'sworange','lowDice_ok':'swblue','lowDice_fail':'sworange'}
strip=''
for k,(q,_) in enumerate(qs):
 vals=sorted(float(x['log2rp']) for x in dp if x['quadrant']==q);n=len(vals);assert n==QUAD[q]
 for i,v in enumerate(vals):
  strip+=rf'\addplot[only marks,mark=*,mark size=.9pt,{qcol[q]},mark options={{fill opacity=.35,draw opacity=.5}}] coordinates {{({k+1+((i*37)%21-10)/36:.3f},{v:.4f})}};'
 med=statistics.median(vals)
 strip+=rf'\draw[black,line width=1.3pt] (axis cs:{k+0.68},{med:.4f})--(axis cs:{k+1.32},{med:.4f});\node[font=\fontsize{{7}}{{8}}\selectfont,anchor=south] at (axis cs:{k+1},1.32) {{$n={n}$}};'
build('discordance',r'''\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]
\begin{axis}[name=a,width=5.2cm,height=3.9cm,scale only axis,xmin=0,xmax=1,ymin=0,ymax=1,ymajorgrids,xlabel={False positive rate},ylabel={True positive rate},
 xticklabel style={'''+FMT1+r'''},yticklabel style={'''+FMT1+r'''},title={Reference-dependent detectors of CIMT failure},
 legend style={at={(0.98,0.03)},anchor=south east,draw=none,fill=white,fill opacity=.9,text opacity=1,font=\fontsize{7}{8.5}\selectfont},legend cell align=left]
\addplot[swgray,line width=.5pt,domain=0:1,samples=2] {x};\addlegendentry{chance}
'''+rocs+r'''
\end{axis}
'''+lab('(a)','a')+r'''
\begin{axis}[name=b,at={($(a.south west)+(0,-2.0cm)$)},anchor=north west,width=5.2cm,height=3.6cm,scale only axis,clip=false,xmin=0.4,xmax=4.6,ymin=-0.6,ymax=1.5,ymajorgrids,
 xtick={1,2,3,4},xticklabels={'''+','.join('{'+t+'}' for _,t in qs)+r'''},xticklabel style={align=center,font=\fontsize{7}{8}\selectfont},
 ylabel={$\log_2$(recall / precision)},yticklabel style={'''+FMT1+r'''},title={Size ratio by Dice/CIMT quadrant}]
\draw[swgray,line width=.4pt] (axis cs:0.4,0)--(axis cs:4.6,0);
'''+strip+r'''
\end{axis}
'''+lab('(b)','b')+r'''
\end{tikzpicture}''')

# Model-selection practice: training history (aligned s1) and 15 test checkpoints.
TD=list(csv.DictReader(open(DATA/'training_dynamics.csv')))
SL={int(s['fold']):s for s in EXP['selection']['folds']}
fc=['swblue','swgreen','sworange','swpurple','swnavy']
traj='';marks=''
for f in range(1,6):
 pts=[(int(x['epoch']),float(x['val_mae_cimt_mm'])) for x in TD if int(x['fold'])==f and x['eligible']=='True']
 traj+=rf'\addplot[{fc[f-1]},line width=.7pt,opacity=.85] coordinates {{'+' '.join(f'({e},{v:.5f})' for e,v in pts)+'};'
 s=SL[f]
 marks+=rf'\addplot[only marks,mark=*,mark size=2.2pt,{fc[f-1]},mark options={{draw=black,line width=.4pt}}] coordinates {{({s["epoch_best_dice"]},{s["mae_at_best_dice"]:.5f})}};'
 marks+=rf'\addplot[only marks,mark=triangle*,mark size=2.8pt,{fc[f-1]},mark options={{draw=black,line width=.4pt}}] coordinates {{({s["epoch_best_mae"]},{s["min_mae"]:.5f})}};'
bars=''.join(rf'\addplot[xbar,bar shift=0pt,bar width=8pt,fill={fc[f-1]}!65,draw={fc[f-1]}] coordinates {{({SL[f]["regret_mm"]:.5f},{f})}};' for f in range(1,6))
CK=list(csv.DictReader(open(DATA/'checkpoints15.csv')))
rc={'region_s42':('swblue','*','Region-only / seed 42'),'aligned_s42':('swgreen','square*','Aligned / seed 42'),'aligned_s1':('sworange','triangle*','Aligned / seed 1')}
cks=''.join(rf'\addplot[only marks,mark={rc[r][1]},mark size=2.3pt,{rc[r][0]}] coordinates {{'+' '.join(f"({float(x['dice']):.5f},{float(x['mae']):.5f})" for x in CK if x['run']==r)+rf'}};\addlegendentry{{{rc[r][2]}}}' for r in rc)
SE=EXP['selection']
build('model_selection',r'''\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]
\begin{axis}[name=a,width=4.4cm,height=4.0cm,scale only axis,xmin=5,xmax=100,ymin=0.105,ymax=0.175,ymajorgrids,
 xlabel={Epoch},ylabel={Validation CIMT MAE (mm)},yticklabel style={'''+FMT2+r'''},title={Validation trajectories (aligned / seed 1)}]
'''+traj+marks+r'''
\end{axis}
'''+lab('(a)','a')+r'''
\begin{axis}[name=b,at={($(a.east)+(1.75cm,0)$)},anchor=west,width=3.6cm,height=4.0cm,scale only axis,clip=false,xmin=0,xmax=0.016,ymin=0.4,ymax=5.6,y dir=reverse,
 ytick={1,...,5},yticklabels={Fold 1,Fold 2,Fold 3,Fold 4,Fold 5},xtick={0,0.005,0.010,0.015},
 xticklabel style={/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=3},
 xlabel={Selection regret (mm)},scaled x ticks=false,title={Cost of selecting by Dice}]
'''+bars+r'''
\draw[swgray,dashed,line width=.8pt] (axis cs:0.0040,0.4)--(axis cs:0.0040,5.6);
\node[anchor=west,font=\fontsize{7}{8.5}\selectfont,text=swgray,align=left,fill=white,inner sep=1pt] at (axis cs:0.0042,3.5) {matched\\aligned-loss\\effect};
\end{axis}
'''+lab('(b)','b')+r'''
\begin{axis}[name=c,at={($(b.east)+(1.75cm,0)$)},anchor=west,width=4.0cm,height=4.0cm,scale only axis,xmin=0.840,xmax=0.851,ymin=0.110,ymax=0.162,ymajorgrids,
 xlabel={Fixed-test Dice},ylabel={Fixed-test CIMT MAE (mm)},xtick={0.840,0.845,0.850},
 xticklabel style={/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=3},yticklabel style={/pgf/number format/fixed,/pgf/number format/fixed zerofill,/pgf/number format/precision=2},ytick={0.11,0.12,0.13,0.14,0.15},
 title={Fifteen BC-VMamba checkpoints},legend style={at={(0.97,0.97)},anchor=north east,draw=none,fill=white,fill opacity=.85,text opacity=1,font=\fontsize{7}{8.5}\selectfont},legend cell align=left]
'''+cks+r'''
\end{axis}
'''+lab('(c)','c')+r'''
\begin{scope}[shift={($(a.south west)+(0.1,-1.2)$)},font=\fontsize{7.5}{9}\selectfont]
\fill[black] (0,0) circle (2pt);\node[right] at (0.1,0) {epoch with maximum validation Dice};
\fill[black] (4.9,-0.08)--(5.06,-0.08)--(4.98,0.08)--cycle;\node[right] at (5.1,0) {epoch with minimum validation MAE};
\end{scope}
\end{tikzpicture}''')

# Source strata with reader variability (aligned s42, procedure-matched readers).
SS=EXP['sources']
nice={'Cyprus':'Cyprus','Pisa_clin':'Pisa (clin.)','Munich':'Munich','Pisa_tech':'Pisa (tech.)','Porto':'Porto','Torino':'Torino','Toronto':'Toronto'}
ylabels=','.join(f"{{{nice[s['center']]} ($n={s['n']}$)}}" for s in SS)
def forest(key,col,mk='*',off=0.0):
 b=''
 for k,s in enumerate(SS):
  v,l,h=s[key];y=k+1+off
  b+=rf'\addplot[{col},line width=1pt] coordinates {{({l},{y}) ({h},{y})}};\addplot[only marks,mark={mk},mark size=2pt,{col}] coordinates {{({v},{y})}};'
 return b
build('source_strata',r'''\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]
\begin{axis}[name=a,width=3.6cm,height=4.4cm,scale only axis,xmin=0.78,xmax=0.92,ymin=0.4,ymax=7.6,y dir=reverse,ytick={1,...,7},yticklabels={'''+ylabels+r'''},
 xlabel={Dice},xtick={0.80,0.85,0.90},xticklabel style={'''+FMT2+r'''},title={Segmentation overlap}]
'''+forest('dice','swblue')+r'''
\end{axis}
'''+lab('(a)','a')+r'''
\begin{axis}[name=b,at={($(a.east)+(0.55cm,0)$)},anchor=west,width=4.4cm,height=4.4cm,scale only axis,xmin=0,xmax=0.3,ymin=0.4,ymax=7.6,y dir=reverse,ytick={1,...,7},yticklabels={},
 xlabel={Absolute CIMT error (mm)},xticklabel style={'''+FMT1+r'''},title={Model and reader disagreement},
 legend style={at={(0.5,-0.3)},anchor=north,draw=none,fill=none,legend columns=3,font=\fontsize{7.5}{9}\selectfont,/tikz/every even column/.append style={column sep=6pt}},legend cell align=left]
\addlegendimage{only marks,mark=*,swgreen}\addlegendentry{Model $-$ A1}
\addlegendimage{only marks,mark=square*,swpurple}\addlegendentry{A1$^{\prime}$ $-$ A1}
\addlegendimage{only marks,mark=diamond*,swgray}\addlegendentry{A2 $-$ A1}
'''+forest('mae_mm','swgreen','*',-0.22)+forest('a1p_mae','swpurple','square*',0)+forest('a2_mae','swgray','diamond*',0.22)+r'''
\end{axis}
'''+lab2('(b)','b')+r'''
\begin{axis}[name=c,at={($(b.east)+(0.55cm,0)$)},anchor=west,width=3.6cm,height=4.4cm,scale only axis,xmin=0,xmax=2.2,ymin=0.4,ymax=7.6,y dir=reverse,ytick={1,...,7},yticklabels={},
 xlabel={Model MAE (grid pixels)},xticklabel style={'''+FMT1+r'''},title={Error in pixel units}]
\draw[swgray,dashed,line width=.7pt] (axis cs:1,0.4)--(axis cs:1,7.6);
'''+forest('mae_px','sworange')+r'''
\end{axis}
'''+lab2('(c)','c')+r'''
\end{tikzpicture}''')
print('Built fifteen vector PDF figures and PNG previews in',OUT)
