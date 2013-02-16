all:
	cat wildfire.csv | sed '1d' | sed 's/nosmoke/-1/' | sed 's/smoke/1/' > wf.csv
	pdflatex doku-12.tex
