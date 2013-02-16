all:
	cat pendigits-testing.txt | awk -f ./make_plt.awk > pen.tst
	cat pendigits-training.txt | awk -f ./make_plt.awk > pen.trn
	mkdir images 2>/dev/null
	gnuplot visual.gnu

small:
	cat pendigits-testing-small.txt | awk -f ./make_plt.awk > pen.tst
	cat pendigits-training-small.txt | awk -f ./make_plt.awk > pen.trn
	mkdir images 2>/dev/null
	gnuplot visual-small.gnu
