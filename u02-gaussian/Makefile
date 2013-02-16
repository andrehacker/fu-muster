all:
	matlab -nodesktop -nosplash -r 'covar; analyze; mitte; confusion; quit;' > /dev/null 
	cat mean_digits.mat | awk -f ./make_plt.awk > mean_digits.plt
	mkdir -p images 2>/dev/null
	echo "i=0" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=1" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=2" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=3" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=4" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=5" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=6" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=7" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=8" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=9" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	rm tmp.gnu
	cat analysis_results.txt
	cat confusion_matrix.mat

clean:
	rm -f *mat
	rm -f *txt
	rm -f *~
	rm -r -f images

visual:
	cat mean_digits.mat | awk -f ./make_plt.awk > mean_digits.plt
	mkdir -p images 2>/dev/null
	echo "i=0" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=1" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=2" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=3" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=4" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=5" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=6" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=7" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=8" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	echo "i=9" > tmp.gnu
	cat visual.gnu >> tmp.gnu
	gnuplot tmp.gnu
	rm tmp.gnu
