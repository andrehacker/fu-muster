set title "Pendigits Visualized"
set terminal png

set size square
set xrange [0:100]
set yrange [0:100]


filename = sprintf("images/mean%i.png",i)
set output filename
set multiplot
plot 'mean_digits.plt' index i using (95):(88):1 every ::::0 with labels notitle\
     font "Arial,44"
plot 'mean_digits.plt' index i using 1:2 every ::1 with lines notitle
plot 'mean_digits.plt' index i using 1:2 every ::1::1 with points notitle ps 3 pt 6
unset multiplot
set output
