set title "Pendigits Visualized"
set terminal png

set size square
set xrange [0:100]
set yrange [0:100]



INDEX=1

do for[i=0:3497] {
filename = sprintf("images/pendigit-testing%i.png",i)
#message = sprintf("Generating images/pendigit-testing%i.png",i)
#print message
set output filename
set multiplot
plot 'pen.tst' index i using (95):(88):1 every ::::0 with labels notitle\
     font "Arial,44"
plot 'pen.tst' index i using 1:2 every ::2 with lines notitle
unset multiplot
set output
}

do for[i=0:7493] {
filename = sprintf("images/pendigit-training%i.png",i)
#message = sprintf("Generating images/pendigit-testing%i.png",i)
#print message
set output filename
set multiplot
plot 'pen.trn' index i using (95):(88):1 every ::::0 with labels notitle\
     font "Arial,44"
plot 'pen.trn' index i using 1:2 every ::2 with lines notitle
unset multiplot
set output
}
