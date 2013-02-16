#!/usr/bin/awk

{print $17; for(i=1; i<=15;i = i + 2) {print $i " " $(i+1) " "}; print "\n\n"}
