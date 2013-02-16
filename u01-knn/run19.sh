#!/bin/sh

echo "matlabpool 8; nn(7494,3498,0,1,1,19); analyze" | matlab -nodisplay -nosplash | tee 19feat1k.txt
echo "matlabpool 8; nn(7494,3498,1,3,1,19); analyze" | matlab -nodisplay -nosplash | tee 19feat3k.txt
echo "matlabpool 8; nn(7494,3498,1,5,1,19); analyze" | matlab -nodisplay -nosplash | tee 19feat5k.txt
echo "matlabpool 8; nn(7494,3498,1,15,1,19); analyze" | matlab -nodisplay -nosplash | tee 19feat15k.txt

echo "matlabpool 8; nn(7494,3498,0,1,4,19); analyze" | matlab -nodisplay -nosplash | tee 16feat1k.txt
echo "matlabpool 8; nn(7494,3498,1,3,4,19); analyze" | matlab -nodisplay -nosplash | tee 16feat3k.txt
echo "matlabpool 8; nn(7494,3498,1,5,4,19); analyze" | matlab -nodisplay -nosplash | tee 16feat5k.txt
echo "matlabpool 8; nn(7494,3498,1,15,4,19); analyze" | matlab -nodisplay -nosplash | tee 16feat15k.txt

echo "matlabpool 8; nn(7494,3498,0,1,1,3); analyze" | matlab -nodisplay -nosplash | tee 3feat1k.txt
echo "matlabpool 8; nn(7494,3498,1,3,1,3); analyze" | matlab -nodisplay -nosplash | tee 3feat3k.txt
echo "matlabpool 8; nn(7494,3498,1,5,1,3); analyze" | matlab -nodisplay -nosplash | tee 3feat5k.txt
echo "matlabpool 8; nn(7494,3498,1,15,1,3); analyze" | matlab -nodisplay -nosplash | tee 3feat15k.txt