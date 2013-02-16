resultfile = 'recognition_results.mat';

isVsShould = dlmread(resultfile);

confm = zeros(10,10);

for i = 1:size(isVsShould,1)
  y = isVsShould(i,1)+1;
  x = isVsShould(i,2)+1;
  confm(x, y) = confm(x, y) + 1;
end
confm = [0 1 2 3 4 5 6 7 8 9 ; confm];
confm = [[NaN 0 1 2 3 4 5 6 7 8 9]'  confm];
confm; % horizontal: predicted, vertical: actual
dlmwrite('confusion_matrix.mat', confm, '\t');