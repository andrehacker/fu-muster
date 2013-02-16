
resultfile = 'recognition_results.mat';

isVsShould = dlmread(resultfile);
%load (resultfile);

hit = 0;
miss = 0;

misses = zeros(10,10);
hits = zeros(1,10);

fid = fopen('analysis_results.txt', 'w');
fprintf(fid, '\n** RESULTS **\n\n');

for k = 1:size(isVsShould, 1)
  if isVsShould(k,1) == isVsShould(k,2)
    hit = hit +1;
    hits(isVsShould(k,2)+1) = hits(isVsShould(k,2)+1) + 1;
  else
    miss = miss + 1;
    misses(isVsShould(k,1)+1,isVsShould(k,2)+1) = misses(isVsShould(k,1)+1,isVsShould(k,2)+1) + 1;
  end;
end;


for k=1:10
  fprintf(fid,'hits for %d: %d\n', k-1, hits(k));  % octave is 0-bounded. This makes it ugly here...
end;

% prediction is left, real value is right!
for j=1:10
  for i=1:10
    if misses(i,j) ~= 0
      fprintf(fid,'Number %d, missclassified as %d: %d\n', j-1, i-1, misses(i,j));
    end;
  end;
end;

fprintf(fid,'\n');
fprintf(fid,'Total classifications: %d\n', hit+miss);
fprintf(fid,'Total hits: %d\n', hit);
fprintf(fid,'Total misses: %d\n', miss);
fprintf(fid,'ERROR RATE: %.3f\n', miss/(hit+miss));
fprintf(fid,'SUCCESS RATE: %.3f\n\n', 1-(miss/(hit+miss)));
fclose(fid);