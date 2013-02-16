
% Execute this script file to analyze the results produced by nn.m

resultfile = 'recognition_results.mat';

load (resultfile);

hit = 0;
miss = 0;

misses = zeros(10,10);
hits = zeros(1,10);


fprintf('\n** RESULTS **\n\n');

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
  fprintf('hits for %d: %d\n', k-1, hits(k));  % octave is 0-bounded. This makes it ugly here...
end;

% prediction is left, real value is right!
for j=1:10
  for i=1:10
    if misses(i,j) ~= 0
      fprintf('Number %d, missclassified as %d: %d\n', j-1, i-1, misses(i,j));
    end;
  end;
end;

fprintf('\n');
fprintf('Total classifications: %d\n', hit+miss);
fprintf('Total hits: %d\n', hit);
fprintf('Total misses: %d\n', miss);
fprintf('ERROR RATE: %.3f\n\n', miss/(hit+miss));
