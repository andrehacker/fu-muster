function mitte 
  tra = load('pendigits.tra');
  
  f = 'mean_digits.mat';
  A = [];
  for n = 0:9
    A = [A ; mean(filterByLast(tra, n)) n];
  end
  dlmwrite(f, A, ' ');
end

function r = filterByLast(data, n)
  r = data(ismember(data(:,end),n),1:end-1);
end

function [m c] = meanAndCov(data)
  m = mean(data);
  c = cov(data);
  c = c + (0.0001 * eye(size(c)));
end