function threshold

  %threshold should be 0.25
  t1 = myThreshold (0.5, 1, 0, 1);

  % Let's see what Matlab computes and compare with this:
  t2 = solve( '1 / (sqrt(2*pi) * 1) * exp( -(x-0)^2 / (1^2) ) = 1 / (sqrt(2*pi) * 1) * exp( -(x-0.5)^2 / (1^2) )');
  %solve( 'x-2 = -x' ) %should meet at x=1

  fprintf('Our result: %d\n', t1);
  fprintf('Matlab result: %d\n', double(t2));

end


function t = myThreshold(m1, s1, m2, s2)

  % Calculate threshold
  % Attention: Not definied for s1=s2. If s1=s2 there is another formula (not implemented here)
  % TODO: Use the simplified formula in this case
  if s1==s2
    s2 = s2 + 0.00001;
  end

  % There is only one point of intersection.
  % Normalize, so m1 is always left from m2
  fact = -1;
  if m2 > m1
    fact = 1;
  end
  
  t = 2*(m1*s2^2 - m2*s1^2)/(2*(s2^2 - s1^2));
  t = t + fact ...
    * sqrt( (2*(m2*s1^2 - m1*s2^2)/(2*(s2^2 - s1^2)))^2 - ...
      (m1^2*s2^2 - m2^2*s1^2 - ...
        2*log(s2/s1)*s1^2*s2^2)/(s2^2 - s1^2) );

end
