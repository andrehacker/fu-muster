function [W H] = nmf(V, w_size, num_of_iterations)
  % random start values
  W = rand(size(V,1),w_size);
  H = rand(size(W,2),size(V,2));

  % repmat magic utility
  H_rep_index = reshape(repmat([1:size(H,1)],size(V,1),1),1,[]);
  V_rep_index = reshape(repmat([1:size(V,2)],size(W,2),1),1,[]);
  
  for i = 1:num_of_iterations
    [i num_of_iterations]
    % compute current prediction
    V_dash = W * H;
    % compute quotien between current prediction and actual image
    V_by_V = V./V_dash;

    % repmat magic utility
    V_by_V_dash = repmat(sum(V_by_V,2),1,size(W,2));
    
    % recompute W
    tmp1 = repmat(V_by_V,size(H,1),1);
    tmp2 = H(H_rep_index,:);
    big_sum = reshape(sum(tmp1 .* tmp2,2),size(W));
    W = W .* big_sum;
    
    % normalize W
    sigma_W = repmat(sum(W,1),size(W,1),1);
    W = W./sigma_W;

    % repmat magic
    tmp1 = repmat(W,1,size(V,2));
    tmp2 = V_by_V(:,V_rep_index);
    big_sum = reshape(sum(tmp1 .* tmp2,1),size(H));

    % recompute H
    H = H .* big_sum;
  end

end



function [W H] = nmf2(V, w_size, num_of_iterations)
  W = rand(size(V,1),w_size);
  H = rand(size(W,2),size(V,2));
  
  for k = 1:num_of_iterations
    V_dash = W * H;
    V_by_V = V./V_dash;
    factor = zeros(size(W));
    for i = [1:size(W,1)]
      for a = [1:size(W,2)]
        for mu = [1:size(H,2)]
          factor(i,a) = factor(i,a) + (V_by_V(i,mu) * H(a,mu));
        end
      end
    end
    W = W .* factor;
    
    sigma_W = repmat(sum(W,1),size(W,1),1);
    W = W./sigma_W;

    factor = zeros(size(H));
    for a = [1:size(W,2)]
      for mu = [1:size(H,2)]
        for i = [1:size(W,1)]
          factor(a,mu) = factor(a,mu) + (W(i,a) * V_by_V(i,mu));
        end
      end
    end
    H = H .* factor;
  end
end