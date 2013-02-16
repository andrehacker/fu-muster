% Same as linearNNSearch, but k-Nearest Neighbor search
% Uses the simple approach to sort the result and take first k
% TODO: Add weighting (closed neighbors are weighted more).
% E.g. give each neighbor the weigth 1/d (according to Wikipedia)
function result = linearKNNSearch(k, database, database_nums, new_pattern)
  n = size(database, 1);

  % calculate distances to all test-objects.
  % Store distance (left) and id (to get class after sorting) (right)
  dist = [zeros(1,n); 1:n]';
  for i = 1:n
    dist(i, 1) = sqDistance(database(i,:)', new_pattern');
    %dist(i, 1) = distancePoints(database(i,:), new_pattern);
  end;
  firstk = sortrows(dist, 1); %sort by first column, take first k rows
  firstk = firstk(1:k,:);

  % count the frequency of the classes for the k nearest neighbors
  classFrequency = zeros(10,1); % what I really want is 0 to 9;-(
  for i = 1:k
    classFrequency(database_nums(firstk(i,2))+1) = classFrequency(database_nums(firstk(i,2))+1) + 1;  % attention, offset +1!
  end;

  % take the most frequent one. Attention: We have an offset (starting with class 1 instead of 0)
  [~, index] = max(classFrequency);
  result = index - 1;    % minimum returns index as 2nd argument

end

function D = sqDistance(X, Y)
  D = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
end