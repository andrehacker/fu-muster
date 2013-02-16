% Implements Linear Nearest Neighbor Search for 1 new item
% n test items, f features
% database = (n x f) matrix. Feature space of test data
% database_nums = (n x 1) matrix. Classes of test data
% new_pattern = (1 x f) feature vector of obect to be classified
% return value: predicted class
function result = linearNNSearch(database, database_nums, new_pattern)
  min = inf;
  index = 1;
  for i = 1:size(database, 1)
    d = sqDistance(database(i,:)', new_pattern');
    %d = distancePoints(database(i,:), new_pattern);
    if d < min
      min = d;
      index = i;
    end;
  end;
  result = database_nums(index);
end

