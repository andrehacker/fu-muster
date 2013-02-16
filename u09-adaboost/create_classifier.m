function r = create_classifier(max_x, max_y)
 classifier_A = create_classifier_A(max_x, max_y);
 save('classifier_A.mat','classifier_A');
end

function r = create_classifier_A(max_x, max_y)
  r = [];
  for y1 = [1:max_y]
    for x1 = [1:max_x]
      for y2 = [y1+1:max_y]
	for x2 = [x1+1:2:max_x]
	  r = [r ; x1 y1 x2 y2 0];
	end
      end
    end
  end
end