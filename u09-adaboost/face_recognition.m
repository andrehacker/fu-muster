function face_recognition()
  raw  = load('faces_vj.mat','faces');
  rawFaces = single(raw.faces);
  iiFaces = integral_image(rawFaces);

  raw = load('nonfaces_vj.mat');
  rawNonFaces = single(raw.nonfaces);
  iiNonFaces = integral_image(rawNonFaces);
  
  iiAll = cat(3,iiFaces,iiNonFaces);
  y = repmat(1,size(iiFaces,3),1);
  y = [y ; repmat(0,size(iiNonFaces,3),1)];

  w = repmat(1/(2*size(iiFaces,3)),size(iiFaces,3),1);
  w = [w ; repmat(1/(2*size(iiNonFaces,3)),size(iiNonFaces,3),1)];

  for t = [1:1]
    w = w ./ sum(w);
  end
  rf = rawFaces(:,:,1);
  rf(1:10,1:10)
  train_weak_classifier(iiFaces,iiNonFaces,y,w);
end

function r = train_weak_classifier(faces,nonfaces, y, w)
  images = cat(3,faces,nonfaces);
  w = repmat(1/(2*size(faces,3)),size(faces,3),1);
  w = [w ; repmat(1/(2*size(nonfaces,3)),size(nonfaces,3),1)];
  tmp = load('classifier_A','classifier_A');
  cA = tmp.classifier_A;
  classifier_weighted_error = repmat(1,size(cA,1),1);
  size(classifier_weighted_error)
  for i =  [1:1]%size(cA,1)]
    results = zeros(size(images,3),1);
    for j = [1:size(images,3)]
      results(j) = weak_A(cA(i,1),cA(i,2),cA(i,3),cA(i,4),images(:,:, ...
						  j));
    end
    tmp0 = repmat(w,1,size(results));
    tmp1 = repmat(results,1,size(results));
    tmp2 = repmat(results',size(results),1);
    tmp3 = repmat(y,1,size(results));
    tmp4 = (tmp1 > tmp2);
    tmp5 = abs(tmp4 - tmp3).*tmp0;
    tmp6 = sum(tmp5,1);
    [val index] = min(tmp6');
    cA(i,5) = results(index);
    classifier_weighted_error(i) = val;
  end
  classifier_weighted_error(1:10)

  %tmp4 = ((tmp1 > tmp2) == tmp3);
  %[val index] = max(sum(tmp4,1)')
  %threshold = results(index)
  %threshold = mean(results)
  

end

function r = weak_A(x1, y1, x2, y2, img)
  img = [repmat(0,1,size(img,2)) ; img];
  img = [repmat(0,1,size(img,1))'  img];
  x1 = x1   + 1;
  y1 = y1   + 1;
  x2 = x2   + 1;
  y2 = y2   + 1;
  x3 = ceil((x2-x1)/2)+1;
  A = ii_area(x3+1,y1,x2,y2,img);
  B = ii_area(x1,y1,x3,y2,img);
  r = A-B;
end

function r = weak_C(x1, y1, x2, y2, img)
  img = [repmat(0,1,size(img,2)) ; img];
  img = [repmat(0,1,size(img,1))'  img];
  x3 = x2/3 + 1;
  x4 = (x2/3)*2 + 1;
  x1 = x1   + 1;
  y1 = y1   + 1;
  x2 = x2   + 1;
  y2 = y2   + 1;
  A = ii_area(x1,y1,x3,y2,img);
  B = ii_area(x3+1,y1,x4,y2,img)*2;
  C = ii_area(x4+1,y1,x2,y2,img);
  r = A-B;
end

function r = ii_area(x1,y1,x2,y2,img)
  r = img(y2,x2) + img(y1-1,x1-1) - (img(y2,x1-1) + img(y1-1,x2));
end

function ii = integral_image(images)
  ii = repmat(0,size(images));
  for i = [1:size(images,3)]
    ii(:,:,i) = cumsum(cumsum(images(:,:,i),2),1);
  end
end