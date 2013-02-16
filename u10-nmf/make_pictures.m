function make_pictures()
  subset = [1:8]+20;

  W = load('w1000_20.mat'); W = W.W;
  H = load('h1000_20.mat'); H = H.H;  
  tes_p = load('usps.ascii/test_patterns.txt');
  tra_p = load('usps.ascii/train_patterns.txt'); 

  visualize_simple(tra_p(:,subset),'digits');

  visualize_simple(W,'codebook');

  visualize_simple(W * H(:,subset),'reconst');

  H_tes = ((W'*W)^-1)*W'*tes_p(:,subset);

  visualize_simple(W*H_tes,'H_tes');
  
  visualize_simple(tes_p(:,subset),'tes_original');

end


function visualize_simple(W,filename)
  for i = 1:size(W,2)
    h = figure('visible','off'); 
    small_img = reshape(W(:,i),16,16)';
    imagesc(small_img);
    colormap(gray);
    num = sprintf('%d',i);
    print(h,'-deps',[filename num  '.eps']);
  end
end