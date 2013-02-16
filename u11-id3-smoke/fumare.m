% IDEAs
% - Detect moving camera: Camera is moving for example 2
%   and also for example 1 between picture 2 and 3
%   So we recognize diffs where there are no diffs
%   See "Registraion-noise reduction in difference images for change detection"
%
% - Analyze color: Smoke is grey.
%
% - Analyze "entropy" or level of detail for an area: Should be low for smoke
%
% - Smoke source does not change it's place, so should be on both diff images
%
% - Growing region detection? (see other paper)
%
% - ...

% IDEAS from last week brainstorming:
% - Wir können zwei Differenzbilder erzeugen: 1>2 und 2>3 (und 1>3)
% - Rauchquelle verlagert Standort nicht: Auf allen Differenbildern ist an gleicher Stelle %   Differenz. Wolken verlagern ihren Standort.
% - Schatten ist nur Skalierung der Helligkeit, Rauch verdeckt den Hintergrund.
%  - Alle Differenzen ausschließen, bei denen sich nur Helligkeit verändert hat
%  - Schauen, wo Struktur verloren geht wenn es heller wird (wo nicht nur Helligkeit
%   skaliert wird, oder wo keine Struktur entsteht)
% - Rauch ist hell
% - Rauch ist wie filter, macht alles unscharf, weniger Entropie. Bei Sonnenschein viel
%   Informationen.
% - Idee: Zauberstab, zusammenhängende sich ändernde Flächen
% - Differenzmaß entwickeln, dass Rauch als stärkere Differenz zurückliefert.
% - Fehlermaß: False positives sind erlaubt. False negatives haben großen Fehler

% http://www.dti.unimi.it/genovese/wild/wildfire.htm
% http://www.firesense.eu/
% http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=5487172&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D5487172
% http://wildfire.fesb.hr/

function fumare()
  % Will be read as uint16
  A=imread('png/1-1.png');
  B=imread('png/1-2.png');
  C=imread('png/1-3.png');
  compare(A,B,C, '1');
  
  A=imread('png/2-1.png');
  B=imread('png/2-2.png');
  C=imread('png/2-3.png');
  compare(A,B,C, '2');
  
  A=imread('png/3-1.png');
  B=imread('png/3-2.png');
  C=imread('png/3-3.png');
  compare(A,B,C, '3');
  
end

function compare(A, B, C, name)
  % 0=black, 1=white
  bits = 8;
  A=int16(normalize(A,bits)); % To be able to handle negative diff
  B=int16(normalize(B,bits));
  C=int16(normalize(C,bits));
  D1 = B - A;
  D2 = C - B;
  D12 = D2 - D1;
  D1 = normalize(D1, bits);
  D2 = normalize(D2, bits);
  D12 = normalize(D12, bits);
  
  saveim(D1, [name '-d1']);
  saveim(D2, [name '-d2']);
  saveim(D12, [name '-d12']);
end

function saveim(X, name)
  X = abs(X);
  X = uint8(X);
  imdebug(X);
  Xf = noiseThreshold(X, 8, 0.15);
  Xc = colorize(Xf);
  Xsmall = scale_image(X, 0.5);
  %imwrite(X, [name '.png'],'bitdepth',8);
  %imwrite(Xf, [name '-f.png'],'bitdepth',8);
  imwrite(Xc, [name '-c.png']);
  %imwrite(Xsmall, [name '-small.png'],'bitdepth',8);
end

function R = colorize(X)
  %cmap = gray;    % default
  cmap=jet;
  R = ind2rgb(X, cmap);
end

function imdebug(X)
  %mean(mean(X));
  %mean(std(double(X)));
  %X(100:110,100:110);
end

% Filter out small diffs
% Threshold 0.5 would be: Everything lower than 0.5*white is filtered away
function X = noiseThreshold(X, bits, threshold)
  X(X<=(threshold*(2^bits))) = 0;
end

% Use full spectrum of greyscale (to make it brigth)
function R = normalize(X, bits)
  % default Bitdepth=8 for greyscale. However we use more than 8 bit/256 values
  limit = 2^bits - 1;
  factor = limit / double(max(max(X)));
  R = X .* factor;
end

% Copied from http://stackoverflow.com/questions/6183155/resizing-an-image-in-matlab
function pic_new = scale_image(pic,scale_zoom)

  oldSize = size(pic);                               %# Old image size
  newSize = max(floor(scale_zoom.*oldSize(1:2)),1);  %# New image size
  newX = ((1:newSize(2))-0.5)./scale_zoom+0.5;  %# New image pixel X coordinates
  newY = ((1:newSize(1))-0.5)./scale_zoom+0.5;  %# New image pixel Y coordinates
  oldClass = class(pic);  %# Original image type
  pic = double(pic);      %# Convert image to double precision for interpolation

  if numel(oldSize) == 2  %# Interpolate grayscale image

    pic_new = interp2(pic,newX,newY(:),'cubic');

  else                    %# Interpolate RGB image

    pic_new = zeros([newSize 3]);  %# Initialize new image
    pic_new(:,:,1) = interp2(pic(:,:,1),newX,newY(:),'cubic');  %# Red plane
    pic_new(:,:,2) = interp2(pic(:,:,2),newX,newY(:),'cubic');  %# Green plane
    pic_new(:,:,3) = interp2(pic(:,:,3),newX,newY(:),'cubic');  %# Blue plane

  end

  pic_new = cast(pic_new,oldClass);  %# Convert back to original image type

end
