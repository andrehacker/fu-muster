function Handles = imshowTruesize(imagesCellArray, margins, alignment)
% IMSHOWTRUESIZE plots series of images in a single figure while preserving the actual 
% aspect ratio or scale of the images (similar to the functionality of truesize but in
% a multiple images situation). 
%
% the images can be aligned to the left of the figure, to the top or centered, as desired.
% Using the returned handles the user can set properties for the plots such as title,
% axis, etc.
% This function comes in handy when you want to plot images with different sizes in one
% figure and you want to preserve the aspect ratio of the images. This is not the case
% with imshow since imshow changes the scales of the images.
%
%   Note: figures bigger than the screen size are allowed and could be created without
%   their sizes being reduced. Since this function's goal is to present the images in
%   true size this behavior is regarded as appropriate by the author. Please note that in
%   these extreme cases the figure position and size would be awkward. I advise to avoid
%   these situations and reduce the number of images or their sizes. 
%
% Inputs:
%   imagesCellArray - cell array of images to plot, [n by m]
%
%   alignment - string, either 'left' or 'top' (or any subset such as 'col') 
%               setting the alignment of the images in the figure.
%               default: 'center'
%
%   margins - scalar or array with width and height dimensions of the margins. 
%             if scarlar than the value for width and height would be set equal.
%             Units in pixels!
%             default: [10 10]
%
% Outputs: 
%   Handels - structure with handles for the plotted figure and subplots, with fields:
%      Handels.hFigure - handle to the figure
%      Handels.hSubplot - matrix of handles to the subplots
%
% Example usage:
%   % compare imshow vs. imshoTruesize
%   % images of Mandelbrot set in differenct scales
%   mand = imread('mandelbrot_set.jpg');
%   dim = 3;
%   clear imagesCellArray
%   [imagesCellArray{1:dim,1:dim}] = deal(mand);
%   for iRow = 1:dim
%      for iCol = 1:dim
%      imagesCellArray{iRow,iCol} = imresize(imagesCellArray{iRow,iCol},1/(1.5*(iCol*iRow)));
%      end
%   end
%   % plot with imshow - true aspect ration is lost
%   figure
%   for iRow = 1:dim
%      for iCol = 1:dim
%         subplot(dim,dim,sub2ind([dim,dim],iRow,iCol))
%         imshow(imagesCellArray{iRow,iCol})
%         axis on
%      end
%   end
%   % plot with imshowTruesize - true aspect ration is preserved
%   margins = [25 25];
%   Handles = imshowTruesize(imagesCellArray,margins);
%   for iRow = 1:dim
%      for iCol = 1:dim
%         axis(Handles.hSubplot(iRow,iCol),'on')
%      end
%   end
%
% Revision history:
%   2011-07-05, Adi N., Creation


%% validate inputs and set defaults
error(nargchk(1, 3, nargin)) % nargin check

if ~exist('margins','var') || isempty(margins) % default margins
   margins = [10 10];
elseif isequal(size(margins,2),1) % scalar
   margins = margins([1,1]);
elseif size(margins,2) > 2
   error('Error in imshowTruesize.m: invalid size for margins')
end

if ~exist('alignment','var') % default alignment
   alignment = 'center';
else % validate alignment string
   validAlignmentStr = {'left','top','center'};
   alignment = validatestring(alignment,validAlignmentStr,'imshowTruesize','alignment');
end


%% initialize variables
[mRowsImgs,nColsImgs] = size(imagesCellArray);
heightMat = zeros(mRowsImgs,nColsImgs);
widthMat = zeros(mRowsImgs,nColsImgs);

%% loop over images and get dimensions of images
for iRow = 1:mRowsImgs % loop through left
   for iCol = 1:nColsImgs % loop through top
      [mRowsImg, nColsImg, ~] = size(imagesCellArray{iRow,iCol}); % dimensions of the i_th image
      thisImgWidth = nColsImg;
      thisImgHeight = mRowsImg;
      % fill matrix with dimensions for following 'for' loop
      widthMat(iRow,iCol) = thisImgWidth; % allocate image width to structure
      heightMat(iRow,iCol) = thisImgHeight; % allocate image height to structure
   end
end


%% calculate the total maximal dimensions of the given images
rowHeight = max(heightMat,[],2);  % heighest image in each row
colWidth = max(widthMat,[],1);    % widest image in each column

switch alignment
   case 'left'
      % in left alignment the total maximal width is the maximum of the sum of images'
      % width in columns
      imgWidthTotalMax = max(sum(widthMat,2));
      % in left alignment the total maximal height is the sum of the maximum of images'
      % height in rows
      imgHeightTotalMax = sum(rowHeight);
   case 'top'
      % in top alignment the total maximal width is the sum of the maximum of images'
      % width in columns
      imgWidthTotalMax = sum(colWidth);
      % in top alignment the total maximal height is the maximum of the sum of images'
      % height in rows
      imgHeightTotalMax = max(sum(heightMat,1));
   case 'center'
      % in center alignment the total maximal width is the sum of the maximum of images'
      % width in columns
      imgWidthTotalMax = sum(colWidth);
      % in center alignment the total maximal height is the sum of the maximum of images'
      % height in rows
      imgHeightTotalMax = sum(rowHeight);
end


%% set figure dimensions in pixels including margins
fWidth = imgWidthTotalMax + (nColsImgs+1)*margins(1);
fHeight = imgHeightTotalMax + (mRowsImgs+1)*margins(2);

% open figure, positioned in the center of the screen
if nargout == 0 % just plot
   screenSize = get(0,'screensize');
   scrWidth = screenSize(3);
   scrHeight = screenSize(4);
   figPosition = [((scrWidth-fWidth)/2)/scrWidth, ((scrHeight-fHeight)/2)/scrHeight,...
                  fWidth/scrWidth, fHeight/scrHeight];
   figure('Units','Normalized','Position',figPosition);  % by default the position units are pixels
else % plot and return handles structure to figure and subplots
   screenSize = get(0,'screensize');
   scrWidth = screenSize(3);
   scrHeight = screenSize(4);
   figPosition = [((scrWidth-fWidth)/2)/scrWidth, ((scrHeight-fHeight)/2)/scrHeight,...
                  fWidth/scrWidth, fHeight/scrHeight];
   Handles.hFig = figure('Units','Normalized','Position',figPosition);  % by default the position units are pixels
end

%% plot images
switch alignment
   % for left alignment the inner 'for' loop is on the columns
   case 'left'
      for iRow = 1:mRowsImgs % loop through left
         % initialize left position of subplot
         leftPos = margins(1);
         for iCol = 1:nColsImgs % loop through top
            % set position so to align images equally centered on a left line
            % aggregated bottom position - subtract one margin, half of row height and
            % half of current image's height
            % position in normalized units in [left bottom width height]
            rowBottomPos = fHeight - iRow*margins(2) - sum(rowHeight(1:iRow));
            bottomPos = rowBottomPos + rowHeight(iRow)/2 - heightMat(iRow,iCol)/2;
            posLeft   = leftPos / fWidth;
            posBottom = bottomPos / fHeight;
            posWidth  = widthMat(iRow,iCol) / fWidth;
            posHeight = heightMat(iRow,iCol) / fHeight;
            position  = [posLeft, posBottom, posWidth, posHeight];
            % subplot with the current position
            Handles.hSubplot(iRow,iCol) = subplot('Position',position);
            % update left position for next plot
            leftPos = leftPos + margins(1) + widthMat(iRow,iCol);
            
            imshow(imagesCellArray{iRow,iCol});
         end % iCol
      end % iRow
      % for top alignment the inner 'for' loop is on the rows
   case 'top'
      for iCol = 1:nColsImgs % loop through top
         % initialize bottom position of subplot
         bottomPos = fHeight;
         for iRow = 1:mRowsImgs % loop through left
            % set position so to align images equally centered on a top line.
            % aggregated bottom position - subtract one margin and current image's height.
            % position in normalized units: [left bottom width height]
            
            colRightPos = iCol*margins(1) + sum(colWidth(1:iCol));
            bottomPos = bottomPos - margins(2) - heightMat(iRow,iCol);
            leftPos = colRightPos - colWidth(iCol)/2 - widthMat(iRow,iCol)/2; 
            posLeft   = leftPos / fWidth;
            posBottom = bottomPos / fHeight;
            posWidth  = widthMat(iRow,iCol) / fWidth;
            posHeight = heightMat(iRow,iCol) / fHeight;
            position  = [posLeft, posBottom, posWidth, posHeight];
            % subplot with the current position
            Handles.hSubplot(iRow,iCol) = subplot('Position',position); %nImages,mColsSubplot,iImage,'Units','Pixels','Position',position);
            
            imshow(imagesCellArray{iRow,iCol});
         end % iRow
      end % iCol
      
   case 'center'
      for iCol = 1:nColsImgs % loop through columns
         for iRow = 1:mRowsImgs % loop through rows
            rowBottomPos = fHeight - iRow*margins(2) - sum(rowHeight(1:iRow));
            bottomPos = rowBottomPos + rowHeight(iRow)/2 - heightMat(iRow,iCol)/2;
            colRightPos = iCol*margins(1) + sum(colWidth(1:iCol));
            leftPos = colRightPos - colWidth(iCol)/2 - widthMat(iRow,iCol)/2;
            posLeft   = leftPos / fWidth;
            posBottom = bottomPos / fHeight;
            posWidth  = widthMat(iRow,iCol) / fWidth;
            posHeight = heightMat(iRow,iCol) / fHeight;
            position  = [posLeft, posBottom, posWidth, posHeight];
            % subplot with the current position
            Handles.hSubplot(iRow,iCol) = subplot('Position',position); %nImages,mColsSubplot,iImage,'Units','Pixels','Position',position);
            
            imshow(imagesCellArray{iRow,iCol});
         end
      end
end % switch alignment