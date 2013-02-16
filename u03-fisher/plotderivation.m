function plotderivation()
  [X, Y] = meshgrid(-2:0.2:2);
  plotfigure(X, Y, @f1, 'task3-f1');
  plotfigure(X, Y, @f1derivatedx, 'task3-f1-derivatedX');
  plotfigure(X, Y, @f1derivatedy, 'task3-f1-derivatedY');
  plotwithquiver(X, Y, @f1, @f1derivatedx, @f1derivatedy, 'task3-f1-quiver');

  plotfigure(X, Y, @f2, 'task3-f2');
  plotfigure(X, Y, @f2derivatedx, 'task3-f2-derivatedX');
  plotfigure(X, Y, @f2derivatedy, 'task3-f2-derivatedY');
  plotwithquiver(X, Y, @f2, @f2derivatedx, @f2derivatedy, 'task3-f2-quiver');
end

function plotfigure(X, Y, fun, name)
  Z = fun(X,Y);
  h = figure;
  surfc(X, Y, Z);
  print(h,'-dpng',[name '.png']);
end

function plotwithquiver(X, Y, fun, derix, deriy, name)
  Z = fun(X,Y);
  DX = derix(X,Y);  % same as gradient(Z)!
  DY = deriy(X,Y);
  h = figure;
  contour(X, Y, Z);
  hold on;
  quiver(X, Y, DX, DY);
  print(h,'-dpng',[name '.png']);
end

% --------------------
% FUNCTIONS TO PLOT
% --------------------
function r = f1(x, y)
  r = -4*(x.^2+y.^2);
end

function r = f1derivatedx(x, y)
  r = -8*x;
end

function r = f1derivatedy(x, y)
  r = -8*y;
end

function r = f2(x, y)
  r = exp(1/3*( -4*x.^2 + 6*x -4*y.^2 - 6*y + 4*x.*y - 3 ));
end

function r = f2derivatedx(x, y)
  r = exp(1/3 *( -4*x.^2 + 6*x -4*y.^2 - 6*y + 4*x.*y - 3 ));
  r = r * 1/3 .* (-8*x + 4*y + 6);
end

function r = f2derivatedy(x, y)
  r = exp(1/3*( -4*x.^2 + 6*x -4*y.^2 - 6*y + 4*x.*y - 3 ));
  r = r * 1/3 .* (-8*y + 4*x - 6);
end


% Links & Comments
% http://www.mathworks.de/de/help/matlab/visualize/representing-a-matrix-as-a-surface.html
% Surface plots are useful for visualizing matrices that are too large to display in numerical form and for graphing functions of two variables.
% 
% Visualizing functions of 2 variables:
% http://www.mathworks.de/de/help/matlab/visualize/representing-a-matrix-as-a-surface.html#f0-5208
