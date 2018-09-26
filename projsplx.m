function x = projsplx(y)
% project an n-dim vector y to the simplex Dn
% Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}

% (c) Xiaojing Ye
% xyex19@gmail.com
%
% Algorithm is explained as in the linked document
% http://arxiv.org/abs/1101.6081
% or
% http://ufdc.ufl.edu/IR00000353/
%
% Jan. 14, 2011.

m = length(y); bget = false;

s = sort(y,'descend'); tmpsum = 0;

for ii = 1:m-1
    tmpsum = tmpsum + s(ii);
    tmax = (tmpsum - 1)/ii;
    if tmax >= s(ii+1)
        bget = true;
        break;
    end
end
    
if ~bget, tmax = (tmpsum + s(m) -1)/m; end

x = max(y-tmax,0);

return;

% function X = projsplx(Y)
% project an n-dim vector y to the simplex Dn
% Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}

% (c) Xiaojing Ye
% xyex19@gmail.com
%
% Algorithm is explained as in the linked document
% http://arxiv.org/abs/1101.6081
% or
% http://ufdc.ufl.edu/IR00000353/
%
% Jan. 14, 2011.
% [row,col]=size(Y);
% X = zeros(row,col);
% for c=1:col
%     y=Y(:,c);
%     m = length(y); bget = false;
%     
%     s = sort(y,'descend'); tmpsum = 0;
%     
%     for ii = 1:m-1
%         tmpsum = tmpsum + s(ii);
%         tmax = (tmpsum - 1)/ii;
%         if tmax >= s(ii+1)
%             bget = true;
%             break;
%         end
%     end
%     
%     if ~bget, tmax = (tmpsum + s(m) -1)/m; end
%     
%     X(:,c) = max(y-tmax,0);
% end
% return;