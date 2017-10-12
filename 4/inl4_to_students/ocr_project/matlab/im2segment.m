function [S] = im2segment(im)
% [S] = im2segment(im)

nrofsegments = 5; % max nbr of segments
m = size(im,1); % nbr y px
n = size(im,2); % nbr x px
S = cell(1,nrofsegments);

% converts the image to a binirized image i.e. each pixel is either
% 0 or 1. Pixels who's value is above the specified threshold becomes 1
binary_img = imbinarize(im, 135);

% Inverts the image. Black becomes white
compl_img = imcomplement(binary_img);

% Segments the image based on their connectivity. Each pixel of a
% segment get's set to it's corresponding segment.
% Either the 4 or 8 neighbours can be used to determine connectivity 


L2 = bwareafilt(compl_img, nrofsegments);
L = bwlabel(L2, 8);

% Loops over all the segments
for kk = 1:nrofsegments
    
    % Creates a black image the size of the original one
    segment_img = zeros(m,n);
    
    % Finds the pixel coordinates corresponding to a segment
    [r, c] = find(L==kk);
    
    % Colors all the pixels in a segment white
    for i = 1:size(r)
        segment_img(r(i), c(i)) = 1;
    end;
    
    segment_img2 = imbinarize(segment_img, 0.1);
    % Set the final image
    S{kk} = segment_img2;
end;
