function features = segment2features(input_img)
% features = segment2features(I)
features = zeros(6,1);

% Number of white pixels in the image
% I imagine this could separate for example I and B as B has more
% white pixels.
brightness = sum(sum(input_img));
features(1) = brightness / 10;

% Number of white pixels in the whitest column
% Could detect if the letter countains a long vertical line.
% Compare C vs P
brightest_col = max(sum(input_img));
features(2) = brightest_col;

% Number of white pixels in the whitest row
% Similar to the previous feature but horizontional.
% Compare T vs J
brightest_row = max(sum(input_img, 2));
features(3) = brightest_row;

% Number of segments in the inverted image
% Some letters such as A or B encloses part of the image within
% themselves that isn't a part of letter pixels. By inverting the
% image and running the bwlabel, we can detetc how many such
% segments there are. J has 0, A has 1, B has two enclosed segments.
compl_img = imcomplement(input_img);
[labels, number_seg] = bwlabel(compl_img, 8);
features(4) = number_seg * 10;


% Number of "feet"
% Taking the first row that countains a white pixel, starting from
% the bottom, and check how many segments that is.
% Compare A vs B

% Feet width
% Similar to the previous feature I check the bottom row that contains
% a white pixel. I then take the sum of that row.
% Compare L vs C
nbr_feet = 0;
feet_length = 0;
for index_seg = size(input_img,1): -1 : 1
    if sum(input_img(index_seg,:)) ~= 0
        [segment_label, nbr_segments] = bwlabel(input_img(index_seg,:), 4);
        nbr_feet = nbr_segments;
        feet_length = sum(input_img(index_seg,:));
        break;
    end
end
features(5) = nbr_feet * 10;
features(6) = feet_length;