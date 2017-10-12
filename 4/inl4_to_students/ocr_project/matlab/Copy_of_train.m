load('ocrsegments.mat')

X = cellfun(@segment2features, S ,'uniformoutput',false);
X = transpose(cell2mat(X));

classification_data = {X,y};
save('classification_data.mat', 'classification_data')

