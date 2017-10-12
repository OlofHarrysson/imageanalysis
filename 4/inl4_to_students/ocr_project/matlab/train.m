load('ocrsegments.mat')

X = cellfun(@segment2features, S ,'uniformoutput',false);
X = transpose(cell2mat(X));

% mdl = fitcecoc(X, y); % multiclass SVM max 10% 4 minutes
% mdl = fitctree(X, y); % Tree max 20%

% mdl = fitcensemble(X, y); % ensemble max 43%

% mdl = fitcknn(X, y); % Nearest Neighbour Classifier max 69%
% mdl = fitcknn(X, y, 'Distance', 'euclidean', 'NumNeighbors', 1, 'Standardize',1, 'DistanceWeight', 'inverse'); % Nearest Neighbour Classifier max 69%
% mdl = fitcknn(X, y, 'NSMethod', 'kdtree', 'OptimizeHyperparameters', 'all');


mdl = fitcknn(X, y, 'Standardize', 1);



% 5 -> 35%, 10 -> 35%, 3 -> 34%
% KNN
% t = templateKNN('NumNeighbors',3,'Standardize',1);

% 'DiscrimType','pseudoLinear' -> 36%, 
% Discri
% t = templateDiscriminant('DiscrimType','pseudoLinear')


% mdl = fitcensemble(X, y, 'Learners', t); % ensemble

classification_data = mdl;
save('classification_data.mat', 'classification_data')