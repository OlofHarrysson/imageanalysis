function y = features2class(x,classification_data)
x_train = cell2mat(classification_data(1));
y_train = cell2mat(classification_data(2));

mdl = fitcknn(x_train, y_train); % Nearest Neighbour Classifier

y = mdl.predict(transpose(x));