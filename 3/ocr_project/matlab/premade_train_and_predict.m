function [ accuracy ] = premade_train_and_predict(X, Y, letter1, letter2 )

letter_to_position = @(x) x - 'a' + 1;
letter1 = letter_to_position(letter1);
letter2 = letter_to_position(letter2);

l1_or_l2 = (Y == letter1 | Y == letter2);

X_filtered = X(l1_or_l2, :);
Y_filtered = Y(l1_or_l2, :);

n_data = size(X_filtered, 1);
part = cvpartition(n_data, 'HoldOut', 0.20);

x_train = X_filtered(part.training);
y_train = Y_filtered(part.training);

x_test = X_filtered(part.test);
y_test = Y_filtered(part.test);


tree = fitctree(x_train, y_train); % Regression tree classifier
SVMModel = fitcsvm(x_train, y_train); % Support Vector Machine
mdl = fitcknn(x_train, y_train); % Nearest Neighbour Classifier


tree_p = tree.predict(x_test);
SVM_p = SVMModel.predict(x_test);
mdl_p = mdl.predict(x_test);

n_test = size(x_test, 1);

tree_acc = sum(tree_p == y_test) / n_test;
SVM_acc = sum(SVM_p == y_test) / n_test;
mdl_acc = sum(mdl_p == y_test) / n_test;

accuracy = [tree_acc SVM_acc mdl_acc];

end

