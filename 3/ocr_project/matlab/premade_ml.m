load('../../ocrfeaturestrain');
X = transpose(X);
Y = transpose(Y);

letters_to_compare = [['p' 't'];
                      ['o' 'q']
                      ['i' 'l']
                      ['w' 'm']
                      ['y' 'v']
                      ['j' 'l']
                      ['e' 'f']
                      ];
n_pairs = size(letters_to_compare,1);

A = [];
for i= 1:n_pairs
    l1 = letters_to_compare(i,1);
    l2 = letters_to_compare(i,2);
    A = [A; premade_train_and_predict(X, Y, l1, l2)];
end;

A
mean = sum(A, 1) / n_pairs