function y = features2class(x,classification_data)

model = classification_data;

y = model.predict(transpose(x));