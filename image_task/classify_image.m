function prediction = classify_image(filepass)

% ネットワークの呼び出し
net = load('image_task_cnn.mat');
data = imageDatastore(filepass);
prediction = classify(net.myNet,data);
prediction = char(prediction);

end