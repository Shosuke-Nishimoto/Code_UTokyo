function prediction = classify_image(filepass)

% �l�b�g���[�N�̌Ăяo��
net = load('image_task_cnn.mat');
data = imageDatastore(filepass);
prediction = classify(net.myNet,data);
prediction = char(prediction);

end