function [myNet, predictedLabels, accuracy] = make_image_cnn(filepath)

% データをimageDatastoreとして持つ
allImages = imageDatastore(filepath,'LabelSource','foldernames');

% データを分割
[trainStore, testStore] = splitEachLabel(allImages, 0.8,'randomize');

% データの水増し
trainStore = shuffle(trainStore);
bootstrap_factor = 2;
alphabetical_labels = {'class1','class2','class3','class4','class5'};
labels = trainStore.Labels;
labelCounts = countEachLabel(trainStore);
labelCounts = labelCounts.Count;
weights = labelCounts/sum(labelCounts);
weights = weights.^(-1);
weightVec = [];
for lab = 1:length(labels)
  for labidx = 1:length(alphabetical_labels)
    if labels(lab) == alphabetical_labels(labidx)
      weightVec(lab) = weights(labidx);
    end
  end
end
trainFiles = trainStore.Files;
bootstrapSize = round(length(trainFiles) * bootstrap_factor);
Bootstrap = datasample(trainFiles, bootstrapSize, 'Weights', weightVec);
bootStrapTrainStore = imageDatastore(Bootstrap, 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

% 訓練データの整形
augmentedResolution = [40 40];
augmenter = imageDataAugmenter('RandRotation', [-10 10]);
trainStoreAug = augmentedImageDatastore(augmentedResolution, bootStrapTrainStore, 'DataAugmentation', augmenter);

% Deep Neural Netの作成
% opts = trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs'...
%     ,200, 'MiniBatchSize',256, 'Plots','training-progress');

opts = trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs'...
    ,200, 'MiniBatchSize',256);
layers = [
    imageInputLayer([40 40 1],"Name","imageinput")
    convolution2dLayer([7 7],8,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Stride",[2 2])
    convolution2dLayer([3 3],16,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")
    maxPooling2dLayer([2 2],"Name","maxpool_4","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    maxPooling2dLayer([2 2],"Name","maxpool_5","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_6")
    fullyConnectedLayer(128,"Name","fc_2")
    reluLayer("Name","relu")
    fullyConnectedLayer(5,"Name","fc_1")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

myNet = trainNetwork(trainStoreAug,layers,opts);

% テストデータの評価
predictedLabels = classify(myNet, testStore);
accuracy = mean(predictedLabels == testStore.Labels);

end